#coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim

import os
import sys
import math
import random

import Model
import Params
import FileUtil
import DataUtil
import EvaluateUtil


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


TRAIN_FILE = '../data/train.data'
VALID_FILE = '../data/valid.data'
TEST_FILE  = '../data/test.data'

# TRAIN_FILE = "../data/conversation.data"
# VALID_FILE = "../data/conversation.data"
# TEST_FILE  = "../data/conversation.data"

pairs = FileUtil.read_pairs(TRAIN_FILE)
inputVoc, outputVoc = DataUtil.prepareVoc(pairs)

dev_pairs = FileUtil.read_pairs(VALID_FILE)
test_pairs = FileUtil.read_pairs(TEST_FILE)


def train(input_variable, lengths, target_variable, max_target_len, criterion, encoder, decoder,
          encoder_optimizer, decoder_optimizer, clip):
    global device
    
    batch_size = input_variable.size()[1]
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)

    # Initialize variables
    loss = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[Params.SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < Params.teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            t_loss = criterion(decoder_output, target_variable[t])
            loss += t_loss

    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            t_loss = criterion(decoder_output, target_variable[t])
            loss += t_loss

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    if(Params.is_clip):
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def trainIters(model_name, inputVoc, outputVoc, pairs, dev_pairs,
               encoder, decoder, encoder_optimizer, decoder_optimizer,
               input_embedding, output_embedding, encoder_n_layers, decoder_n_layers,
               save_dir, n_iteration, batch_size, print_every, check_every, clip, corpus_name, loadFilename):

    weight = torch.ones(outputVoc.num_words)
    weight[Params.PAD_token] = 0

    criterion = nn.NLLLoss(ignore_index=Params.PAD_token)
    # criterion = nn.NLLLoss(weight=weight, ignore_index=Params.PAD_token)

    training_batches = []
    batch_num = int(math.ceil(len(pairs) / batch_size))
    print("Batch Number (Train):", batch_num)
    for i in range(batch_num):
        batch_data = DataUtil.batch2TrainData(inputVoc, outputVoc, pairs[i * batch_size: (i + 1) * batch_size])
        training_batches.append(batch_data)

    dev_batches = []
    dev_batch_num = int(math.ceil(len(dev_pairs) / batch_size))

    for i in range(dev_batch_num):
        dev_batch_data = DataUtil.batch2TrainData(inputVoc, outputVoc, dev_pairs[i * batch_size: (i + 1) * batch_size])
        dev_batches.append(dev_batch_data)


    # Initializations
    print('Initializing ...')
    start_iteration = 1
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    print_loss = 0
    larger_count = 0
    best_dev_ppl = sys.maxsize

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[(iteration - 1) % batch_num]
        # Extract fields from batch
        input_variable, lengths, target_variable, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, max_target_len, criterion,
                     encoder, decoder, encoder_optimizer, decoder_optimizer, clip)

        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % check_every == 0):


            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, Params.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'input_voc_dict': inputVoc.__dict__,
                'output_voc_dict': outputVoc.__dict__,
                'input_embedding': input_embedding.state_dict(),
                'output_embedding': output_embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

            encoder.eval()
            decoder.eval()

            dev_ppl = EvaluateUtil.calc_ppl(encoder, decoder, outputVoc.num_words, dev_batches, Params.PAD_token)

            if (dev_ppl < best_dev_ppl):
                best_dev_ppl = dev_ppl

                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'input_voc_dict': inputVoc.__dict__,
                    'output_voc_dict': outputVoc.__dict__,
                    'input_embedding': input_embedding.state_dict(),
                    'output_embedding': output_embedding.state_dict()
                }, os.path.join(directory, '{}.tar'.format('best_ppl')))

                larger_count = 0

            else:
                larger_count += 1

            print("#CHECK POINT# Iteration: {}; Best PPL: {:.4f}; Current PPL: {:.4f}; Larger count: {}".format(iteration, best_dev_ppl, dev_ppl, larger_count))

            encoder.train()
            decoder.train()

        if(larger_count > Params.break_count):
            print("BREAK: Meet Break Count")
            break


# Set checkpoint to load from; set to None if starting from scratch
# loadFilename = "../model/cb_model/test/2-2_512/100_checkpoint.tar"
loadFilename = None

print('Building encoder and decoder ...')

# Initialize word embeddings
input_embedding = nn.Embedding(inputVoc.num_words, Params.encoder_embedding_size)
output_embedding = nn.Embedding(outputVoc.num_words, Params.decoder_embedding_size)

# Initialize encoder & decoder models
encoder = Model.EncoderRNN(Params.encoder_embedding_size, Params.hidden_size,
                           input_embedding, Params.encoder_n_layers, Params.dropout)

decoder = Model.LuongAttnDecoderRNN(Params.attn_model, Params.decoder_embedding_size, output_embedding,
                                    Params.hidden_size, outputVoc.num_words, Params.decoder_n_layers, Params.dropout)

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    input_embedding_sd = checkpoint['input_embedding']
    output_embedding_sd = checkpoint['output_embedding']
    inputVoc.__dict__ = checkpoint['input_voc_dict']
    outputVoc.__dict__ = checkpoint['output_voc_dict']

    input_embedding.load_state_dict(input_embedding_sd)
    output_embedding.load_state_dict(output_embedding_sd)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)



# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')

# encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=Params.learning_rate)
# decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=Params.learning_rate)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=Params.learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=Params.learning_rate)

# encoder_optimizer = optim.SGD(encoder.parameters(), lr=Params.learning_rate)
# decoder_optimizer = optim.SGD(decoder.parameters(), lr=Params.learning_rate)

if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations

print("Starting Training!")
trainIters(Params.model_name, inputVoc, outputVoc, pairs, dev_pairs,
           encoder, decoder, encoder_optimizer, decoder_optimizer,
           input_embedding, output_embedding,
           Params.encoder_n_layers, Params.decoder_n_layers,
           Params.save_dir, Params.n_iteration, Params.batch_size,
           Params.print_every, Params.check_every,
           Params.clip, Params.corpus_name, loadFilename)

'''
bestFileName = "../model/cb_model/test/2-2_512/best_ppl.tar"
# bestFileName = "../model/cb_model/test/2-2_128/200_checkpoint.tar"
# Load model if a bestFileName is provided
if bestFileName:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(bestFileName)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    input_embedding_sd = checkpoint['input_embedding']
    output_embedding_sd = checkpoint['output_embedding']
    inputVoc.__dict__ = checkpoint['input_voc_dict']
    outputVoc.__dict__ = checkpoint['output_voc_dict']

    input_embedding.load_state_dict(input_embedding_sd)
    output_embedding.load_state_dict(output_embedding_sd)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
'''


encoder.eval()
decoder.eval()

# Initialize search module
# searcher = Model.GreedySearchDecoder(encoder, decoder)
searcher = Model.BeamSearchDecoder(encoder, decoder)

result_list = EvaluateUtil.evaluate_data(searcher, inputVoc, outputVoc, test_pairs)

for i in result_list:
    for j in i:
        print(j)
    print()