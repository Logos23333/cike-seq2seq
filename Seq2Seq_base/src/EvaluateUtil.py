#coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

import math
import numpy as np
from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

import Params
import DataUtil


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def evaluate(searcher, inputVoc, outputVoc, sentence, max_length=Params.MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [DataUtil.indexesFromSentence(inputVoc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [outputVoc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_data(searcher, inputVoc, outputVoc, data_pairs):

    result_list = []

    for i in range(len(data_pairs)):
        pair = data_pairs[i]
        try:
            # Get input sentence and target sentence
            input_sentence = pair[0]
            target_sentence = pair[1]

            # Normalize sentence
            input_sentence = DataUtil.normalizeString(input_sentence)

            # Evaluate sentence
            output_words = evaluate(searcher, inputVoc, outputVoc, input_sentence)

            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            output_sentence = ' '.join(output_words)

            result_list.append([input_sentence, target_sentence, output_sentence])

        except KeyError:
            print("Error: Encountered unknown word.")

    return result_list



def evaluateInput(searcher, inputVoc, outputVoc):

    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = DataUtil.normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, inputVoc, outputVoc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def calc_nll(logits, targets, weight):
    batch_size = logits.size(0)
    nll = F.nll_loss(input=logits.view(-1, logits.size(-1)),
                     target=targets.contiguous().view(-1),
                     ignore_index=Params.PAD_token,
                     weight=weight,
                     reduction='none')
    nll = nll.view(batch_size, -1).sum(dim=1)
    return nll


def perplexity(logits, targets, weight=None, padding_idx=None, batch_size=64):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    if weight is None and padding_idx is not None:
        weight = torch.ones(logits.size(-1))
        weight[padding_idx] = 0

    nll_list = []
    for i in range(math.ceil(logits.size(0) / batch_size)):
        logits_ = logits[i * batch_size: (i + 1) * batch_size, :, :]
        targets_ = targets[i * batch_size: (i + 1) * batch_size, :]

        nll = calc_nll(logits_, targets_, weight)
        nll_list.append(nll.sum())

    nll_tensor = torch.tensor(nll_list)
    nll_sum = nll_tensor.sum()

    if padding_idx is not None:
        word_cnt = targets.ne(padding_idx).float().sum()
    else:
        word_cnt = targets.size()[0] * targets.size()[1]

    nll_sum = nll_sum / word_cnt
    ppl = nll_sum.exp()
    return ppl


def calc_ppl(encoder, decoder, output_size, dev_batches, padding_idx):

    weight = torch.ones(output_size)
    weight[padding_idx] = 0
    weight = weight.to(device)

    nll_list = []
    word_cnt = 0
    print(len(dev_batches))
    for i in range(len(dev_batches)):
        batch_data = dev_batches[i]

        input_variable, lengths, target_variable, max_target_len = batch_data

        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)

        batch_size = input_variable.size()[1]

        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[Params.SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        for t in range(max_target_len):
            # Forward pass through decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            nll = calc_nll(decoder_output, target_variable[t], weight)
            nll_list.append(nll.sum())

            word_cnt += target_variable[t].ne(padding_idx).float().sum()

    nll_tensor = torch.tensor(nll_list)
    nll_sum = nll_tensor.sum()
    nll_sum = nll_sum / word_cnt
    ppl = nll_sum.exp()

    return ppl


def bleu(hyps, refs):
    """
    bleu
    """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        print(hyp)
        print(ref)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                # smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                # smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)

    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def distinct(seqs):
    """
    distinct
    """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))

        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2
