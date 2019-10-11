#coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

import Params


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {Params.PAD_token: "PAD", Params.SOS_token: "SOS", Params.EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {Params.PAD_token: "PAD", Params.SOS_token: "SOS", Params.EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        return outputs, hidden


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)



class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding_size, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        # self.concat = nn.Linear(embedding_size + hidden_size, embedding_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        attn_weights = self.attn(last_hidden[self.n_layers - 1], encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.squeeze(1)
        embedded = embedded.squeeze(0)
        concat_input = torch.cat((embedded, context), 1).unsqueeze(0)

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(concat_input, last_hidden)

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)

        # Predict next word using Luong eq. 6
        output = self.out(rnn_output)
        output = F.log_softmax(output, dim=1)

        # Return output and final hidden state
        return output, hidden


class Candidate(object):
    def __init__(self, decoder_hidden):
        self.all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        self.all_scores = torch.zeros([0], device=device)
        self.total_score = self.all_scores.sum().item()

        self.decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * Params.SOS_token
        self.decoder_hidden = decoder_hidden

        self.size = 0
        self.is_finished = False

    def copy(self):
        spring = Candidate(self.decoder_hidden)
        spring.all_tokens = self.all_tokens.clone().detach()
        spring.all_scores = self.all_scores.clone().detach()
        spring.total_score = spring.all_scores.sum().item()

        spring.decoder_input = self.decoder_input.clone().detach()
        spring.decoder_hidden = self.decoder_hidden.clone().detach()

        spring.size = self.size
        spring.is_finished = self.is_finished

        return spring

    def update(self, token, score, decoder_hidden, max_length):

        token = torch.unsqueeze(token, 0)
        score = torch.unsqueeze(score, 0)

        self.all_tokens = torch.cat((self.all_tokens, token), dim=0)
        self.all_scores = torch.cat((self.all_scores, score), dim=0)
        self.total_score = self.all_scores.sum().item()
        self.decoder_input = torch.unsqueeze(token, 0)
        self.decoder_hidden = decoder_hidden
        self.size = self.all_tokens.size()[0]

        if(self.size > max_length or token.item() == Params.EOS_token):
            self.is_finished = True

    def __lt__(self, other):
        return self.total_score < other.total_score


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:Params.decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * Params.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, beam_size=Params.beam_size):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.k = beam_size

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:Params.decoder_n_layers]

        # Initialize decoder input with SOS_token
        candidate_list = [Candidate(decoder_hidden)]

        # Iteratively decode one word token at a time
        while(True):
            spring_list = []
            end_flag = True
            for candidate in candidate_list:

                if(not candidate.is_finished):
                    # Not completely finished yet
                    end_flag = False

                    # Forward pass through decoder
                    decoder_output, decoder_hidden = self.decoder(candidate.decoder_input, candidate.decoder_hidden, encoder_outputs)

                    # Obtain most likely word token and its softmax score
                    decoder_scores, decoder_inputs = torch.topk(decoder_output, self.k, dim=1)

                    decoder_scores = torch.squeeze(decoder_scores, 0)
                    decoder_inputs = torch.squeeze(decoder_inputs, 0)

                    for i in range(self.k):
                        spring = candidate.copy()

                        spring.update(decoder_inputs[i], decoder_scores[i], decoder_hidden, max_length)
                        spring_list.append(spring)
                else:
                    spring_list.append(candidate)

            if(end_flag):
                break

            spring_list.sort(reverse=True)
            candidate_list = spring_list[0: self.k]

        candidate_list.sort(reverse=True)
        candidate = candidate_list[0]
        return candidate.all_tokens, candidate.all_scores