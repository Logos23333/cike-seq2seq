#coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 20

# Configure models
model_name = 'cb_model'

# attn_model = 'dot'
attn_model = 'general'

hidden_size = 512
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 32

save_dir = "../model"
corpus_name = "test"

is_clip = False
clip = 50.0

n_iteration = 2000
print_every = 10
check_every = 10
break_count = 10

learning_rate = 0.0001
teacher_forcing_ratio = 0.5

#decoder_learning_ratio = 5.0

encoder_embedding_size = 256
decoder_embedding_size = 256

beam_size = 10