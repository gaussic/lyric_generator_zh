# coding: utf-8

import torch
from torch.autograd import Variable
import random
from lyric_seq2seq_loader import Corpus
from lyric_seq2seq_word import *

corpus = Corpus('data/周杰伦_6452_clean')
config = LGConfig(len(corpus.dictionary))
embedding = nn.Embedding(config.vocab_size, config.embedding_size)
encoder = Encoder(embedding, config)

index = random.randint(0, len(corpus.train))

input_var = Variable(torch.LongTensor(corpus.train[index][0]))

output, hidden = encoder(input_var)



corpus = Corpus('data/周杰伦_6452_clean')
config = LGConfig(len(corpus.dictionary))
embedding = nn.Embedding(config.vocab_size, config.embedding_size)
encoder = Encoder(embedding, config)

decoder = Decoder(embedding, config)

enc_optim = optim.RMSprop(encoder.parameters(), lr=config.learning_rate, weight_decay=0.1)
dec_optim = optim.RMSprop(decoder.parameters(), lr=config.learning_rate, weight_decay=0.1)
criterion = nn.CrossEntropyLoss()

index = random.randint(0, len(corpus.train))
input_var = Variable(torch.LongTensor(corpus.train[index][0]))
target_var = Variable(torch.LongTensor(corpus.train[index][1]))

train(input_var, target_var, encoder, decoder, enc_optim, dec_optim, criterion, corpus)
