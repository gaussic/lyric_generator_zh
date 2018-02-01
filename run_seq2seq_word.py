# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import random

from lyric_seq2seq_loader import Corpus
from lyric_seq2seq_word import LGConfig, Encoder, Decoder

use_cuda = torch.cuda.is_available()


def variable_from_pair(pair):
    input_var = Variable(torch.LongTensor(pair[0]))
    target_var = Variable(torch.LongTensor(pair[1]))

    if use_cuda:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
    return input_var, target_var


def train(input_var, target_var, encoder, decoder, enc_optim, dec_optim, criterion, corpus):
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    loss = 0

    encoder.train()
    decoder.train()

    target_len = target_var.size(0)
    enc_outputs, enc_hidden = encoder(input_var)

    dec_input = Variable(torch.LongTensor([corpus.dictionary.word2idx['<sos>']]))
    dec_hidden = enc_hidden
    if use_cuda:
        dec_input = dec_input.cuda()

    for di in range(target_len):
        dec_output, dec_hidden = decoder(dec_input, dec_hidden)
        loss += criterion(dec_output, target_var[di])
        dec_input = target_var[di]

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), 5)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 5)
    enc_optim.step()
    dec_optim.step()

    return loss.data[0] / target_len


def evaluate(input_var, target_var, encoder, decoder, criterion, corpus):
    loss = 0

    encoder.eval()
    decoder.eval()

    target_len = target_var.size(0)
    enc_outputs, enc_hidden = encoder(input_var)

    dec_input = Variable(torch.LongTensor([corpus.dictionary.word2idx['<sos>']]))
    dec_hidden = enc_hidden
    if use_cuda:
        dec_input = dec_input.cuda()

    for di in range(target_len):
        dec_output, dec_hidden = decoder(dec_input, dec_hidden)
        loss += criterion(dec_output, target_var[di])
        dec_input = target_var[di]

    return loss.data[0], target_len


def evaluate_full(data):
    total_loss = 0.0
    total_len = 0
    for pair in data:
        input_var, target_var = variable_from_pair(pair)
        cur_loss, cur_len = evaluate(input_var, target_var, encoder, decoder, criterion, corpus)
        total_loss += cur_loss
        total_len += cur_len
    return total_loss / total_len


if __name__ == '__main__':
    corpus = Corpus('data/周杰伦_6452_clean')
    config = LGConfig(len(corpus.dictionary))
    embedding = nn.Embedding(config.vocab_size, config.embedding_size)
    encoder = Encoder(embedding, config)
    decoder = Decoder(embedding, config)

    enc_optim = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    dec_optim = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    for i in range(1, 10000+1):
        index = random.randint(0, len(corpus.train) - 1)
        input_var, target_var = variable_from_pair(corpus.train[index])
        total_loss += train(input_var, target_var, encoder, decoder, enc_optim, dec_optim, criterion, corpus)

        if i % 100 == 0:
            print('Train:', total_loss / 100)
            total_loss = 0.0
            print('Test:', evaluate_full(corpus.test))



