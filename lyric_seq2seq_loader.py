# coding: utf-8

import os
import jieba
from sklearn.model_selection import train_test_split


def segment(s):
    return ' '.join(jieba.cut(s))


def open_file(filename, mode="r"):
    return open(filename, mode=mode, encoding='utf-8', errors='ignore')


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train, self.test = self.tokenize(path)

    def tokenize(self, path):
        assert os.path.exists(path)

        lyric_all = []
        for filename in sorted(os.listdir(path)):
            try:
                lyric = []
                for line in open_file(os.path.join(path, filename)).readlines():
                    if len(line.strip()) == 0:
                        continue
                    words = '<sos> ' + segment(line.strip()) + ' <eos>'
                    words = words.split()
                    for word in words:
                        self.dictionary.add_word(word)
                    lyric.append(words)
                lyric_all.append(lyric)
            except Exception as e:
                print(e)

        print("共读取到歌词数：", len(lyric_all))
        print("词汇量：", len(self.dictionary))

        data = []
        for lyric in lyric_all:
            lyric = [self.words_to_ids(words) for words in lyric]
            data.extend(list(zip(lyric[:-1], lyric[1:])))

        print("数据总量：", len(data))

        data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)
        return data_train, data_test

    def words_to_ids(self, words):
        return [self.dictionary.word2idx[x] for x in words]

    def ids_to_words(self, ids):
        return [self.dictionary.idx2word[x] for x in ids]

