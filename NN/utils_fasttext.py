import os 
import torch
import numpy as np
import re
import random
import pickle as pkl

from utils import read_process_dataset, build_vocab

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_dataset(train_paths, val_paths, word_level=False):

    if word_level:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    vocab = None
    train = None
    val = None

    
    train, val = read_process_dataset(train_paths, val_paths, word_level)
    vocab = build_vocab(train, tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    print(f"Vocab size: {len(vocab)}")


    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def _load_dataset(dataset_list, pad_size=250):
        if dataset_list is None:
            dataset_list = []
        contents = []
        for index, d in enumerate(dataset_list):
            for song in d:
                token = tokenizer(song)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                    seq_len = pad_size

                # word to id
                words_line = []
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                
                # fasttext ngram
                buckets = 250499
                bigram = []
                trigram = []
                # ngram
                for i in range(pad_size):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))

                contents.append((words_line, int(index), seq_len, bigram, trigram))
        random.shuffle(contents)
        return contents
        

    train = _load_dataset(train)
    val = _load_dataset(val)

    return vocab, train, val


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        import random 
        random.shuffle(self.batches)
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches



if __name__ == "__main__":
    paths = [
        '/mnt/c/Users/gpzlx/Desktop/netease/split-data/data/SPLIT/train/ancient.txt', 
        '/mnt/c/Users/gpzlx/Desktop/netease/split-data/data/SPLIT/train/ballad.txt',
        '/mnt/c/Users/gpzlx/Desktop/netease/split-data/data/SPLIT/train/rap.txt',
        '/mnt/c/Users/gpzlx/Desktop/netease/split-data/data/SPLIT/train/rock.txt'
    ]
    vocab, train, val = build_dataset(paths, None, True)
    for i in DatasetIterater(train, 32, 'cpu'):
        print(i)
        break