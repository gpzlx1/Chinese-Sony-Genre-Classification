import os
from tqdm import tqdm
import time
import pickle as pkl
import re
import random
import torch

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def read_process_dataset(train_dataset_paths, val_dataset_paths, word_level=False):
    if train_dataset_paths is None:
        train_dataset_paths = []
    if val_dataset_paths is None:
        val_dataset_paths = []
    train = []
    val = []
    for p in train_dataset_paths:
        songs = None
        with open(p, 'r') as f:
            songs = f.read()

        if word_level:
            songs = re.sub('[ ]+', ' ', songs)
        else:
            songs = re.sub('[ ]+', '', songs)

        songs = songs.split('\n')
        train.append(songs)

    for p in val_dataset_paths:
        songs = None
        with open(p, 'r') as f:
            songs = f.read()

        if word_level:
            songs = re.sub('[ ]+', ' ', songs)
        else:
            songs = re.sub('[ ]+', '', songs)

        songs = songs.split('\n')
        val.append(songs)

    return train, val



def build_vocab(train_dataset, tokennizer, max_size, min_freq):
    vocab_dic = {}
    if train_dataset is None:
        train_dataset = []

    dataset = train_dataset
    total_songs = sum([len(d) for d in dataset])
    print("total songs num:", total_songs)

    for part_dataset in dataset:
        for line in part_dataset:
            for t in tokennizer(line):
                vocab_dic[t] = vocab_dic.get(t, 0) + 1

    avg_song_length = sum([_ for _ in vocab_dic.values()]) / total_songs
    print("AVG song length:", avg_song_length)
    
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic



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

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

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


    def _load_dataset(dataset_list, pad_size=250):
        if dataset_list is None:
            dataset_list = []
        contents  = []
        for index, d in enumerate(dataset_list):
            for song in d:
                token = tokenizer(song)
                label = index
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
                contents.append((words_line, int(index), seq_len))
        random.shuffle(contents)
        return contents



    train = _load_dataset(train)
    val = _load_dataset(val)

    return vocab, train, val




if __name__ == "__main__":
    paths = [
        '/mnt/c/Users/gpzlx/Desktop/netease/split-data/data/SPLIT/train/ancient.txt', 
        '/mnt/c/Users/gpzlx/Desktop/netease/split-data/data/SPLIT/train/ballad.txt',
        '/mnt/c/Users/gpzlx/Desktop/netease/split-data/data/SPLIT/train/rap.txt',
        '/mnt/c/Users/gpzlx/Desktop/netease/split-data/data/SPLIT/train/rock.txt'
    ]
    vocab, train, val = build_dataset(paths, None, 'cache', True)
    for i in DatasetIterater(train, 32, 'cpu'):
        print(i)
        break