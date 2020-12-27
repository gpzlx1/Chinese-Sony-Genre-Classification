import world
import jieba
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join, isdir
from os import listdir

label_paths = [
    p for p in listdir(world.DATA)
    if isdir(join(world.DATA, p))
]
with open(join(world.DATA, 'stopword.txt'), 'r') as f:
    stopwords = f.readlines()
    stopwords = [word.strip() for word in stopwords]
    stopwords = set(stopwords)


def count_songs(avaliable_paths=label_paths):
    avaliable_paths = [join(world.DATA, p) for p in avaliable_paths]
    counts = []
    for l in avaliable_paths:
        song_paths = listdir(l)
        songs = [song.endswith('txt') for song in song_paths]
        counts.append(len(songs))
    return counts


def get_one_song(song):
    if not song.endswith('txt'):
        return
    with open(song, 'r') as f:
        lyrics = [line.strip() for line in f.readlines() if line.strip() != '']
    not_token = "%%%".join(lyrics)
    lyrics = " ".join(lyrics)
    token = jieba.lcut(lyrics, cut_all=False, HMM=True)
    token = [t for t in token if t not in stopwords]
    lyrics = " ".join(token)
    return lyrics, not_token

def merge2one(avaliable_paths=label_paths,
              name=None,
              core=8,
              split=False,
              ratio=0.7):
    """merge all the training lyrics into one corpus
    """
    from tqdm import tqdm
    from multiprocessing import Pool
    from random import shuffle
    name = name or 'corpus'
    file2save = join(world.DATA, f'{name}.txt')
    avaliable_paths = [join(world.DATA, p) for p in avaliable_paths]
    corpus = open(file2save, 'w')
    if split:
        corpus_test = open(join(world.DATA, f'{name}-test.txt'), 'w')
        corpus_all = open(join(world.DATA, f'{name}-all.txt'), 'w')
        corpus_all_test = open(join(world.DATA, f'{name}-all-test.txt'), 'w')

    for label_path in tqdm(avaliable_paths):
        song_paths = listdir(label_path)
        song_paths = [join(label_path, p) for p in song_paths]
        with Pool(core) as p:
            results = p.map(get_one_song, song_paths)
        results = list(filter(lambda x: x is not None, results))
        if split:
            shuffle(results)
            not_token = [r[1] for r in results]
            results = [r[0] for r in results]

            train_num = len(results) - 1000

            train_lyc = '\n'.join(results[:train_num])
            test_lyc = '\n'.join(results[train_num:])
            corpus.write(train_lyc)
            corpus_test.write(test_lyc)

            all_lyc = '\n'.join(not_token[:train_num])
            all_lyc_test = '\n'.join(not_token[train_num:])
            corpus_all.write(all_lyc)
            corpus_all_test.write(all_lyc_test)
        else:
            results = [r[0] for r in results]
            lyrics = "\n".join(results)
            corpus.write(lyrics)
    corpus.close()
    if split:
        corpus_test.close()
        corpus_all.close()
        corpus_all_test.close()

def _get_vocab_one(one):
    result = set()
    all_token = []
    for song in one:
        all_token.extend(song)
    return set(all_token)


def get_vocab(*labels, core=4):
    from multiprocessing import Pool

    total_vocab = set()
    with Pool(core) as p:
        results = p.map(_get_vocab_one, labels)
    for r in results:
        total_vocab = total_vocab.union(r)
    return total_vocab

def count_freq(vocab, *labels):
    """helper to count the vocab's freq in diff labels

    Args:
        vocab (set)
        *labels (list) : token of the whole class, ['one', 'two',...]
    Return:
        freq (ndarray)
        word_index (dict)
    """
    print(len(labels))
    index = {
        word : i
        for i, word in enumerate(vocab)
    }
    reverse_index = {
        i : word
        for word, i in index.items()
    }
    reverse_index = [
        reverse_index[i]
        for i in range(len(reverse_index))
    ]
    freq = np.zeros((len(vocab), len(labels)))
    print('here')
    for i, label in tqdm(enumerate(labels)):
        for song in label:
            for word in song:
                w_index = index.get(word, None)
                if w_index is not None:
                    freq[w_index, i] += 1
    return freq, index, reverse_index

def load_data():
    labels = ['ancient', 'ballad', 'rap', 'rock']
    corpus = [
        join(join(world.DATA, 'train'), f"{name}.txt") for name in labels
    ]
    corpus_test = [
        join(join(world.DATA, 'test'), f"{name}-test.txt") for name in labels
    ]
    datas = []
    for c in corpus:
        with open(c, 'r') as f:
            data = f.readlines()
            data = [d.split() for d in data]
        datas.append(data)

    test_datas = []
    for c in corpus_test:
        with open(c, 'r') as f:
            data = f.readlines()
            data = [d.split() for d in data]
        test_datas.append(data)
    vocab = get_vocab(*datas)
    # with open('vocab.txt', 'w') as f:
    #     v_w = list(vocab)
    #     v_w = '\n'.join(v_w)
    #     f.writelines(v_w)
    with open('vocab.txt', 'r') as f:
        vocab = f.readlines()
        vocab = [v.strip() for v in vocab]
        vocab = set(vocab)
    vocab = vocab
    return datas, test_datas, vocab, labels


def load_data_all():
    labels = ['ancient', 'ballad', 'rap', 'rock']
    corpus = [
        join(join(world.DATA, 'train'), f"{name}-all.txt") for name in labels
    ]
    corpus_test = [
        join(join(world.DATA, 'test'), f"{name}-all-test.txt") for name in labels
    ]
    datas = []
    for c in corpus:
        with open(c, 'r') as f:
            data = f.readlines()
            data = [d.split("%%%") for d in data]
        datas.append(data)

    test_datas = []
    for c in corpus_test:
        with open(c, 'r') as f:
            data = f.readlines()
            data = [d.split("%%%") for d in data]
        test_datas.append(data)
    datas = parse2word(datas)
    test_datas = parse2word(test_datas)
    vocab = get_vocab(*datas)
    # with open('vocab.txt', 'w') as f:
    #     v_w = list(vocab)
    #     v_w = '\n'.join(v_w)
    #     f.writelines(v_w)
    # with open('vocab.txt', 'r') as f:
    #     vocab = f.readlines()
    #     vocab = [v.strip() for v in vocab]
    #     vocab = set(vocab)
    vocab = vocab
    return datas, test_datas, vocab, labels

def parse2word(datas):
    for i in range(len(datas)):
        new_data = [' '.join(song) for song in datas[i]]
        new_data = [list(song) for song in new_data]
        datas[i] = new_data
    return datas

if __name__ == "__main__":
    merge2one(avaliable_paths=['../data/ancient-songs'],
              name="ancient",
              split=0.7)
