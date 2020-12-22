import world
import numpy as np
from tqdm import tqdm
from os.path import join


def NB_predict():
    from model import NaiveBayes
    from data import get_vocab

    labels = [
        'ancient', 'ballad', 'rap', 'rock'
    ]
    corpus = [
        join(join(world.DATA, 'train'),f"{name}.txt")
        for name in labels
    ]
    corpus_test = [
        join(join(world.DATA, 'test'), f"{name}-test.txt")
        for name in labels
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
    # vocab = get_vocab(*datas)
    with open('vocab.txt', 'r')  as f:
        vocab = f.readlines()
        vocab = [v.strip() for v in vocab]
        vocab = set(vocab)
    vocab = vocab

    print(f"VOCAB size: {len(vocab)}")

    classifier = NaiveBayes(vocab=vocab)
    classifier.train(*datas)
    confusion_matrix = np.zeros((len(labels), len(labels)))

    np.set_printoptions(precision=1)

    confusion_matrix = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        print("predict(train)", labels[i])
        for song in tqdm(test_datas[i]):
            p = classifier.predict(song)
            l = np.argmax(p)
            confusion_matrix[i, l] += 1
    print("test:\n", confusion_matrix / confusion_matrix.sum(1).reshape(4, -1) * 100)

    for i in range(len(labels)):
        print("predict(train)", labels[i])
        for song in tqdm(datas[i]):
            p = classifier.predict(song)
            l = np.argmax(p)
            confusion_matrix[i, l] += 1
    print("train:\n", confusion_matrix/confusion_matrix.sum(1).reshape(4, -1)*100)


if __name__ == "__main__":
    NB_predict()