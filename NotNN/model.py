import numpy as np
from tqdm import tqdm
from utils import timer
from data import count_freq


class BasicClassifier:
    def train(self, *args, **kwargs):
        """Manage the training process
        """
        raise NotImplementedError
    def predict(self, *arg, **kwargs):
        """return labels based on the current model
        """
        raise NotImplementedError


class NaiveBayes(BasicClassifier):
    def __init__(self,
                 vocab : set):
        self.vocab = vocab
        self.num = {}

    def train(self, *labels):
        freq, index, reverse_index = count_freq(self.vocab, *labels)
        args = np.argsort(freq.sum(1))[::-1]
        # for a in args[:200]:
        #     print(reverse_index[a], freq.sum(1)[a])
        # import matplotlib.pyplot as plt
        # plt.scatter(range(len(reverse_index)), np.sort(freq.sum(1))[::-1])
        # plt.show()

        V = len(self.vocab)
        nc = freq.sum(0)

        freq = (freq + 1)/(nc + V)
        print(freq.sum(0))

        classes = np.array([len(l) for l in labels]).astype('float')

        self.num['P(c)'] = classes/classes.sum()
        self.num["P(w|c)"] = freq
        self.num['C'] = len(labels)
        self.num['index'] = index
        self.num['reverse_index'] = reverse_index

    def predict(self, song):
        pc = self.num['P(c)']
        for word in song:
            w_index = self.num["index"].get(word, None)
            if w_index is not None:
                pc = pc*self.num['P(w|c)'][w_index]
                pc = pc/pc.max()
            # print(pc)
        # print(pc)
        return pc