import os
import world
import numpy as np
import gensim
from gensim.models import Word2Vec
from tqdm import tqdm
from utils import timer, minibatch, shuffle
from data import count_freq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.externals import joblib
import joblib
import torch

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
        return np.argmax(pc)




class word2vec(BasicClassifier):
    def __init__(self,
                 name,
                 labels,
                 dim=50):
        self.labels = labels
        self.whole = []
        for l in self.labels:
            self.whole.extend(l)
        print(len(self.whole))
        self.model = None
        self.name = name
        self.dim = dim

    def train(self, retrain=False):
        if os.path.exists(f"./checkpoints/corpus-{self.name}-{self.dim}.model") and not retrain:
            print("loading", f"./checkpoints/corpus-{self.name}-{self.dim}.model")
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                f"./checkpoints/corpus-{self.name}-{self.dim}.model", binary=False)
        else:
            self.model = Word2Vec(self.whole, sg=1, size=self.dim, window=5,
                                iter=10, min_count=5, negative=3,
                                sample=1e-3, hs=1)
            self.model.wv.save_word2vec_format(f"./checkpoints/corpus-{self.name}-{self.dim}.model", binary=False)

    def train_with_pretrain(self, word_file):
        self.model = Word2Vec(size=300, sg=1, window=3, min_count=3, )
        self.model.build_vocab(self.whole)
        other = gensim.models.KeyedVectors.load_word2vec_format(word_file, binary=False)

        # self.model.build_vocab([list(other.vocab.keys())], update=True)
        self.model.intersect_word2vec_format(word_file, binary=False, lockf=0.)
        self.model.train(self.whole,
                         total_examples=self.model.corpus_count,
                         epochs=5)
        self.model.wv.save_word2vec_format(f'./checkpoints/ensemble-{self.name}-300.model',binary=False)

    def __getitem__(self, index):
        try:
            return self.model[index]
        except KeyError:
            return None

class LR(torch.nn.Module):
    def __init__(self,
                 name,
                 datas,
                 input_dim=50,
                 label_num=4,
                 batch_size = 100):
        super(LR, self).__init__()
        self.name = name
        self.datas = datas
        self.batch = batch_size
        self.w2v = word2vec(name, datas, dim=input_dim)
        self.w2v.train()
        self.label_num = label_num
        self.M = torch.nn.Linear(input_dim, label_num)
        self.cri = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.M.parameters(), lr=0.01, weight_decay=1e-5)
        print(list(self.M.parameters()))

    def train(self, epochs=2000):
        datas = self.datas
        label_num = len(datas)
        labels = []
        for i in range(label_num):
            datas[i] = list(filter(lambda x: len(x) > 0, datas[i]))
        for i in range(label_num):
            labels.extend([i]*len(datas[i]))
        train_data = []
        for i in range(label_num):
            train_data.extend(datas[i])
        features = self.word2num(train_data)
        features = torch.from_numpy(features)
        labels = torch.LongTensor(labels)
        (features,
         labels) = shuffle(features, labels)
        print(labels[:10])

        for epoch in range(epochs):
            for song_batch, label_batch in minibatch(features, labels, batch_size=self.batch):
                pred = self.M(song_batch)
                loss = self.cri(pred, label_batch)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                print(f"loss: {loss.item()} \r", end='')

    def word2num(self, datas):
        num_feature = []
        for song in tqdm(datas):
            v = self.song2num(song)
            num_feature.append(v)
        return np.array(num_feature)

    def song2num(self, song):
        v = 0.
        for word in song:
            w = self.w2v[word]
            if w is not None:
                v += w
        if isinstance(v, float):
            return None
        v = v/len(song)
        return v

    def predict(self, song):
        song_numpy = self.song2num(song)
        if song_numpy is None:
            return np.random.randint(self.label_num)
        feature = torch.from_numpy(song_numpy).unsqueeze(0)
        pred = self.M(feature).squeeze()
        # print(pred)
        return torch.argmax(pred).item()

    def predict_batch(self, datas):
        feature = self.word2num(datas)
        pred = self.M(feature)
        return torch.argmax(pred, dim=1).numpy()

class RFClassifier(BasicClassifier):
    def __init__(self,
                 name,
                 datas,
                 n_estimators=100,
                 word_dim = 20):
        self.n = n_estimators
        self.name = name
        self.datas = datas
        self.label_num = len(datas)
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.w2v = word2vec(name, datas, dim=word_dim)
        self.w2v.train()

    def train(self):
        if not os.path.exists('./rf.m'):
            datas = self.datas
            label_num = len(datas)
            labels = []
            for i in range(label_num):
                datas[i] = list(filter(lambda x: len(x) > 0, datas[i]))
            for i in range(label_num):
                labels.extend([i] * len(datas[i]))
            labels = np.array(labels)
            train_data = []
            for i in range(label_num):
                train_data.extend(datas[i])
            features = self.word2num(train_data)

            index = np.arange(len(labels))
            np.random.shuffle(index)
            features = features[index]
            labels = labels[index]

            self.model.fit(features, labels)
            joblib.dump(self.model, "rf.m")
        else:
            print("load rf.m")
            self.model = joblib.load("rf.m")

    def word2num(self, datas):
        num_feature = []
        for song in tqdm(datas):
            v = self.song2num(song)
            num_feature.append(v)
        return np.array(num_feature)

    def song2num(self, song):
        v = 0.
        for word in song:
            w = self.w2v[word]
            if w is not None:
                v += w
        if isinstance(v, float):
            return None
        v = v/len(song)
        return v

    def predict(self, song):
        song = self.song2num(song)
        if song is None:
            return np.random.randint(self.label_num)
        song = song.reshape(1, -1)
        return self.model.predict(song)[0]


class LRClassifier(BasicClassifier):
    def __init__(self, name, datas, n_estimators=100, word_dim=20):
        self.n = n_estimators
        self.name = name
        self.datas = datas
        self.label_num = len(datas)
        self.model = LogisticRegression(random_state=0)
        self.w2v = word2vec(name, datas, dim=word_dim)
        self.w2v.train()

    def train(self):
        if not os.path.exists('./lr.m'):
            datas = self.datas
            label_num = len(datas)
            labels = []
            for i in range(label_num):
                datas[i] = list(filter(lambda x: len(x) > 0, datas[i]))
            for i in range(label_num):
                labels.extend([i] * len(datas[i]))
            labels = np.array(labels)
            train_data = []
            for i in range(label_num):
                train_data.extend(datas[i])
            features = self.word2num(train_data)

            index = np.arange(len(labels))
            np.random.shuffle(index)
            features = features[index]
            labels = labels[index]

            self.model.fit(features, labels)
            joblib.dump(self.model, "lr.m")
        else:
            print("load lr.m")
            self.model = joblib.load("lr.m")

    def word2num(self, datas):
        num_feature = []
        for song in tqdm(datas):
            v = self.song2num(song)
            num_feature.append(v)
        return np.array(num_feature)

    def song2num(self, song):
        v = 0.
        for word in song:
            w = self.w2v[word]
            if w is not None:
                v += w
        if isinstance(v, float):
            return None
        v = v / len(song)
        return v

    def predict(self, song):
        song = self.song2num(song)
        if song is None:
            return np.random.randint(self.label_num)
        song = song.reshape(1, -1)
        return self.model.predict(song)[0]


class SVMClassifier(BasicClassifier):
    def __init__(self,
                 name,
                 data):
        pass