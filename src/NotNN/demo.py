import world
import numpy as np
from tqdm import tqdm
from os.path import join
import utils
from data import load_data, load_data_all

np.set_printoptions(precision=1)

def NB_predict():
    from model import NaiveBayes
    datas, test_datas, vocab, labels = load_data()

    print(f"VOCAB size: {len(vocab)}")

    classifier = NaiveBayes(vocab=vocab)
    classifier.train(*datas)

    np.set_printoptions(precision=1)

    mat, acc = utils.metrics(classifier, labels, *datas)
    utils.plot_confusion(mat, labels, "Naive Bayes - train")
    print(mat, acc)

    mat, acc = utils.metrics(classifier, labels, *test_datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, 'Naive Bayes - test')

def w2vector():
    from model import word2vec
    datas, test_datas, vocab, labels = load_data()
    w2v = word2vec("token",
                   datas)
    # w2v.train()
    w2v.train_with_pretrain("./checkpoints/sgns.literature.word")


# def LR_predict():
#     from model import LR
#     datas, test_datas, vocab, labels = load_data()
#     model = LR('token',
#                datas,
#                input_dim=50
#                )
#     print(model)
#     model.train()

#     mat, acc = utils.metrics(model, labels, *datas)
#     print(mat, acc)
#     utils.plot_confusion(mat, labels, "LR - train")

#     mat, acc = utils.metrics(model, labels, *test_datas)
#     print(mat, acc)
#     utils.plot_confusion(mat, labels, "LR - test")


def RF_predict():
    from model import RFClassifier
    datas, test_datas, vocab, labels = load_data()
    model = RFClassifier('token',
                         datas,
                         word_dim=50)
    model.train()
    np.set_printoptions(precision=1)

    mat, acc = utils.metrics(model, labels, *datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, "Random Forests - train")

    mat, acc = utils.metrics(model, labels, *test_datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, "Random Forests - test")


def LR_predict():
    from model import LRClassifier
    datas, test_datas, vocab, labels = load_data()
    model = LRClassifier('token', datas, word_dim=50)
    model.train()
    np.set_printoptions(precision=1)

    mat, acc = utils.metrics(model, labels, *datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, "Random Forests - train")

    mat, acc = utils.metrics(model, labels, *test_datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, "Random Forests - test")


def plot():
    mat = np.eye(4)
    name = ['1', '2', '3', '4']
    utils.plot_confusion(mat, name)


def NB_predict_word():
    from model import NaiveBayes
    datas, test_datas, vocab, labels = load_data_all()

    print(f"VOCAB size: {len(vocab)}")

    classifier = NaiveBayes(vocab=vocab)
    classifier.train(*datas)

    np.set_printoptions(precision=1)

    mat, acc = utils.metrics(classifier, labels, *datas)
    utils.plot_confusion(mat, labels, "Naive Bayes - train")
    print(mat, acc)

    mat, acc = utils.metrics(classifier, labels, *test_datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, 'Naive Bayes - test')


def RF_predict_word():
    from model import RFClassifier
    datas, test_datas, vocab, labels = load_data_all()
    model = RFClassifier('word', datas, word_dim=20)
    model.train()
    np.set_printoptions(precision=1)

    mat, acc = utils.metrics(model, labels, *datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, "Random Forests - train")

    mat, acc = utils.metrics(model, labels, *test_datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, "Random Forests - test")


def LR_predict_word():
    from model import LR
    datas, test_datas, vocab, labels = load_data_all()
    model = LR('word', datas, input_dim=20)
    print(model)
    model.train()

    mat, acc = utils.metrics(model, labels, *datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, "LR - train")

    mat, acc = utils.metrics(model, labels, *test_datas)
    print(mat, acc)
    utils.plot_confusion(mat, labels, "LR - test")


if __name__ == "__main__":
    # NB_predict()
    # w2vector()
    LR_predict()
    # RF_predict()
    # plot()
    # NB_predict_word()
    # RF_predict_word()
    # LR_predict_word()
