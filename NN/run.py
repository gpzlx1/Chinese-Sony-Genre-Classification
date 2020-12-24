import time
import torch
import torch.nn.functional as F
import numpy as np
#from utils import build_dataset, DatasetIterater
from utils_fasttext import build_dataset, DatasetIterater
from model import FastText, TextCNN
from sklearn import metrics
import argparse

#parser = argparse.ArgumentParser(description='Chinese Text Classification')
#parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
#parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
#parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
#args = parser.parse_args()


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def train(model, train_iter, num_epochs=20):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    total_batch = 0

    for epoch in range(num_epochs):
        
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                predic = torch.max(outputs.data, 1)[1].cpu()
                labels = labels.data.cpu().numpy()
                train_acc = metrics.accuracy_score(labels, predic)
                
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}'
                print(msg.format(total_batch, loss.item(), train_acc))

                model.train()
            total_batch += 1

def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=['ancient', 'ballad', 'rap', 'rock'], digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

if __name__ == '__main__':

    train_paths = [
        './split-data/data/SPLIT/train/ancient.txt', 
        './split-data/data/SPLIT/train/ballad.txt',
        './split-data/data/SPLIT/train/rap.txt',
        './split-data/data/SPLIT/train/rock.txt'
    ]
    val_paths = [
        './split-data/data/SPLIT/test/ancient-test.txt',
        './split-data/data/SPLIT/test/ballad-test.txt',
        './split-data/data/SPLIT/test/rap-test.txt',
        './split-data/data/SPLIT/test/rock-test.txt'
    ]

    batch_size = 64
    vocab, train_dataset, val_dataset = build_dataset(train_paths, val_paths, word_level=True, pad_size=100)
    train_iter = DatasetIterater(train_dataset, batch_size, device)
    val_iter = DatasetIterater(val_dataset, batch_size, device)
    #model = TextCNN(batch_size, 4, len(vocab), 300, None).to(device)
    model = FastText(batch_size, 4, len(vocab), 300, None).to(device)
    #_, _, report, confusion =  evaluate(model, val_iter, test=True)
    #print(report)
    #print(confusion)
    num_epochs = 30
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        train(model, train_iter, num_epochs=1)
        _, _, report, confusion =  evaluate(model, val_iter, test=True)
        print(report)
        print(confusion)