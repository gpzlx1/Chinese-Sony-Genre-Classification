import torch
from sklearn import metrics
import torch.nn.functional as F
import numpy as np

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def train_one_epoch(model, train_iter, optimizer):
    model.train()
    total_batch = 0

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