import time
import torch
import numpy as np
from model import FastText, TextCNN
from train_eval import train_one_epoch, evaluate, adjust_learning_rate
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, FastText')
parser.add_argument('--word', default=False, type=bool, help='true for word, false for char')
parser.add_argument('--batch-size', default=64, type=int, help='batch size for training')
parser.add_argument('--epochs', default=30, type=int, help='train epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--balance', default=False, type=bool, help='balance train dataset')
args = parser.parse_args()




if args.model == 'FastText':
    from utils_fasttext import build_dataset, DatasetIterater
else:
    from utils import build_dataset, DatasetIterater


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


train_paths = [
    './data/train/ancient.txt', 
    './data/train/ballad.txt',
    './data/train/rap.txt',
    './data/train/rock.txt'
]

val_paths = [
    './data/test/ancient-test.txt',
    './data/test/ballad-test.txt',
    './data/test/rap-test.txt',
    './data/test/rock-test.txt'
]

if args.word:
    pad_size = 110
else:
    pad_size = 220
vocab, train_dataset, val_dataset = build_dataset(train_paths, val_paths, word_level=args.word, pad_size=pad_size, balance=args.balance)


model = FastText(args.batch_size, 4, len(vocab), 300, None) if args.model == 'FastText' else TextCNN(args.batch_size, 4, len(vocab), 300, None)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_iter = DatasetIterater(train_dataset, args.batch_size, device)
val_iter = DatasetIterater(val_dataset, args.batch_size, device)

for c_epoch in range(args.epochs):
    print("Epoch {}/{}".format(c_epoch, args.epochs))
    train_one_epoch(model, train_iter, optimizer)
    acc, loss, report, confusion = evaluate(model, val_iter, True)
    print(acc, loss)
    print(report)
    print(confusion)




    