import time
import torch
import numpy as np
from model import FastText, TextCNN
import horovod.torch as hvd
from train_eval import train_one_epoch, evaluate
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, FastText')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--batch-size', default=64, type=int, help='Using how many GPU to train')
parser.add_argument('--epochs', default=30, type=int, help='train epochs')
args = parser.parse_args()




if args.model == 'FastText':
    from utils_fasttext import build_dataset, DatasetIterater
else:
    from utils import build_dataset, DatasetIterater


hvd.init()

device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.rank())
    device = 'cuda'


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


vocab, train_dataset, val_dataset = build_dataset(train_paths, val_paths, word_level=args.word, pad_size=100)


model = FastText(args.batch_size, 4, len(vocab), 300, None) if args.model == 'FastText' else TextCNN(args.batch_size, 3, len(vocab), 300, None)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1.2e-6)
optimizer = hvd.DistributedOptimizer(
    optimizer=optimizer, \
    named_parameters=model.named_parameters(),
    backward_passes_per_step=1
)
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

train_iter = DatasetIterater(train_dataset, args.batch_size, device)
val_iter = DatasetIterater(val_dataset, args.batch_size, device)

for i in range(args.epochs):
    train_one_epoch(model, train_iter, optimizer, hvd.rank())
    if hvd.rank() == 0:
        acc, loss, report, confusion = evaluate(model, val_iter, True)
        print(acc, loss)
        print(report)
        print(confusion)




    