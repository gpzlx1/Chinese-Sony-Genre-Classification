# A simple implementation for predicting music genre by lyrics using neural network

Support Network:

* TextCNN

* FastText

Usage:

```shell
usage: run.py [-h] --model MODEL [--word WORD] [--batch-size BATCH_SIZE]
              [--epochs EPOCHS] [--lr LR] [--balance BALANCE]

Chinese Text Classification

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         choose a model: TextCNN, FastText
  --word WORD           True for word, False for char
  --batch-size BATCH_SIZE
                        Using how many GPU to train
  --epochs EPOCHS       train epochs
  --lr LR               learning rate
  --balance BALANCE     balance train dataset
```

Train and evaluate:

```shell
cd $PROJECT_ROOT

python3 src/NN/run.py
```

