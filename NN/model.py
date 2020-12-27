import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



      
class TextCNN(nn.Module):
    def __init__(self, batch_size, num_classes, embedding_size, embedding_dim, embedding_pretrained):
        super(TextCNN, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_pretrained,
                freeze=True
            )
        else:
            self.embedding = nn.Embedding(embedding_size, embedding_dim)

        self.filter_sizes = (2, 3 ,4)
        self.num_filters = 256
        self.embedding_dim = embedding_dim
        self.dropout_rate = 0.5
        self.num_classes = num_classes

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, embedding_dim)) for k in self.filter_sizes]
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(
            self.num_filters * len(self.filter_sizes),
            self.num_classes    
        )
    
    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        #encode
        x = self.embedding(x[0])
        x_encode = x.unsqueeze(1)
        #run
        x = torch.cat([self.conv_and_pool(x_encode, conv) for conv in self.convs], 1)
        x = self.dropout(x)
        #decode
        out = self.fc(x)
        return out

class FastText(nn.Module):
    def __init__(self, batch_size, num_classes, embedding_size, embedding_dim, embedding_pretrained):
        super(FastText, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_pretrained,
                freeze=True
            )
        else:
            self.embedding = nn.Embedding(embedding_size, embedding_dim)

        self.filter_sizes = (2, 3 ,4)
        self.num_filters = 256
        self.embedding_dim = embedding_dim
        self.dropout_rate = 0.5
        self.num_classes = num_classes
        self.hidden_size = 256

        # problem?
        self.n_gram_vocab = 250499
        
        self.embedding_ngram2 = nn.Embedding(self.n_gram_vocab, self.embedding_dim)
        self.embedding_ngram3 = nn.Embedding(self.n_gram_vocab, self.embedding_dim)
        self.embedding_ngram4 = nn.Embedding(self.n_gram_vocab, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.embedding_dim * 4, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        #encode
        x_word = self.embedding(x[0])
        x_bigram = self.embedding_ngram2(x[2])
        x_trigram = self.embedding_ngram3(x[3])
        x_tetragram = self.embedding_ngram4(x[4])
        x = torch.cat((x_word, x_bigram, x_trigram, x_tetragram), -1)
        #run
        x = x.mean(dim=1)
        x = self.dropout(x)
        #decode
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


if __name__ == "__main__":
    batch_size = 32
    num_classes = 4
    embedding_size = 300
    embedding_pretrained = torch.tensor(np.random.randn(20000, embedding_size).astype('float32'))
    
    #textCNN
    model1 = TextCNN(batch_size, num_classes, embedding_size, embedding_pretrained)
    input_for_model1 = torch.tensor(np.random.randint(0, 20000, size=(batch_size, 32, 32)))
    print(model1(input_for_model1))
    #FastText
    model2 = FastText(batch_size, num_classes, embedding_size, embedding_pretrained)
    input_for_model2 = torch.tensor(np.random.randint(0, 20000, size=(batch_size, 32, 32)))