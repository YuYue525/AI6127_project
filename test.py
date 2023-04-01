import os
import numpy
import torch
import argparse

from torchtext import data
from torchtext import datasets

from rnn import RNN
from gru import GRU
from lstm import LSTM
from birnn import BiRNN
from bigru import BiGRU
from bilstm import BiLSTM

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['rnn', 'lstm', 'gru', 'birnn', 'bilstm', 'bigru'], default='rnn',\
                        help="type of model select from ['rnn', 'lstm', 'gru', 'birnn', 'bilstm', 'bigru']")
    parser.add_argument('--model_path', default="./check_points/rnn.pt", help="model path")
    args = parser.parse_args()
    return args

class SNLI():
    def __init__(self, batch_size, device):
        self.inputs = data.Field(lower=True, tokenize = None, batch_first = True)
        self.labels = data.Field(sequential = False, unk_token = None, is_target = True)
        self.train, self.val, self.test = datasets.SNLI.splits(self.inputs, self.labels)
        self.inputs.build_vocab(self.train, self.val)
        self.labels.build_vocab(self.train)
        self.train_iter, self.val_iter, self.test_iter = data.Iterator.splits((self.train, self.val, self.test), batch_size = batch_size, device=device)
    
    def vocabulary_size(self):
        return len(self.inputs.vocab)
    
    def out_dim(self):
        return len(self.labels.vocab)
    
    def labels(self):
        return self.labels.vocab.stoi

def test(model, dataset):
    model.eval()
    dataset.test_iter.init_epoch()

    test_f1_score = 0
    test_precision = 0
    test_recall = 0
    test_acc = 0

    y_true = numpy.array([])
    y_pred = numpy.array([])

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset.test_iter):

            prediction = model(batch)
            
            y_true = numpy.append(y_true,  batch.label.cpu().detach().numpy())
            y_pred = numpy.append(y_pred, torch.argmax(prediction, dim=1).cpu().detach().numpy())

    test_f1_score = f1_score(y_true, y_pred, average='macro')
    test_precision = precision_score(y_true, y_pred, average='macro')
    test_recall = recall_score(y_true, y_pred, average='macro')
    test_acc = accuracy_score(y_true, y_pred)

    return test_f1_score, test_precision, test_recall, test_acc

if __name__ == "__main__":

    args = parse_args()

    batch_size = 128
    embedding_dim = 150
    dropout_ratio = 0.2
    hidden_dim = 256

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda:{}'.format(0))
    else:
        device = torch.device('cpu')

    dataset = SNLI(batch_size, device)
    out_dim = dataset.out_dim()
    vocab_size = dataset.vocabulary_size()

    if args.model_type == "rnn":
        model = RNN(vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim)
    elif args.model_type == "lstm":
        model = LSTM(vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim)
    elif args.model_type == "gru":
        model = GRU(vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim)
    elif args.model_type == "birnn":
        model = BiRNN(vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim)
    elif args.model_type == "bilstm":
        model = BiLSTM(vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim)
    elif args.model_type == "bigru":
        model = BiGRU(vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim)
    else:
        print("Wrong model type!")
        exit(0)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path))

    f1, precision, recall, acc = test(model, dataset)

    print("Test f1: {}; precision: {}; recall: {}; accuracy: {}".format(f1, precision, recall, acc))



