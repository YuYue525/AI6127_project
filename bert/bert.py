import os
import numpy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from transformers import BertTokenizer, BertModel

import logging
logging.disable(logging.WARNING)

BATCH_SIZE = 64
LEARNING_RATE = 2e-5
EPOCHS = 5
MAX_LENGTH = 128

train_file = './SNLI/train.txt'
val_file = './SNLI/valid.txt'
test_file = './SNLI/test.txt'
log_file = "./log_bert.txt"
train_loss_history = []
train_f1_history = []

def preprocessing(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()
        data = [d.split('_!_') for d in data]
    for each in data:
        each[2] = int(each[2])
    return data

class SentencePairDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence_pair = self.sentences[index]
        sent1, sent2, label = sentence_pair[0], sentence_pair[1], sentence_pair[2]
        encoding = self.tokenizer(sent1, sent2, add_special_tokens=True, truncation=True, max_length=self.max_length, padding='max_length')
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        token_type_ids = torch.tensor(encoding['token_type_ids'])
        label = torch.tensor(label)
        return input_ids, attention_mask, token_type_ids, label

class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return self.softmax(logits)

def train(model, data_loader, criterion, optimizer, loss_history, f1_history, device):
    model.train()
    train_loss = 0
    train_f1_score = 0
    
    print_loss = 0
    plot_loss = 0
    print_f1 = 0
    plot_f1 = 0
    print_iter = 1000
    plot_iter = 1000
    
    for idx,batch in enumerate(data_loader):
        
        input_ids, attention_mask, token_type_ids, target = [x.to(device) for x in batch]
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        print_loss += loss.item()
        plot_loss += loss.item()
        
        y_true = target.cpu().detach().numpy()
        y_pred = torch.argmax(output, dim=1).cpu().detach().numpy()
        
        train_f1_score += f1_score(y_true, y_pred, average='weighted')
        print_f1 += f1_score(y_true, y_pred, average='weighted')
        plot_f1 += f1_score(y_true, y_pred, average='weighted')
        
        if (idx + 1) % print_iter == 0:
            f = open(log_file, "a+")
            f.write("Iteration {}: train loss {}, train f1 score {}\n".format(idx + 1, print_loss / print_iter, print_f1 / print_iter))
            print_loss = 0
            print_f1 = 0
            f.close()
            
        if (idx + 1) % plot_iter == 0:
            loss_history.append(plot_loss / plot_iter)
            f1_history.append(plot_f1 / plot_iter)
            plot_loss = 0
            plot_f1 = 0

    return train_loss/len(data_loader), train_f1_score/len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()

    eval_loss = 0

    y_true = numpy.array([])
    y_pred = numpy.array([])

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, target = [x.to(device) for x in batch]

            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            y_true = numpy.append(y_true,  target.cpu().detach().numpy())
            y_pred = numpy.append(y_pred, torch.argmax(output, dim=1).cpu().detach().numpy())

            loss = criterion(output, target)
            eval_loss += loss.item()

    eval_f1_score = f1_score(y_true, y_pred, average='macro')
    eval_precision = precision_score(y_true, y_pred, average='macro')
    eval_recall = recall_score(y_true, y_pred, average='macro')
    eval_acc = accuracy_score(y_true, y_pred)
            
    return eval_loss/len(data_loader), eval_f1_score, eval_precision, eval_recall, eval_acc

if __name__ == '__main__':
    
    if (os.path.exists(log_file)):
        os.remove(log_file)

    train_data = preprocessing(train_file)
    val_data = preprocessing(val_file)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = SentencePairDataset(train_data, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = SentencePairDataset(val_data, tokenizer, MAX_LENGTH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BertClassifier(num_labels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        f = open(log_file, "a+")
        f.write(" === start epoch {} ===\n".format(epoch))
        f.close()
        train_loss, train_f1_score = train(model, train_loader, criterion, optimizer, train_loss_history, train_f1_history, device)
        f = open(log_file, "a+")
        f.write(f'\nEpoch {epoch+1}: Train loss: {train_loss:.4f}, Train F1 score: {train_f1_score:.4f}\n\n')
        f.close()
        
        val_loss, val_f1_score, val_precision, val_recall, val_acc = evaluate(model, val_loader, criterion, device)
    
        f = open(log_file, "a+")
        f.write(f'Valid loss: {val_loss:.4f}, Valid F1 score: {val_f1_score:.4f}, Valid acc: {val_acc:.4f}, Valid precision: {val_precision:.4f}, Valid recall: {val_recall:.4f}\n\n')
        f.close()
    
    plt.figure()
    plt.title("loss curves for BERT on SNLI")
    plt.plot([x / len(train_loss_history) * EPOCHS for x in range(1, len(train_loss_history) + 1)], train_loss_history, label="train loss")
    plt.plot([x / len(train_f1_history) * EPOCHS for x in range(1, len(train_f1_history) + 1)], train_f1_history, label="train f1")
    plt.xlabel("epochs")
    plt.ylabel("values")
    plt.legend()
    plt.savefig("./bert.png")
    
    test_data = preprocessing(test_file)

    test_dataset = SentencePairDataset(test_data, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    test_loss, test_f1_score, test_precision, test_recall, test_acc = evaluate(model, test_loader, criterion, device)
    
    f = open(log_file, "a+")
    f.write(f'Test loss: {test_loss:.4f}, Test F1 score: {test_f1_score:.4f}, Test acc: {test_acc:.4f}, Test precision: {test_precision:.4f}, Test recall: {test_recall:.4f}\n\n')
    f.close()

    torch.save(model.state_dict(), './bert.pt')
