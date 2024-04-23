from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertModel, BertTokenizer

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, activation=nn.Tanh()):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation = activation


        self.stem = nn.Sequential(nn.Linear(input_size, hidden_size), activation)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation)
        
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.output(x)
        return x



class TextDataSet(Dataset):
    def __init__(self, text, labels, tokenizer, max_length):
        self.text = text
        self.labels = labels
        self.tokenizer: BertTokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt', )

        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label)}


class BERTFinetune(nn.Module):
    def __init__(self, bert_name, num_classes, dropout=0.1):
        super(BERTFinetune, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.classify = True
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)

        if self.classify:
            out = self.classifier(pooled_output)
        else:
            out = pooled_output

        return out

    def finetune(self):
        self.classify = True
    
    def feature_extractor(self):
        self.classify = False