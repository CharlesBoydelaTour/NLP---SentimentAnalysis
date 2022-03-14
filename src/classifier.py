###Aspect-Based Sentiment Analysis

# coding=utf-8
# Reference: https://github.com/huggingface/pytorch-pretrained-BERT

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from transformers import BertTokenizer,BertModel, BertConfig
import transformers


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear1(dropout_output)
        act_output = self.relu(linear_output)
        final_layer = self.linear2(act_output)

        return final_layer
    
class Classifier:
    def __init__(self, batch_size=32, max_seq_length=256, lr=1e-5, eps=1e-8, epochs=5, warmup_steps=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.lr = lr
        self.eps = eps
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.criterion = CrossEntropyLoss()
        self.train_losses = []
        self.model = BertClassifier().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr, eps = self.eps)
        
    def data_loading_fn(self, datafile):
            data = pd.read_csv(datafile, sep='\t', names = ["polarity_label","aspect_category", "target_term", "character_offsets", "sentence"])
            data["aspect_category"] = data["aspect_category"].str.replace("#", " ") 
            data["concat"] = data["aspect_category"] + "[SEP] " + data["target_term"] + "[SEP] " + data["sentence"]
            label, corpus = data["polarity_label"].to_list(), data["concat"].to_list()
            classes = {'negative':0,'positive':1,'neutral':2}
            label = [classes[y] for y in label]
            encoded_data = self.tokenizer.batch_encode_plus(corpus,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='longest',
            return_tensors='pt')
            input_ids_train = encoded_data['input_ids']
            attention_masks_train = encoded_data['attention_mask']
            labels_train = torch.tensor(label)
            dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
            return dataset_train
        
    def train(self,trainfile, devfile):
        train_dataset = self.data_loading_fn(trainfile)
        dev_dataset = self.data_loading_fn(devfile)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch_size)
        dev_dataloader = DataLoader(dev_dataset, sampler=RandomSampler(dev_dataset), batch_size=self.batch_size)
        total_steps = len(train_dataloader) * self.epochs
        scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps)
        for epoch in range(self.epochs):
            self.train_losses = []
            self.model.train()
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                input_ids = batch[0].to(self.device)
                mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                outputs = self.model(input_ids, mask)
                loss = self.criterion(outputs, labels)
                self.train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                scheduler.step()
            self.model.eval()
            eval_losses = []
            eval_accs = []
            for batch in dev_dataloader:
                input_ids = batch[0].to(self.device)
                mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, mask)
                    loss = self.criterion(outputs, labels)
                    eval_losses.append(loss.item())
                    preds = torch.argmax(outputs, dim=1).flatten()
                    labels = labels.flatten()
                    accuracy = (preds == labels).cpu().numpy().mean() * 100
                    eval_accs.append(accuracy)
            #print("Epoch {} | Train Loss {:.5f} - - Eval Loss {:.5f} - - Eval Acc {:.2f}".format(epoch, np.mean(self.train_losses), np.mean(eval_losses), np.mean(eval_accs)))
            
    def predict(self, datafile):
        test_dataset = self.data_loading_fn(datafile)
        #No shuffling for test dataset
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=1)
        self.model.eval()
        preds = []
        for batch in test_dataloader:
            input_ids = batch[0].to(self.device)
            mask = batch[1].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, mask)
                preds.append(torch.argmax(outputs, dim=1).flatten().cpu().numpy()[0])
        classes = {0:'negative',1:'positive',2:'neutral'}
        preds = [classes[y] for y in preds]
        return preds