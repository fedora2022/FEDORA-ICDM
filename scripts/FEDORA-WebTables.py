import os
MAX_LEN = 512
SEP_TOKEN_ID = 102
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tqdm
import time
import json
import numpy as np
import random
import torch
import functools
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import jsonlines
from transformers import BertModel, BertForSequenceClassification, BertConfig
from tqdm import tqdm
from math import sqrt

label_dict = {'age': 0, 'weight': 1, 'city': 2, 'state': 3, 'name': 4, 'type': 5, 'location': 6, 'club': 7, 'code': 8, 'description': 9, 'album': 10, 'company': 11, 'symbol': 12, 'elevation': 13, 'address': 14, 'status': 15, 'gender': 16, 'artist': 17, 'year': 18, 'rank': 19, 'team': 20, 'country': 21, 'isbn': 22, 'notes': 23, 'format': 24, 'position': 25, 'result': 26, 'class': 27, 'language': 28, 'origin': 29, 'county': 30, 'order': 31, 'owner': 32, 'genre': 33, 'category': 34, 'continent': 35, 'credit': 36, 'collection': 37, 'filesize': 38, 'day': 39, 'plays': 40, 'species': 41, 'duration': 42, 'affiliation': 43, 'teamname': 44, 'area': 45, 'jockey': 46, 'manufacturer': 47, 'component': 48, 'region': 49, 'creator': 50, 'depth': 51, 'grades': 52, 'industry': 53, 'birthdate': 54, 'birthplace': 55, 'sex': 56, 'sales': 57, 'affiliate': 58, 'publisher': 59, 'family': 60, 'product': 61, 'nationality': 62, 'service': 63, 'brand': 64, 'command': 65, 'ranking': 66, 'requirement': 67, 'range': 68, 'operator': 69, 'capacity': 70, 'classification': 71, 'director': 72, 'person': 73, 'education': 74, 'currency': 75, 'religion': 76}

def load_jsonl(jsonl_path, label_dict):
    target_cols = []
    labels = []
    fd_cols = []
    headers_alias = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    mapping = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25}
    with open(jsonl_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            try:
                target_cols.append(np.array(item['content'])[:,int(item['target'])])
                labels.append(int(label_dict[item['label']]))
            except IndexError:
                continue
            target_alias = headers_alias[int(item['target'])]
            fds = item['FD']
            fd_col_dict = {}
            fd_col = []
            max_size = int(sqrt(len(item['headers'])))
            for fd_pair in fds:
                if fd_pair[1] == target_alias:
                    for char in fd_pair[0]:
                        if not mapping[char] in fd_col_dict.keys():
                            fd_col_dict[mapping[char]] = 1
                        else:
                            fd_col_dict[mapping[char]] += 1
                if fd_pair[0] == target_alias:
                    for char in fd_pair[1]:
                        if not mapping[char] in fd_col_dict.keys():
                            fd_col_dict[mapping[char]] = 1
                        else:
                            fd_col_dict[mapping[char]] += 1
            fd_sorted_dict = dict(sorted(fd_col_dict.items(), key=lambda x:x[1], reverse=True))
            for i in range(min(max_size,len(fd_sorted_dict))):
                fd_col.append(np.array(item['content'])[:,int(list(fd_sorted_dict.keys())[i])])
            fd_cols.append(fd_col)
    return target_cols, fd_cols, labels

class TableDataset(Dataset):
    def __init__(self, target_cols, tokenizer, fd_cols, labels):
        self.labels = labels
        self.target_cols = target_cols
        self.tokenizer = tokenizer
        self.fd_cols = fd_cols

    def tokenize(self, col):
        text = ''
        for cell in col:
            text+=cell
            text+=' '
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __getitem__(self, idx):
        target_token_ids = self.tokenize(self.target_cols[idx])
        fd_token_ids = []
        if len(self.fd_cols[idx]) == 0:
            fd_token_ids.append(target_token_ids)
        else:
            for col in self.fd_cols[idx]:
                fd_token_id= self.tokenize(col)
                fd_token_ids.append(fd_token_id)
        fd_token_ids = torch.stack(fd_token_ids)
        return target_token_ids, fd_token_ids, torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        fd_ids = [x[1] for x in batch]
        labels = torch.stack([x[2] for x in batch])
        return token_ids, fd_ids, labels

def get_loader(target_cols, fds, labels,batch_size=8,is_train=True):
    ds_df = TableDataset(target_cols, Tokenizer, fds, labels)
    loader = torch.utils.data.DataLoader(ds_df, batch_size=batch_size, shuffle=is_train, num_workers=0, collate_fn=ds_df.collate_fn)
    loader.num = len(ds_df)
    return loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class FEDORA(torch.nn.Module):
    def __init__(self, n_classes=77, dim_k=256, dim_v=256, num_heads=4):
        super(FEDORA, self).__init__()
        self.model_name = 'FEDORA'
        self.num_heads = num_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1/sqrt(dim_k//num_heads)
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.fd_bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.fcc= torch.nn.Linear(256, n_classes)
        self.linear_q = torch.nn.Linear(768, dim_k, bias = False)
        self.linear_k = torch.nn.Linear(768, dim_k, bias = False)
        self.linear_v = torch.nn.Linear(768, dim_v, bias = False)

    def attention(self, target_col, fds):
        cols = []
        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v //nh
        for ids in fds:
            attention_mask = (ids > 0)
            _, cur_pooled = self.fd_bert_model(input_ids=ids, attention_mask=attention_mask, return_dict=False)
            cols.append(cur_pooled)
        target_col = target_col.reshape(1,-1)
        fd_cols = torch.stack(cols).view(-1,768)
        concated = torch.cat((target_col, fd_cols)).reshape(1,-1,768)
        q = self.linear_q(concated).reshape(1,-1,nh,dk).transpose(1,2)
        k = self.linear_k(concated).reshape(1,-1,nh,dk).transpose(1,2)
        v = self.linear_v(concated).reshape(1,-1,nh,dv).transpose(1,2)
        dist = torch.matmul(q,k.transpose(2,3))*self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        attention_out = torch.matmul(dist,v)
        attention_out = attention_out.transpose(1,2).reshape(1,-1,self.dim_v)
        attention_out = attention_out[:,0,:]
        return attention_out

    def forward(self,ids,fds):
        attention_mask = (ids > 0)
        _, pooled =self.bert_model(input_ids=ids, attention_mask=attention_mask, return_dict=False)
        attention_out = self.attention(pooled, fds)
        out=self.dropout(attention_out)
        out = self.fcc(out)
        out = out.view(-1, 77)
        
        return out

def metric_fn(preds, labels):
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }


def train_model(model,train_loader,val_loader,lr,model_save_path='pytorch_FEDORA_model.pkl',early_stop_epochs=5,epochs=15):  
    weight_decay = 1e-2
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    cur_best_v_loss =10.0
    for epoch in range(1,epochs+1):
        model.train()
        bar1 = tqdm(train_loader)
        epoch_loss = 0
        epoch_acc = 0
        v_epoch_loss = 0
        v_epoch_acc = 0
        for i,(ids, fds, labels) in enumerate(bar1):
            labels = labels.cuda()
            fds = [fd.cuda() for fd in fds]
            output = model(ids.cuda(), fds)
            y_pred_prob = output
            y_pred_label = y_pred_prob.argmax(dim=1)
            loss = loss_fn(y_pred_prob.view(-1, 77), labels.view(-1))
            acc = ((y_pred_label == labels.view(-1)).sum()).item()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_acc += acc
            length_label = len(labels)
            del ids, fds, labels
            torch.cuda.empty_cache()
        train_length = len(bar1)+1
        print("Epoch:", epoch, "training_loss:", epoch_loss / (train_length), "\t", "current acc:", epoch_acc / ((train_length)-1+length_label))
        model.eval()
        bar2 = tqdm(val_loader)
        pred_labels = []
        true_labels = []
        for j,(ids, fds, labels) in enumerate(bar2):
            labels = labels.cuda()
            fds = [fd.cuda() for fd in fds]
            output = model(ids.cuda(), fds)
            y_pred_prob = output
            y_pred_label = y_pred_prob.argmax(dim=1)
            vloss = loss_fn(y_pred_prob.view(-1, 77), labels.view(-1))
            acc = ((y_pred_label == labels.view(-1)).sum()).item()
            pred_labels.append(y_pred_label.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())
            v_epoch_loss += vloss.item()
            v_epoch_acc += acc
            v_length_label = len(labels)
            del ids, fds
            torch.cuda.empty_cache()
        pred_labels = np.concatenate(pred_labels)
        true_labels = np.concatenate(true_labels)
        val_length = len(bar2)+1
        print("validation_loss:", v_epoch_loss / (val_length), "\t", "current acc:", v_epoch_acc / (val_length-1+v_length_label))
        f1_scores = metric_fn(pred_labels, true_labels)
        print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
        torch.save(model.state_dict(),model_save_path+'_'+str(epoch)+'.pkl')

if __name__ == '__main__':
    setup_seed(20)
    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    jsonl_path_0 = "./data/sato/K0.jsonl"
    jsonl_path_1 = "./data/sato/K1.jsonl"
    jsonl_path_2 = "./data/sato/K2.jsonl"
    jsonl_path_3 = "./data/sato/K3.jsonl"
    jsonl_path_4 = "./data/sato/K4.jsonl"
    jsonls = [jsonl_path_0, jsonl_path_1, jsonl_path_2, jsonl_path_3, jsonl_path_4]
    target_colss = []
    fdss = []
    labelss = []
    for i in range(5):
        target_cols, fds, labels = load_jsonl(jsonls[i], label_dict)
        target_colss.append(target_cols)
        fdss.append(fds)
        labelss.append(labels)

    train_cols_0 = target_colss[0]+target_colss[1]+target_colss[2]+target_colss[3]
    val_cols_0 = target_colss[4]
    train_fds_0 = fdss[0]+fdss[1]+fdss[2]+fdss[3]
    val_fds_0 = fdss[4]
    train_labels_0 = labelss[0]+labelss[1]+labelss[2]+labelss[3]
    val_labels_0 = labelss[4]
    train_cols_1 = target_colss[0]+target_colss[1]+target_colss[2]+target_colss[4]
    val_cols_1 = target_colss[3]
    train_fds_1 = fdss[0]+fdss[1]+fdss[2]+fdss[4]
    val_fds_1 = fdss[3]
    train_labels_1 = labelss[0]+labelss[1]+labelss[2]+labelss[4]
    val_labels_1 = labelss[3]
    train_cols_2 = target_colss[0]+target_colss[1]+target_colss[4]+target_colss[3]
    val_cols_2 = target_colss[2]
    train_fds_2 = fdss[0]+fdss[1]+fdss[4]+fdss[3]
    val_fds_2 = fdss[2]
    train_labels_2 = labelss[0]+labelss[1]+labelss[4]+labelss[3]
    val_labels_2 = labelss[2]
    train_cols_3 = target_colss[0]+target_colss[4]+target_colss[2]+target_colss[3]
    val_cols_3 = target_colss[1]
    train_fds_3 = fdss[0]+fdss[4]+fdss[2]+fdss[3]
    val_fds_3 = fdss[1]
    train_labels_3 = labelss[0]+labelss[4]+labelss[2]+labelss[3]
    val_labels_3 = labelss[1]
    train_cols_4 = target_colss[4]+target_colss[1]+target_colss[2]+target_colss[3]
    val_cols_4 = target_colss[0]
    train_fds_4 = fdss[4]+fdss[1]+fdss[2]+fdss[3]
    val_fds_4 = fdss[0]
    train_labels_4 = labelss[4]+labelss[1]+labelss[2]+labelss[3]
    val_labels_4 = labelss[0]

    train_cols = [train_cols_0, train_cols_1, train_cols_2, train_cols_3, train_cols_4]
    val_cols = [val_cols_0, val_cols_1, val_cols_2, val_cols_3, val_cols_4]
    train_fds = [train_fds_0,train_fds_1,train_fds_2,train_fds_3,train_fds_4]
    val_fds = [val_fds_0,val_fds_1,val_fds_2,val_fds_3,val_fds_4]
    train_labels = [train_labels_0,train_labels_1,train_labels_2,train_labels_3,train_labels_4]
    val_labels = [val_labels_0,val_labels_1,val_labels_2,val_labels_3,val_labels_4]

    BS = 1
    lrs = [1e-5]
    for lr in lrs:
        print("start for learning rate:", lr)
        for cur_fold in range(5):
            ## if cur_fold == 0: #(this line is for parallel training the model on five folds, uncomment this line and adjust the indentation properly, you can then select the fold to be trained by changing 0 to 1,2,3,4)
            cur_train_cols = train_cols[cur_fold]
            cur_train_labels = train_labels[cur_fold]
            cur_train_fds = train_fds[cur_fold]
            cur_val_cols = val_cols[cur_fold]
            cur_val_labels = val_labels[cur_fold]
            cur_val_fds = val_fds[cur_fold]
            train_loader = get_loader(cur_train_cols, cur_train_fds, cur_train_labels, BS, True)
            val_loader = get_loader(cur_val_cols, cur_val_fds, cur_val_labels, 1, False)
            model = FEDORA().cuda()
            model_save_path = './checkpoints/FEDORA_FD_webtables'+"_lr="+str(lr)+'_{}'.format(cur_fold+1)
            print("Starting fold", cur_fold+1)
            train_model(model, train_loader, val_loader,lr, model_save_path=model_save_path)
