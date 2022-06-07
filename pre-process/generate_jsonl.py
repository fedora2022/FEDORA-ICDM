import csv
import torch
import os
import torch.nn as nn
import numpy as np
import random
import json
import jsonlines
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

def read_csv(csv_dir):
    csv_lines = csv.reader(open(csv_dir))
    content = []
    num_row = 0
    for i, line in enumerate(csv_lines):
        if i == 0:
            header = line
            num_col = len(header)   
        else:
            content.append(line)
            num_row += 1
    num_cell = num_col*num_row
    return header, content, num_col, num_row, num_cell

json_files = []
labels = []

rounds = [1,3,4]

for round in rounds:
    csv_reader = csv.reader(open("FEDORA-ICDM/raw_data/Semtab2019/data/Round "+str(round)+'/gt/CTA_Round'+str(round)+"_gt.csv"))
    csv_dir = "FEDORA-ICDM/data/no-headers/Semtab2019/Round"+str(round)+"/"
    FD_dir = "FEDORA-ICDM/data/fd/Semtab2019/Round"+str(round)+"/"
    train_jsonl = "FEDORA-ICDM/data/train_val.jsonl"
    test_jsonl = "FEDORA-ICDM/data/test.jsonl"
    for line in csv_reader:
        file_name = line[0]+".csv"
        target_col = line[1]
        label = line[2].split('/')[-1]
        file_dir = csv_dir+file_name
        FD_name = file_name+".json"
        FD_path = FD_dir+FD_name
        if not os.path.exists(file_dir):
            continue
        if not os.path.exists(FD_path):
            continue
        with open(FD_path,'r') as load_f:
            load_dict = json.load(load_f)
        header, content, num_col, num_row, num_cell = read_csv(file_dir)
        dict = {}
        dict['filename'] = file_name
        dict['header'] = header
        dict['content'] = content
        dict['target'] = target_col
        dict['label'] = label
        dict['FD'] = load_dict['FD']
        json_files.append(dict)
        labels.append(label)
        
sfolder_test = StratifiedKFold(n_splits=10, random_state = 42, shuffle=True)
train_valid_set = []
test_set = []
for train_valid, test in sfolder_test.split(json_files, labels):
    train_valid_index = train_valid
    test_index = test
    break
for index in train_valid_index:
    with jsonlines.open(train_jsonl, "a") as writer:
        writer.write(json_files[index])
for index in test_index:
    with jsonlines.open(test_jsonl, "a") as writer:
        writer.write(json_files[index])
