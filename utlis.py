import re
import os
import random
import numpy as np
import scipy.sparse as sp
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()


def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

parser.add_argument('--dataset', type=str, default='test_short' )
parser.add_argument('--model', type=str, default='ChatGPT')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--hidden_channels', type=int, default=300)
parser.add_argument('--emb_size', type=int, default=768)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=30000)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--shots', type=int, default=50)
parser.add_argument('-usewhole', action='store_true')
parser.add_argument('--remove_high_freq', type=int, default=0)
parser.add_argument('--dropout_rate', type=float, default=0.0)
parser.add_argument('--dropout_rate_feat', type=float, default=0.0)
parser.add_argument('--dp1', type=float, default=0.4)
parser.add_argument('--dp2', type=float, default=0.6)
parser.add_argument('--dp3', type=float, default=0.1)
parser.add_argument('--low_fre_word', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)


args = parser.parse_args()
print(f"Using the {args.dataset}")

def delete_quotation(text):
    
    pattern = r"(?<=\()'|'(?=\))|(?<=,\s)'|'(?=,)|[a-zA-Z \(\)\"\,]+"
    matches = re.findall(pattern, text)
    result = "".join(matches)
    
    return result
    

def read_KG(dataset):
    
    content_list =[]
    with open(f'./data/corpus/{dataset}_abstract.txt','r') as f:
        for line in f.readlines():
            content_list.append(line.strip())

    KG_list=[]
    KG_doc_list = []
    i=0
    with open(f'./data/corpus/{dataset}_KG.txt', 'r') as f:
        for line in tqdm(f.readlines()):
            #print("line is here:", line)
            if line.strip() == 'None.' or line.strip() == 'None': # if the KG extraction returns none, we use the original file to replace it.
                
                line = content_list[i]
                line = re.sub(r"[^a-zA-Z ]+", '', line) # only keep letter
                line = str(tuple(line.strip().split())) # now the original content is in string(tuple) format, so it can be transfer to tuple later. 
                
            KG_doc_list.append(line) # append each line of the KG to a list, the len(KG_doc_list) == len(doc_list)
            
            knowledgeGraph_doc = line.strip().split(';')
            for knowledgeGraph in knowledgeGraph_doc:
                knowledgeGraph = delete_quotation(knowledgeGraph)
                try:
                    knowledgeGraph_tuple = eval(knowledgeGraph)
                except:
                    pass
                KG_list.append(knowledgeGraph_tuple)
                
            i=i+1
            
        return content_list, KG_doc_list, KG_list

def split_shuffle(dataset):
    
    _, KG_doc_list, _ = read_KG(args.dataset)
    num_doc = len(KG_doc_list)

    train_mask = torch.zeros(num_doc)
    val_mask = torch.zeros(num_doc)
    test_mask =  torch.zeros(num_doc)
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    f = open('data/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)
    real_train_ids = train_ids[0:int(len(train_ids)*0.9)]
    val_ids = train_ids[int(len(train_ids)*0.9):]
    
    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    
    train_mask[real_train_ids] = 1
    val_mask[val_ids] = 1
    test_mask[test_ids] = 1
    
    return train_mask, val_mask, test_mask

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[drop_mask,:] = 0

    return x