import re
import numpy as np
from collections import defaultdict
from scipy import sparse
from utlis import *
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from nltk.stem.porter import *
from nltk.stem.porter import *
import math
import pickle as pkl
import nltk
from nltk.corpus import stopwords
import heapq
from operator import itemgetter
from utlis import *
import pickle

with open(f"{args.dataset}_TF_IDF.pkl", "rb") as f:
    doc_tfidf_values = pickle.load(f)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def few_shot(mask):
    mask = mask.numpy()
    true_index = np.where(mask==True)[0]
    np.random.shuffle(true_index)
    print("The shots are:", args.shots)
    few_mask = true_index[0:args.shots] #random pick few answers
    new_mask = np.zeros(mask.shape)
    new_mask[few_mask]= True
    return torch.tensor(new_mask)

def remove_high_freq_word(word_freq, stop_words):
    high_fre_word =[]
    k_keys_sorted = heapq.nlargest(args.remove_high_freq, word_freq.items(), key=itemgetter(1))
    for tu in k_keys_sorted:
        high_fre_word.append(tu[0])
    print("high_fre_word", high_fre_word)
    stop_words = stop_words.union(set(high_fre_word))
    
    return stop_words


def Jaccard_Similarity(doc1, doc2): 
    
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split()) 
    words_doc2 = set(doc2.lower().split())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

def clean(str,stop_words):
        
    str_list_words = str.split()
    clean_list_words = []
    for word in str_list_words:
        word = re.sub(r"[^a-zA-Z ]+", '', word)#.lower()
        if word not in stop_words:
            word = stemmer.stem(word)
            clean_list_words.append(word)
        
    return clean_list_words


def clean_remove(str, word_freq,stop_words):
        
    str_list_words = str.split()
    clean_list_words = []
    #high_stop_words = remove_high_freq_word(word_freq, stop_words)
    for word in str_list_words:
        word = re.sub(r"[^a-zA-Z ]+", '', word)#.lower()
        if word not in stop_words:
            word = stemmer.stem(word)
            if word_freq[word] >= args.low_fre_word:
                clean_list_words.append(word)
        
    return clean_list_words

def build_set(triplet_list):
    
    words_raw = set()
    entities = set()
    relation = set()
    word_freq = {}
    for triplet in triplet_list:
        #print("triplet", triplet)
        if triplet != None:
            try:
                entity1, rel, entity2 = triplet
            except:
                entity1, rel, *entity2 = triplet
                entity2 = ' '.join(entity2)


            if entity1 != None:
                entity1_words = clean(entity1,stop_words)
                words_raw.update(entity1_words)
                for word in entity1_words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                entities.add(entity1)
            if entity2 != None:
                entity2_words = clean(entity2,stop_words)
                words_raw.update(entity2_words)
                for word in entity2_words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                entities.add(entity2)
            if rel != None:
                rel_words = clean(rel,stop_words)
                words_raw.update(rel_words)
                for word in rel_words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                relation.add(rel)
    #print(word_freq)
    if args.remove_high_freq > 0:
        high_stop_words = remove_high_freq_word(word_freq, stop_words)
    else:   
        high_stop_words = stop_words
    
    words = set() 
    for word in sorted(words_raw):
        if word_freq[word] >= args.low_fre_word:
             words.add(word)
    # Step 2: form a vocab based on these words and save the vocab to a txt file
    with open(f'./data/corpus/{args.dataset}_vocab.txt', "w") as f:
        for word in sorted(words):
            f.write(word + "\n")
    
    return words, word_freq, entities, relation, high_stop_words

def build_adj_matrix(abstract_list, doc_list, triplet_list):
    # Step 1: split the triplets into words

    words, word_freq, entities, relation, stop_words = build_set(triplet_list)

    # Step 3: using all the vocab to build an adjacent matrix
    vocab_size = len(words)
    adj_matrix = np.zeros((vocab_size, vocab_size), dtype=int)
    word_to_index = {word: i for i, word in enumerate(sorted(words))}
    
    #with open(f'./ablation_study/{args.dataset}.pickle', 'wb') as handle:
    #    pickle.dump(word_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    entity_to_index = {entity: i + len(words) for i, entity in enumerate(sorted(entities))}

    # Step 4: connect the entities and relations with edges
    edges = defaultdict(int)
    for triplet in triplet_list:
        if triplet != None:
            try:
                entity1, rel, entity2 = triplet
            except:
                entity1, rel, *entity2 = triplet
                entity2 = ' '.join(entity2)
            
            
            if entity1 != None and entity2 != None:
                
                clean_entity1 = clean_remove(entity1, word_freq,stop_words)
                clean_entity2 = clean_remove(entity2, word_freq,stop_words)
              
                for word1 in clean_entity1:
                    for word2 in clean_entity2:
                        
                        if word1 != word2:
                            word1_index = word_to_index[word1]
                            word2_index = word_to_index[word2]
                            edge1 = (word1_index, word2_index)
                            edge2 = (word2_index, word1_index)
                            edges[edge1] += 10 
                            edges[edge2] += 10 
            
            if entity1 != None and rel != None:
                clean_entity1 = clean_remove(entity1, word_freq,stop_words)
                clean_rel = clean_remove(rel , word_freq,stop_words)
                for word1 in clean_entity1:
                    for word2 in clean_rel:
                        
                        if word1 != word2:
                            word1_index = word_to_index[word1]
                            word2_index = word_to_index[word2]
                            edge1 = (word1_index, word2_index)
                            edge2 = (word2_index, word1_index)
                            
                            edges[edge1] += 1 
                            edges[edge2] += 1 
             
            if entity2 != None and rel != None:   
                clean_entity2 = clean_remove(entity2, word_freq,stop_words)
                clean_rel = clean_remove(rel, word_freq,stop_words)    
                for word1 in clean_entity2:
                    for word2 in clean_rel:
                        
                        if word1 != word2:
                            word1_index = word_to_index[word1]
                            word2_index = word_to_index[word2]
                            edge1 = (word1_index, word2_index)
                            edge2 = (word2_index, word1_index)
                            
                            edges[edge1] += 1 
                            edges[edge2] += 1 
          
    # Step 5: connect words within the same entity with edges
    for entity in entities:
        clean_entity_words = clean_remove(entity, word_freq,stop_words)
        for i, word1 in enumerate(clean_entity_words):
            for j, word2 in enumerate(clean_entity_words):
                if i != j and word1!= word2:
                    word1_index = word_to_index[word1]
                    word2_index = word_to_index[word2]
                    edge1 = (word1_index, word2_index)
                    edge2 = (word2_index, word1_index)
                    edges[edge1] += 1 #/(len(clean_entity_words))
                    edges[edge2] += 1 #/(len(clean_entity_words))
    
    # Add edges to the adjacency matrix
    for edge, count in edges.items():
        row, col = edge
        adj_matrix[row, col] = count
        
    pooling_index = torch.zeros((vocab_size, len(doc_list)))
    
    for i, doc in enumerate(doc_list):
        
        doc = doc.replace(';',', ')
        doc = clean(doc,stop_words)
        for word in doc:
            try:
                pooling_index[word_to_index[word]][i] = doc_tfidf_values[i].get(word, 0)
                #pooling_index[word_to_index[word]][i] =+1
            except:
                pass
        
    pooling_index = pooling_index/(pooling_index.sum(dim=0)+0.01)   
    
    return torch.tensor(adj_matrix), pooling_index, word_to_index

def build_feature(doc_list, triplet_list, word_to_index):
    
    words,_, _, _, _ = build_set(triplet_list)
    num_node = len(words) #+ len(doc_list)
    emb_size = args.emb_size
    
    emb_matrix = torch.rand((num_node,emb_size))

    feature_matrix = torch.eye(num_node)
    
    return feature_matrix, emb_matrix

def build_label(dataset):
    label_set = set()
    label_set = set()
    label_list = []
    f = open('data/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        temp = line.split("\t")
        label_set.add(temp[2].strip())
        label_list.append(temp[2].strip())
    f.close()
    
    unique_labels = np.unique(label_list)
    label_tensor = torch.tensor(np.searchsorted(unique_labels, label_list))
    
    
    return label_tensor

def build_graph(dataset):
    
    abs_doc_list, KG_doc_list, KG_list = read_KG(args.dataset)
    adj, pool_index, word_to_index = build_adj_matrix(abs_doc_list, KG_doc_list, KG_list)
    edge_index = dense_to_sparse(adj)[0]
    edge_attr = dense_to_sparse(adj)[1].float()
    feature_matrix, emb_matrix = build_feature(KG_doc_list, KG_list, word_to_index)
    doc_train_mask, doc_val_mask, doc_test_mask = split_shuffle(dataset)
    if args.usewhole:
        print("Using the whole graph")
        train_mask = doc_train_mask.bool()
    else:
        train_mask = few_shot(doc_train_mask).bool()
    print("training dataset samples:", train_mask.sum())
    val_mask = doc_val_mask.bool()
    test_mask = doc_test_mask.bool()
    doc_label = build_label(args.dataset)
    y_label = doc_label.long()
    print(train_mask)
    print(val_mask)
    print(test_mask)
    doc_graph = Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_attr, y = y_label, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, pool_index= pool_index, emb_matrix=emb_matrix)
    
    return doc_graph
