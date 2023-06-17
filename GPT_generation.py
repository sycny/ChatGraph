from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import sys
import argparse
#from openaigpt import Chatting, Instruct
from batch_gpt import Chatting, Instruct
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='test_short' )
parser.add_argument('--model', type=str, default='ChatGPT')
parser.add_argument('--batch_size', type=int, default= 256)

args = parser.parse_args()
print(f"Using the {args.dataset} with {args.model}")


def log_refined(refined_input):
    
    #refined_input = '\n'.join(input) 
    f = open('data/corpus/' + args.dataset + '_KG.txt', 'a')
    # f = open('data/wiki_long_abstracts_en_text.txt', 'r')
    f.write(refined_input + '\n')     
    f.close()
    
def list_refined(refined_input):
    
    #refined_input = '\n'.join(input) 
    f = open('data/corpus/' + args.dataset + '_KG_index.txt', 'a')
    # f = open('data/wiki_long_abstracts_en_text.txt', 'r')
    f.write(refined_input + '\n')     
    f.close()
    

def extract(input):
    
    '''          
    instruction =  "You are a knowledge graph extractor, and your task is to extract a knowledge graph from a given text."+\
                   "A knowledge graph is defined as a set of entities and relations, where entities represent real-world objects or abstract concepts, and relations represent the relationships between entities."+\
                   "Each entity and relation should be summarized as short as possible, and any stop words should be removed."+ "\n" +\
                            "To extract the knowledge graph, you can follow these steps:"+ "\n" +\
                            "(1). Identify the entities in the text. An entity can be a noun or a noun phrase that refers to a real-world object or an abstract concept. You can use a named entity recognition (NER) tool or a part-of-speech (POS) tagger to identify the entities."+ "\n" +\
                            "(2). Identify the relationships between the entities. A relationship can be a verb or a prepositional phrase that connects two entities. You can use dependency parsing to identify the relationships."+ "\n" +\
                            "(3). Extract the types and properties of the entities and their relationships. Types and properties provide a well-defined meaning to the entities and relationships. You can use a knowledge base or a domain-specific ontology to extract the types and properties."+ "\n" +\
                            "(4). Summarize each entity and relation as short as possible and remove any stop words."+ "\n" +\
                            "(5). Present the knowledge in the triplet format, such as: ('head entity', 'relation', 'tail entity')."+ "\n" +\
                    "If you cannot find any knowledge, please just return: None." 
    '''                
    instruction =  "You are a knowledge graph extractor, and your task is to extract a knowledge graph from a given text."+\
                   "A knowledge graph is defined as a set of entities and relations, where entities represent real-world objects or abstract concepts, and relations are verbs that represent the relationships between entities."+\
                   "Each entity and relation should be summarized as short as possible, and any stop words should be removed."+ "\n" +\
                   "Please make sure present the knowledge in the triplet format, such as: ('head entity', 'relation', 'tail entity')."+ "\n" +\
                   "If you cannot find any knowledge, please just return: None."+ "\n" +\
                   "Here is the content:"
                    

    # instruction = "You are a text classifier and your task is to classifiy a given text into the following categories: 'acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship',\
    # 'trade'. You should directly output the predicted label."

    KEY = "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"
    model = getattr(Chatting, args.model)(KEY, instruction)
    KG_list = []
    for batch in tqdm(model.batch_call(input, batch_size=args.batch_size)):
        for doc in batch:
            response = doc[1][0].replace('\n',';')
            log_refined(str(response))
        KG_list.extend(batch)
    return KG_list

if __name__ == "__main__":
    refined_doc_list = []
    f = open('data/corpus/' + args.dataset + '_abstract.txt', 'rb')
    print("Using the abstracted doc")
    # f = open('data/wiki_long_abstracts_en_text.txt', 'r')
    for line in f.readlines():  
        refine_doc = line.strip().decode('latin1')
        refined_doc_list.append(refine_doc)
        
    KG_list = extract(refined_doc_list)

    for Kg_index_pair in KG_list:
        list_refined(str(Kg_index_pair))

    
    f.close()

'''
f = open('data/corpus/' + args.dataset + '_abstract.txt', 'w')
# f = open('data/wiki_long_abstracts_en_text.txt', 'r')
f.write(clean_corpus_str)     
f.close()
'''