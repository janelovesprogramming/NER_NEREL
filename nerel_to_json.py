import json
import os
import argparse
import pandas as pd

import pickle
import torch
import nltk
from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer
nltk.download('punkt')

from tqdm import tqdm

# Download tokenizer for Russian
ru_tokenizer = load("tokenizers/punkt/russian.pickle") 

word_tokenizer = NLTKWordTokenizer()

nerel_parser = argparse.ArgumentParser(description = "NEREL to JSON file")

nerel_parser.add_argument('--dataset_path', type = str, required = True, help = "Path to NEREL dataset")

nerel_parser.add_argument('--tags_path', type = str, required = True, help = 'Path to tags file')

nerel_parser.add_argument('--output_path', type = str, default = None, help = "Output path")

args = nerel_parser.parse_args()

dataset_path = args.dataset_path
output_path = args.output_path
tags_path = args.tags_path


with open(tags_path, "r") as tags_file:
    tags = json.loads(tags_file.read())


sets = ["train", "dev", "test"]

for ds in sets:
    
    data = []
    print(ds + " set:")
    
    jsonpath = os.path.join(output_path + '_' + ds + ".json")
    dataset_path = os.path.join(dataset_path + '/' + ds)
    jsondir = os.path.dirname(jsonpath)

    if not os.path.exists(jsondir):

        os.makedirs(jsondir)
        
    jsonfile = open(jsonpath, "w", encoding='UTF-8')
    doc_count = 0
    entities_count = 0
    doc_ids = []
    doc = []
    for ad, dirs, files in os.walk(dataset_path):
        tokens_all = []
        for f in tqdm(files): 
            s_token = []
            if f[-4:] == '.ann':
                
                try:
                    if os.stat(dataset_path + '/' + f).st_size == 0:
                        continue
                    
                    annfile = open(dataset_path + '/' + f, "r", encoding='UTF-8')
                    txtfile = open(dataset_path + '/' + f[:-4] + ".txt", "r", encoding='UTF-8')
                    txtdata = txtfile.read()
                    
                    entity_types = []
                    entity_start_chars = []
                    entity_end_chars = []
                    tokens = []
                    for line in annfile:
                        
                        line_tokens = line.split()
                        
                        if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                            try:
                                if line_tokens[1] in tags:
                                    token = line_tokens[-1]
                                    entity_type = line_tokens[1]
                                    start_char = int(line_tokens[2])
                                    end_char = int(line_tokens[3])

                                    tokens.append(token)
                                    entity_types.append(entity_type)
                                    entity_start_chars.append(start_char)
                                    entity_end_chars.append(end_char)
                                
                            except ValueError:
                                continue 

                        try:
                            assert len(entity_types) == len(entity_start_chars) == len(entity_end_chars)

                        except AssertionError:
                            raise AssertionError
                        

                    annfile.close()
                    txtfile.close()

                    doc_entities = {

                        'text': txtdata,
                        'tokens':tokens,
                        'entity_types': entity_types,
                        'entity_start_chars': entity_start_chars,
                        'entity_end_chars': entity_end_chars,
                        'id': f[:-4],
                    }

                    
                    words = word_tokenizer.tokenize(txtdata)
                    spans = word_tokenizer.span_tokenize(txtdata)
                    i = 0
                    
                    entity_types_all = []
                    for nr, w in zip(spans, words):
                        if nr[0] in doc_entities['entity_start_chars']:
                            s_token.append([tokens[i], doc_entities['entity_types'][i]])
                            i += 1
                        else:
                            s_token.append([w, '0'])

                    
                    doc_count += 1
                    doc_ids.append(f[:-4])

                except FileNotFoundError:
                    pass
                tokens_all.append(s_token)

    print(f"Docs: {doc_count}")
    doc.append(tokens_all)
   
    with open("dataset/nerel_" + ds, "wb") as fp:
        pickle.dump(doc, fp)

    jsonfile.write(json.dumps(doc, ensure_ascii = False) + '\n')
    jsonfile.close()