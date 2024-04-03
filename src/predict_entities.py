import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

from tqdm import tqdm

from constants import PATH_MODEL, DATASET, OUTPUT_TEST_PATH


class EntitiesNER(object):

   def __init__(self):
      model_name = "google-bert/bert-base-multilingual-cased"
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.label_list = ['0', 'AGE', 'AWARD', 'CITY', 'COUNTRY', 'LOCATION', 'ORGANIZATION', 'PERSON', 'PROFESSION', 'RELIGION', 'WORK_OF_ART']
      self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(self.label_list))
      self.model.load_state_dict(torch.load(PATH_MODEL))


   def predict_one_text(self, text:str) -> dict:
      tokenized_input = self.tokenizer(text, return_tensors="pt").to(self.model.device)
      outputs = self.model(**tokenized_input)
      predicted_labels = outputs.logits.argmax(-1)[0]
      label_map = {label: i for i, label in enumerate(self.label_list)}
      named_entities = {}
      words = []
      labels = []
      inv_map = {v: k for k, v in label_map.items()}
      for token, label in zip(tokenized_input["input_ids"][0], predicted_labels.tolist()):
         if label != 0:
            word = str(self.tokenizer.decode([token]))
            words.append(word)
            labels.append(inv_map[label])

      return (words, labels)
   
   def predict_test(self):
      df = pd.read_csv(DATASET)
      text = df.iloc[0].text[0:511]
      ner = self.predict_one_text(text)
      return ner

   def predict_testset(self) -> dict:
      df = pd.read_csv(DATASET)
      for index, row in tqdm(df.iterrows()):
         text = row['text'][0:511]
         p = self.predict_one_text(text)
         df.loc[index, 'names'] = str(p[0])
         df.loc[index, 'tags'] = str(p[1])
      df.to_csv(OUTPUT_TEST_PATH, index = False, sep = ';')
      return df

   
ener = EntitiesNER()
ener.predict_testset()
