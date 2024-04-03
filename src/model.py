import pandas as pd
import numpy as np
import random
from copy import deepcopy

import pickle
import json

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score

import wandb


class BERTNER(object):
    def __init__(self, model_name:str, labels:list):
        self.model_name = model_name
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self.labels))

    def tokenize_labels(self, data:dict):
        tokenized_inputs = self.tokenizer(
            data["tokens"], truncation=True, is_split_into_words=True, padding=True
        )
        labels = []
        
        for i, label in enumerate(data["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    