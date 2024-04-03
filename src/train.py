import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, TrainerCallback

from datasets import Dataset, DatasetDict
from copy import deepcopy

from model import BERTNER
from constants import MODEL_NAME, PATH_MODEL
from metrics import compute_metrics
from dataset import DatasetNER

import wandb
 
# Opening JSON file
wandb.init(
    # set the wandb project where this run will be logged
    project="NER",
    config={
        "learning_rate": 5e-5,
        "architecture": "BERT",
        "dataset": "NEREL-v1",
        "epochs": 5,
    }
)

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
        
def data_collator(data):
    input_ids = [torch.tensor(item["input_ids"]) for item in data]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in data]
    labels = [torch.tensor(item["labels"]) for item in data]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    conf = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels}

    return conf
        
training_args = TrainingArguments(
    output_dir = "results/",
    evaluation_strategy = "steps",
    eval_steps = 100,
    save_steps = 500,
    num_train_epochs = 5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    logging_steps = 100,
    learning_rate = 5e-5,
    load_best_model_at_end = True,
    metric_for_best_model = "precision",
)

ds = DatasetNER()

ds_dict = ds.make_dataset_dict()

labels = ds.label_all

bertner = BERTNER(MODEL_NAME, labels)

tokenized_datasets = ds_dict.map(bertner.tokenize_labels, batched=True)
tokenizer = bertner.tokenizer
model = bertner.model


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight = torch.tensor([0.2, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0])).cuda()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["dev"],
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics,
)

trainer.add_callback(CustomCallback(trainer)) 

trainer.train()

trainer.evaluate(metric_key_prefix='test',
                eval_dataset=tokenized_datasets["test"])

torch.save(model.state_dict(), PATH_MODEL)

wandb.finish()
