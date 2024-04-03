import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score

import wandb
from datasets import load_metric
metric = load_metric("seqeval")

def compute_metrics(p):
    label_list = ['0', 'AGE', 'AWARD', 'CITY', 'COUNTRY', 'LOCATION', 'ORGANIZATION', 'PERSON', 'PROFESSION', 'RELIGION', 'WORK_OF_ART']
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
      if(k not in flattened_results.keys()):
        flattened_results[k+"_f1"]=results[k]["f1"]

    return flattened_results

