"""
Sample Usage:
```
python evaluate.py --regularizer_rate 10
```
"""
import argparse
import os
import sys
import time

import dagshub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import randint
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             roc_curve)
from tensorflow.keras.applications.vgg16 import VGG16

from config import config as data
from train_CDEP import create_classification_model

sys.path.append('../../src')
import cd
import utils

model_path = os.path.join(data["model_folder"], "ISIC")
dataset_path = os.path.join(data["data_folder"], "calculated_features")

datasets, weights = utils.load_precalculated_dataset(dataset_path)

parser = argparse.ArgumentParser(description='ISIC Skin cancer for CDEP')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--regularizer_rate', type=float, default=0.0, metavar='N',
                    help='hyperparameter for CDEP weight - higher means more regularization')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
args = parser.parse_args()

regularizer_rate = args.regularizer_rate

model = create_classification_model()
model.load_weights(os.path.join(model_path,'model_r{}_s{}'.format(args.regularizer_rate, args.seed)))

if regularizer_rate == -1:  # -1 means that we train only on data with no patches
    datasets['train'] = datasets['train_no_patches']

try:
    dataset_sizes = {}
    for x in datasets:
        dataset_sizes[x] = len(list(datasets[x].as_numpy_iterator()))
except:
    print("\n error_set", x, "\n", datasets[x])

val_y = [y for x, y, z in datasets['val']]
test_y = [y for x, y, z in datasets['test']]

val_predictions = [int(y > 0.5) for y in model.predict(datasets["val"].batch(args.batch_size))]
test_predictions = [int(y > 0.5) for y in model.predict(datasets["test"].batch(args.batch_size))]

val_prediction_probs = [y for y in model.predict(datasets["val"].batch(args.batch_size))]
test_prediction_probs = [y for y in model.predict(datasets["test"].batch(args.batch_size))]

print("Printing Validation Scores............")
print(classification_report(val_y,
                            val_predictions, target_names=["Not Cancer", "Cancer"]))

print("Printing Test Scores.........")
print(classification_report(test_y,
                            test_predictions, target_names=["Not Cancer", "Cancer"]))

# fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y, test_prediction_probs)
# auc_output = auc(fpr_keras, tpr_keras)

# print(f"\nTest AUC {auc_output}")

test_accuracy = accuracy_score(test_y,test_predictions)
print("Test Accuracy :",test_accuracy)

if regularizer_rate > 0:
    model_class = "ISIC CDEP"
else:
    model_class = "ISIC Vanilla"

with dagshub.dagshub_logger() as logger:
    logger.log_hyperparams(model_class=model_class)
    logger.log_hyperparams({'model': {'regularizer_rate':regularizer_rate}})
    logger.log_metrics({'Accuracy': test_accuracy})
