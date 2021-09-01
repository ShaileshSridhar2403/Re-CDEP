import argparse
import os

import dagshub
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from train_mnist_decoy import Net, CustomModel

DATA_PATH = '../../data/DecoyMNIST'
SAVE_PATH = './results'
MODEL_PATH = '../../models/DecoyMNIST'
os.makedirs(SAVE_PATH, exist_ok=True)

# Evaluation settings
parser = argparse.ArgumentParser(description='Tensorflow DecoyMNIST Example')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--regularizer_rate', type=float, default=0.0, metavar='N',
                    help='how heavy to regularize lower order interaction (AKA color)')
parser.add_argument('--test_decoy', type=int, default=1,
                    help='Use the decoy test data during testing')
args = parser.parse_args()

regularizer_rate = args.regularizer_rate

if args.test_decoy:
    data_x = np.load(os.path.join(DATA_PATH, "test_x_decoy.npy"))
else:
    data_x = np.load(os.path.join(DATA_PATH, "test_x.npy"))
data_y = np.load(os.path.join(DATA_PATH, "test_y.npy"))

network = Net()
model = CustomModel(network, regularizer_rate)

model.load_weights(os.path.join(MODEL_PATH, 'model_r{}_s{}_t{}'.format(args.regularizer_rate, args.seed, args.test_decoy)))

predictions = model.model.predict(data_x)

report = classification_report(data_y, np.argmax(predictions, axis=1), output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(SAVE_PATH, "results_r{}_s{}_t{}.csv".format(args.regularizer_rate, args.seed, args.test_decoy)))

accuracy =  accuracy_score(data_y, np.argmax(predictions, axis=1))
print("Accuracy:", accuracy)

with dagshub.dagshub_logger() as logger:
    logger.log_hyperparams(model_class="mnist_classifier_decoy")
    logger.log_hyperparams({'model': {'regularizer_rate':regularizer_rate}})
    logger.log_metrics({'Accuracy': accuracy})
