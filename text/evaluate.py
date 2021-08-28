import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from train import CustomModel, LSTMSentiment, load_data, seed_value

model_path = '../models/Text'

# Evaluation settings
parser = argparse.ArgumentParser(description='Tensorflow Text Example')
parser.add_argument('--data_path', type=str, default='../data/Text/data/bias/')
parser.add_argument('--decoy', nargs='+')
parser.add_argument('--signal_strength', type=float, default=100)
parser.add_argument('--noise_type', type=str)
args = parser.parse_args()

model = LSTMSentiment()
custom_model = CustomModel(model=model, decoy=args.decoy, signal_strength=args.signal_strength)

custom_model.load_weights(os.path.join(model_path, 'model_r{}_s{}_n{}'.format(args.signal_strength, seed_value, args.noise_type)))

files = None
files_bias = {'train':'train_bias_SST.csv',  'val':'dev_bias_SST.csv', 'test':'test_bias_SST.csv'}
files_random = {'train':'train_decoy_SST.csv',  'val':'dev_decoy_SST.csv', 'test':'test_decoy_SST.csv'}
files_gender = {'train':'train_gender_SST.csv',  'val':'dev_gender_SST.csv', 'test':'test_gender_SST.csv'}
if args.noise_type == 'bias':
    files = files_bias
elif args.noise_type == 'random':
    files = files_random
elif args.noise_type == 'gender':
    files = files_gender
data, labels = load_data(args.data_path, files)
tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
seq_test = tokenizer.texts_to_sequences(data['test'])
padded_seq_test = pad_sequences(seq_test, padding='post', truncating='post')
test_labels = np.array(labels['test'])
test_dataset = tf.data.Dataset.from_tensor_slices((padded_seq_test, test_labels))
metrics = custom_model.evaluate(test_dataset.batch(args.batch_size))

line = f'{args.noise_type}_ss{args.signal_strength}_ep{args.num_epochs}'
f = open('run_results.txt', 'a')
line = line+f' result:{metrics[1]}\n'
f.write(line)
f.close()
