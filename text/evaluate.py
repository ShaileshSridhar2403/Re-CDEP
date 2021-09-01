"""
Sample Usage:
```
python evaluate.py --decoy a the --batch_size 64 --signal_strength 200 --noise_type bias
```

Note:
    1. Use the same batch size that was used while training
"""

import argparse
import os

import dagshub
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from train import CustomModel, LSTMSentiment, load_data, seed_value

model_path = '../models/Text'

# Evaluation settings
parser = argparse.ArgumentParser(description='Tensorflow Text Example')
parser.add_argument('--path_to_glove_file', default='../data/Text/glove/glove.6B.300d.txt')
parser.add_argument('--data_path', type=str, default='../data/Text/data/bias/')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--decoy', nargs='+')
parser.add_argument('--signal_strength', type=float, default=100)
parser.add_argument('--noise_type', type=str)
args = parser.parse_args()

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
tokenizer.fit_on_texts(data['train'])
seq_test = tokenizer.texts_to_sequences(data['test'])
padded_seq_test = pad_sequences(seq_test, padding='post', truncating='post')
test_labels = np.array(labels['test'])
test_dataset = tf.data.Dataset.from_tensor_slices((padded_seq_test, test_labels))

model = LSTMSentiment()

embeddings_index = {}
with open(args.path_to_glove_file, encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

hits = 0
misses = 0
num_tokens = len(tokenizer.word_index)
model.embedding.input_dim = num_tokens + 2
embedding_dim = 300
embedding_matrix = np.zeros((num_tokens+2, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
model.embedding.embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix)

opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.load_weights(os.path.join(model_path, 'model_r{}_s{}_n{}'.format(args.signal_strength, seed_value, args.noise_type)))

metrics = model.evaluate(test_dataset.batch(args.batch_size))

line = f'r{args.signal_strength}_s{seed_value}_n{args.noise_type}'
f = open('run_results_evaluate.txt', 'a')
line = line+f' result:{metrics[1]}\n'
f.write(line)
f.close()

with dagshub.dagshub_logger() as logger:
    logger.log_hyperparams(model_class=f"SST_{args.noise_type}")
    logger.log_hyperparams({'model': {'regularizer_rate':args.signal_strength}})
    logger.log_metrics({'Accuracy': metrics[1]})
