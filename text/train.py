import argparse
import os
import pdb
import random
import time
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import metrics
from tensorflow.python.keras.utils.generic_utils import default
from tensorflow.python.training import optimizer

from model import CustomModel, LSTMSentiment

data_path = '../data/Text/data/bias/'
model_path = '../models/Text/'

seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_glove_file', default='')
    parser.add_argument('--data_path', type=str, default='../data/Text/data/bias/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--decoy', nargs='+')
    parser.add_argument('--signal_strength', type=float, default=100)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--noise_type', type=str)
    parser.add_argument('--quick_run', action='store_true')

    args = parser.parse_args()
    return args


def download_glove():
    import os
    os.system('wget http://nlp.stanford.edu/data/glove.6B.zip')
    os.system('mkdir -p glove')
    os.system('unzip -q glove.6B.zip -d glove/')
    os.system('mkdir -p ../data/Text')
    os.system('mv glove ../data/Text')
    os.system('rm glove.6B.zip')


#0 positive 1 negative
def load_data(data_path : str, files : dict):
    """reads text corpus and along with labels and stores in dictionary"""
    data = defaultdict(list)
    labels = defaultdict(list)
    for split, file in files.items():
        path = data_path+file
        f = open(path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            text, label  = line[:-2], line[-1]
            label = int(label)
            data[split].append(text)
            labels[split].append(label)
    
    return data, labels


def train(args, model, files):

    data, labels = load_data(args.data_path, files)
    tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
    tokenizer.fit_on_texts(data['train'])

    tokenizer.word_index[args.decoy[0]]
    model.decoy = (tokenizer.word_index[args.decoy[0]], tokenizer.word_index[args.decoy[1]])
    # if args.path_to_glove_file == '':
    #     download_glove()
    args.path_to_glove_file = '../data/Text/glove/glove.6B.300d.txt'
    
    embeddings_index = {}
    with open(args.path_to_glove_file, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    hits = 0
    misses = 0
    num_tokens = len(tokenizer.word_index)
    #pdb.set_trace()
    model.model.embedding.input_dim = num_tokens + 2
    #pdb.set_trace()
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
    #pdb.set_trace()
    model.model.embedding.embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix)
    print("Converted %d words (%d misses)" % (hits, misses))
    #pdb.set_trace()
    seq_train = tokenizer.texts_to_sequences(data['train'])
    seq_val = tokenizer.texts_to_sequences(data['val'])
    seq_test = tokenizer.texts_to_sequences(data['test'])
    #pdb.set_trace()
    padded_seq_train = pad_sequences(seq_train, padding='post', truncating='post')
    padded_seq_val = pad_sequences(seq_val, padding='post', truncating='post')
    padded_seq_test = pad_sequences(seq_test, padding='post', truncating='post')
    train_labels = np.array(labels['train'])
    val_labels = np.array(labels['val'])
    test_labels = np.array(labels['test'])
    opt = tf.keras.optimizers.Adam()
    #opt = tfa.optimizers.AdamW(weight_decay=0.001)
    #pdb.set_trace()
    model.compile(optimizer=opt, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'], run_eagerly=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((padded_seq_train, train_labels))
    train_dataset.shuffle(len(train_dataset),seed=42)
    train_dataset = train_dataset.batch(args.batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((padded_seq_val, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((padded_seq_test, test_labels))

    if args.quick_run:
        train_dataset = train_dataset.take(10)

    es = EarlyStopping(monitor='val_total_val_loss', mode='min', verbose=1, patience=3)
    #ms = ModelCheckpoint(args.noise_type+'best.pt', monitor="val_accuracy", verbose=0, save_best_only=True)
    
    callbacks = [es]
    H = model.fit(train_dataset, validation_data=val_dataset.batch(args.batch_size), epochs=args.num_epochs, callbacks=callbacks)
    
    metrics = model.evaluate(test_dataset.batch(args.batch_size))
    # pdb.set_trace()
    return H, model, metrics


if __name__ == '__main__':
    args = parse_args()
    files_bias = {'train':'train_bias_SST.csv',  'val':'dev_bias_SST.csv', 'test':'test_bias_SST.csv'}
    files_random = {'train':'train_decoy_SST.csv',  'val':'dev_decoy_SST.csv', 'test':'test_decoy_SST.csv'}
    files_gender = {'train':'train_gender_SST.csv',  'val':'dev_gender_SST.csv', 'test':'test_gender_SST.csv'}
    files = None
    if args.noise_type == 'bias':
        files = files_bias
    elif args.noise_type == 'random':
        files = files_random
    elif args.noise_type == 'gender':
        files = files_gender

    
    model = LSTMSentiment()
    custom_model = CustomModel(model=model, decoy=args.decoy, signal_strength=args.signal_strength)

    since = time.time()
    H, model, metrics = train(args, custom_model, files)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.save_weights(os.path.join(model_path, 'model_r{}_s{}_n{}'.format(args.signal_strength, seed_value, args.noise_type)))

    line = f'{args.noise_type}_ss{args.signal_strength}_ep{args.num_epochs}'
    f = open('run_results.txt', 'a')
    line = line+f' result:{metrics[1]}\n'
    f.write(line)
    f.close()
    
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(np.arange(0, args.num_epochs), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, args.num_epochs), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, args.num_epochs), H.history["accuracy"], label="train_acc")
    # plt.plot(np.arange(0, args.num_epochs), H.history["val_accuracy"], label="val_acc")
    # plt.title("Training Loss and Accuracy on Dataset")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")

    # plt.savefig(line+".png")
