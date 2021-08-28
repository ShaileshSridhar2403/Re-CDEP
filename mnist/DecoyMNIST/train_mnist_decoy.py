from __future__ import print_function

import argparse
import copy
import os
import pickle as pkl
import sys
import time
from copy import deepcopy
from os.path import join as oj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

sys.path.append('../../src')
import cd
# from score_funcs import eg_scores_2d, gradient_sum

DATA_PATH = '../../data/DecoyMNIST'
SAVE_PATH = './results'
MODEL_PATH = '../../models/DecoyMNIST'
os.makedirs(SAVE_PATH, exist_ok=True)


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(20, 5, data_format='channels_first')  # input channels = 1
        self.conv2 = tf.keras.layers.Conv2D(50, 5, data_format='channels_first')  # input channels = 20
        self.fc1 = tf.keras.layers.Dense(256)  # input shape = 4*4*50
        self.fc2 = tf.keras.layers.Dense(10)  # input shape = 256

    def call(self, x):
        x = tf.nn.relu(self.conv1(x))
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, data_format='channels_first')(x)
        x = tf.nn.relu(self.conv2(x))
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, data_format='channels_first')(x)
        x = tf.reshape(x, [-1, 4*4*50])
        x = tf.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def logits(self, x):
        x = tf.nn.relu(self.conv1(x))
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = tf.nn.relu(self.conv2(x))
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = tf.reshape(x, [-1, 4*4*50])
        x = tf.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomModel(tf.keras.Model):
    def __init__(self, model, regularizer_rate):
        super(CustomModel, self).__init__()
        self.model = model
        self.regularizer_rate = regularizer_rate

    def train_step(self, data):
        x, y = data
        x = tf.cast(x, dtype=tf.float64)

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # print("Before CDEP Loss:", loss)

            if self.regularizer_rate != 0:
                add_loss = tf.zeros(1,)

                if args.grad_method == 0:
                    rel, irrel = cd.cd(blob, x, self.model)
                    add_loss += tf.math.reduce_mean(
                        tf.cast(
                            tf.nn.softmax(tf.stack([tf.reshape(rel, [-1]), tf.reshape(irrel, [-1])], axis=1), axis=1)[:, 0],
                            dtype=tf.float32))
                    loss = (self.regularizer_rate*add_loss + loss)
                    # print("Add Loss:", add_loss)
                    # print("Loss:", loss)

                # # RRR
                # elif args.grad_method == 1:
                #     add_loss += gradient_sum(data, target, torch.FloatTensor(blob).to(device),  model, F.nll_loss)
                #     (self.regularizer_rate*add_loss).backward()
                #     # print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
                #     optimizer.step()
                #     loss = F.nll_loss(output, target)
                #     loss.backward()

                # # Expected Gradients
                # elif args.grad_method == 2:
                #     for j in range(len(data)):
                #         add_loss += (eg_scores_2d(model, data, j, target, num_samples) * torch.FloatTensor(blob).to(device)).sum()
                #     (self.regularizer_rate*add_loss).backward()
                #     # print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
                #     optimizer.step()
                #     loss = F.nll_loss(output, target)
                #     loss.backward()

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # if batch_idx % args.log_interval == 0:
        #     pred = output.argmax(dim=1, keepdim=True)
        #     acc = 100.*pred.eq(target.view_as(pred)).sum().item()/len(target)
        #     # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: ({:.0f}%), CD Loss: {:.6f}'.format(
        #         # epoch, batch_idx * len(data), len(train_loader.dataset),
        #         # 100. * batch_idx / len(train_loader), loss.item(),acc,   add_loss.item()))
        #     s.losses_train.append(loss.item())
        #     s.accs_train.append(acc)
        #     s.cd.append(add_loss.item())

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def build_graph(self):
        x = tf.keras.Input(shape=(1, 28, 28))
        return tf.keras.Model(inputs=[x], outputs=self.model.call(x))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Tensorflow DecoyMNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--regularizer_rate', type=float, default=0.0, metavar='N',
                        help='how heavy to regularize lower order interaction (AKA color)')
    parser.add_argument('--grad_method', type=int, default=0, metavar='N',
                        help='how heavy to regularize lower order interaction (AKA color)')
    parser.add_argument('--test_decoy', type=int, default=1,
                        help='Use the decoy test data during testing')
    # parser.add_argument('--gradient_method', type=string, default="CD", metavar='N',
                        # help='what method is used')
    args = parser.parse_args()

    regularizer_rate = args.regularizer_rate
    num_blobs = 1
    num_samples = 200

    # sys.exit()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    train_x_tensor = tf.convert_to_tensor(np.load(oj(DATA_PATH, "train_x_decoy.npy")))
    train_y_tensor = tf.convert_to_tensor(np.load(oj(DATA_PATH, "train_y.npy")), dtype=tf.int64)
    complete_dataset = tf.data.Dataset.from_tensor_slices((train_x_tensor, train_y_tensor))  # create your train-validation dataset
    complete_dataset = complete_dataset.shuffle(buffer_size=len(complete_dataset))

    num_train = int(len(complete_dataset)*.9)
    num_test = len(complete_dataset) - num_train
    tf.random.set_seed(0)
    # TODO: Check if split between train and test happens randomly - otherwise might affect accuracy

    # create your train dataloader
    train_loader = complete_dataset.take(num_train)
    train_loader = train_loader.batch(args.batch_size)
    # TODO: Check if we need to shuffle train dataset

    # create your test dataloader
    test_loader = complete_dataset.skip(num_train)
    test_loader = test_loader.batch(args.batch_size)
    # TODO: Check if we need to shuffle test dataset

    test_x_tensor = tf.convert_to_tensor(np.load(oj(DATA_PATH, "test_x_decoy.npy")))
    test_y_tensor = tf.convert_to_tensor(np.load(oj(DATA_PATH, "test_y.npy")), dtype=tf.int64)
    val_dataset = tf.data.Dataset.from_tensor_slices((test_x_tensor, test_y_tensor))  # create your test dataset
    val_dataset = val_dataset.shuffle(buffer_size=len(val_dataset))
    # create your validation dataloader
    val_loader = val_dataset.take(len(val_dataset))
    val_loader = val_loader.batch(args.test_batch_size)

    # make the sampling thing
    blob = np.zeros((28,28))
    size_blob = 5
    blob[:size_blob, :size_blob ] = 1
    blob[-size_blob:, :size_blob] = 1
    blob[:size_blob, -size_blob:] = 1
    blob[-size_blob:, -size_blob:] = 1

    network = Net()
    model = CustomModel(network, regularizer_rate)

    # `pydot` and `graphviz` installation is prone to error. Please refer to this link to install them
    # correctly: https://stackoverflow.com/questions/47605558/importerror-failed-to-import-pydot-you-must-install-pydot-and-graphviz-for-py
    # tf.keras.utils.plot_model(model.build_graph(), show_shapes=True, show_dtype=True, expand_nested=True)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # TODO: using a weight_decay of 0.001 (suggested by the paper to use in all experiments) gives ~76%
    optimizer = tfa.optimizers.AdamW(weight_decay=0.0001, epsilon=1e-08)  #, lr=args.lr, momentum=args.momentum)

    # TODO: A different reduction function is used in the test_step -> SUM instead of SUM_OVER_BATCH
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    best_model_weights = None
    best_test_loss = 100000
    patience = 2
    cur_patience = 0

    start = time.time()

    model.compile(optimizer=optimizer, loss=scce, metrics=['accuracy'], run_eagerly=True)
    H = model.fit(train_loader, epochs=args.epochs, validation_data=test_loader)
    model.save_weights(os.path.join(MODEL_PATH, 'model_r{}_s{}_t{}'.format(args.regularizer_rate, args.seed, args.test_decoy)))

    end = time.time()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, args.epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, args.epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, args.epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, args.epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(SAVE_PATH, "plot_r{}_s{}_t{}.png".format(args.regularizer_rate, args.seed, args.test_decoy)))
