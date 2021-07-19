import os
import sys

import numpy as np
# import torch
# import torch.utils.data as utils
# import torchvision
# import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from colour import Color
from tqdm import tqdm

DATA_PATH = '../../data/ColorMNIST'
os.makedirs(DATA_PATH, exist_ok=True) 

np.random.seed(0)
red = Color("red")
colors = list(red.range_to(Color("purple"),10))
colors = [np.asarray(x.get_rgb()) for x in colors]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# splitting train dataset between train and validation in 9:1 ratio
# x_train_train, x_train_val, y_train_train, y_train_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

num_samples = len(x_train)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
color_y = np.empty(num_samples, dtype = np.int16)
for i in tqdm(range(num_samples)):
    my_color  = colors[int(y_train[i])]
    color_x[i] = x_train[i].astype(np.float32)[np.newaxis] * my_color[:, None, None]
    color_y[i] = y_train[i]
np.save(os.path.join("../../data/ColorMNIST", "train_x.npy"), color_x)
print('train_x shape', color_x.shape)
np.save(os.path.join("../../data/ColorMNIST", "train_y.npy"), color_y)
print('train_y shape', color_y.shape)

# num_samples = len(x_train_val)
# color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
# color_y = np.empty(num_samples, dtype = np.int16)
# for i in tqdm(range(num_samples)):
#     my_color  = colors[9 - int(y_test[i])]
#     color_x[i] = x_train_val[i].astype(np.float32)[np.newaxis] * my_color[:, None, None]
#     color_y[i] = y_train_val[i]
# np.save(os.path.join("../../data/ColorMNIST", "val_x.npy"), color_x)
# print('val_x shape', color_x.shape)
# np.save(os.path.join("../../data/ColorMNIST", "val_y.npy"), color_y)
# print('val_y shape', color_y.shape)

num_samples = len(x_test)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
color_y = y_test.copy()
for i in tqdm(range(num_samples)):
    color_x[i] = x_test[i].astype(np.float32)[np.newaxis] * colors[9 - color_y[i]][:, None, None]
np.save(os.path.join("../../data/ColorMNIST", "test_x.npy"),  color_x)
print('test_x shape', color_x.shape)
np.save(os.path.join("../../data/ColorMNIST", "test_y.npy"), color_y)
print('test_y shape', color_y.shape)
