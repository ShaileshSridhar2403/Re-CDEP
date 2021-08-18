import os

import numpy as np
from PIL import Image

DATA_PATH = "../../data/DecoyMNIST"

data_x = np.load(os.path.join(DATA_PATH, "train_x_decoy.npy"))
data_y = np.load(os.path.join(DATA_PATH, "train_y.npy"))
print(data_x[0:10])

PIL_image = Image.fromarray(np.uint8(data_x[0][0]))
PIL_image.save('train_decoy.png')

print(data_y[0])


data_x = np.load(os.path.join(DATA_PATH, "test_x_decoy.npy"))
data_y = np.load(os.path.join(DATA_PATH, "test_y.npy"))
print(data_x[0:10])

PIL_image = Image.fromarray(np.uint8(data_x[0][0]))
PIL_image.save('test_decoy.png')

print(data_y[0])
