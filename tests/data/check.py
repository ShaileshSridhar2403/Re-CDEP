import numpy as np

train_x_decoy_original = np.load('../../data/ColorMNIST/train_x_decoy.npy')
train_x_decoy = np.load('../../data/DecoyMNIST/train_x_decoy.npy')

test_x_decoy_original = np.load('../../data/ColorMNIST/test_x_decoy.npy')
test_x_decoy = np.load('../../data/DecoyMNIST/test_x_decoy.npy')

print(np.array_equal(train_x_decoy_original, train_x_decoy))
print(np.array_equal(test_x_decoy_original, test_x_decoy))
