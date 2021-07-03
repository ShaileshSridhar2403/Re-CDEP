import tensorflow as tf
import numpy as np


def test_tensordataset():
    a = tf.constant([[1, 2], [3, 4], [5, 6]])
    b = tf.constant([[7, 8], [9, 10], [11, 12]])
    complete_dataset = tf.data.Dataset.from_tensor_slices((a, b))
    # complete_dataset = complete_dataset.batch(3)
    print(list(complete_dataset.as_numpy_iterator()))

    num_train = int(len(complete_dataset)*.9)
    print(num_train)
    # num_test = len(complete_dataset) - num_train
    # tf.random.set_seed(0)
    complete_dataset = complete_dataset.shuffle(buffer_size=3)
    train_dataset = complete_dataset.take(num_train)
    train_dataset = train_dataset.batch(3)
    # test_dataset = tf.split(complete_dataset, [num_train, num_test])
    print(train_dataset)
    for (data, target) in train_dataset:
        print(data)
        print(target)
    # print(test_dataset)

test_tensordataset()
