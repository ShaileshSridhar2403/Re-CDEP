import tensorflow as tf
import numpy as np

import sys
sys.path.append('../src')
import cd


def test_propagate_conv_linear():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    # c = tf.constant([[1.0, 1.0], [0.0, 1.0]])

    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.Input(shape=(2,)))
    # model.add(tf.keras.layers.Dense(2))
    inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    layers = model.layers[1:]  # skipping the input layer
    layers[0].set_weights([np.ones((2, 2)), np.ones((2,))])
    print(layers[0].get_weights())

    r1, r2 = cd.propagate_conv_linear(a, b, layers[0])
    print(r1)
    print()
    print(r2)
    # print()
    # print(r3)

def test_unpool():
    a = tf.constant([[[[1., 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]])

    b = tf.nn.max_pool2d(a, ksize=2, strides=2, padding='VALID', data_format='NCHW')

    indices = cd.get_indices(a, b)
    print(indices)

    c = cd.unpool(b, indices)
    print(c)


    x = tf.constant([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]])
    x = tf.reshape(x, [1, 3, 3, 1])
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
       strides=(1, 1), padding='valid')
    print(max_pool_2d.strides)
    a = max_pool_2d(x)

    print(a)
    print(a.strides)


test_propagate_conv_linear()
