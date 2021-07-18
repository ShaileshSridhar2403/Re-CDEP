import tensorflow as tf
import numpy as np

import sys
sys.path.append('../src')
import cd


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
        # TODO: check if this is correct (since I am using scce and not nll_loss)
        return tf.nn.log_softmax(x, axis=1)
        # return x

    def logits(self, x):
        x = tf.nn.relu(self.conv1(x))
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = tf.nn.relu(self.conv2(x))
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = tf.reshape(x, [-1, 4*4*50])
        x = tf.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# make the sampling thing
blob = np.zeros((28,28))
size_blob =5
blob[:size_blob, :size_blob ] = 1

blob[-size_blob:, :size_blob] = 1
blob[:size_blob, -size_blob:] = 1
blob[-size_blob:, -size_blob:] = 1


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
    # a = tf.constant([[[[1., 2, 3, 4],
    #                 [5, 6, 7, 8],
    #                 [9, 10, 11, 12],
    #                 [13, 14, 15, 16]]]])
    # a = tf.reshape(a, [1, 4, 4, 1])

    x = np.arange(48).reshape(3, 4, 4, 1)
    a = tf.constant(x, dtype=tf.float32)

    # a = tf.transpose(a, perm=[0, 2, 3, 1])

    b, b_ind = tf.nn.max_pool_with_argmax(a, ksize=2, strides=2, padding='VALID', include_batch_in_index=True)

    # b = tf.transpose(b, perm=[0, 3, 1, 2])
    # b_ind = tf.transpose(b_ind, perm=[0, 3, 1, 2])

    c = cd.unpool(b, b_ind, output_size='NHWC')
    print(c)
    print(c.shape)

def test_propagate_dropout():
    a = tf.constant([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    b = tf.constant([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])

    c = tf.keras.layers.Dropout(rate=0.5)
    result = cd.propagate_dropout(a, b, c)

    print(result)

def test_propagate_pooling():
    # a = tf.constant([[[[4., 5, 6, 7],
    #                 [8, 9, 10, 11],
    #                 [12, 13, 14, 15],
    #                 [16, 17, 18, 19]]]])
    # b = tf.constant([[[[1., 2, 3, 4],
    #                 [5, 6, 7, 8],
    #                 [9, 10, 11, 12],
    #                 [13, 14, 15, 16]]]])

    x = np.arange(2352).reshape(3, 1, 28, 28)
    y = np.arange(2352).reshape(3, 1, 28, 28)
    a = tf.constant(x, dtype=tf.float32)
    b = tf.constant(y, dtype=tf.float32)
    # c = tf.keras.layers.MaxPool2D(pool_size=2, data_format='channels_first')
    result = cd.propagate_pooling(a, b)

    print(result)

def test_cd():
    model = Net()

    # float64 is double
    x = np.arange(2352).reshape(3, 1, 28, 28)
    # x = np.arange(784).reshape(1, 1, 28, 28)
    a = tf.constant(x, dtype=tf.float64)
    result = cd.cd_batch(blob=blob, im_torch=a, model=model)
    print(result)

def test_shape_check():
    x = np.arange(2352).reshape(3, 1, 28, 28)
    # x = np.arange(784).reshape(1, 1, 28, 28)
    a = tf.constant(x, dtype=tf.float64)
    for i in a:
        print(tf.expand_dims(i, axis=0).shape)

test_unpool()
