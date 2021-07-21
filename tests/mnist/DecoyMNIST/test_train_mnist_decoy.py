import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K


def test_scce():
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    x = np.arange(15).reshape(3, 5)
    print(x.shape)
    a = tf.constant(x, dtype=tf.float32)
    # print(tf.nn.log_softmax(a))
    target = tf.constant([1, 0, 4])
    output = float(scce(target, a))
    print(output)

test_scce()
