import tensorflow as tf
import numpy as np

import sys
sys.path.append('../../src')
import cd


def test_cd_text():
    # model = build_model()

    model = LSTMSentiment()
    model.build((None, None))
    # model.compile(optimizer='adam', loss='mean_squared_error')
    input_text = np.random.randint(1000, size=(32, 18))
    start = np.random.randint(18, size=(32,))
    stop = start + 1

    scores, irrel_scores = cd.cd_text_irreg_scores(input_text, model, start, stop)

    s = scores + irrel_scores
    s = tf.transpose(s)
    lstm_out = model(input_text) + model.layers[2].get_weights()[1]
    print(tf.reduce_sum(tf.abs(s-lstm_out)))
