from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow import tanh
from tensorflow.keras import layers

STABILIZING_CONSTANT = 10e-20


def unpool(pooled, ind, output_size='NCHW', ksize=(2, 2), strides=(2, 2), padding=(0, 0)):
    """
    Unpools the tensor after max pooling.
    Some points to keep in mind:
        1. In tensorflow the indices in argmax are flattened, so that a
           maximum value at position [b, y, x, c] becomes flattened index
           ((b * height + y) * width + x) * channels + c.
        2. Due to point 1, use broadcasting to appropriately place the
           values at their right locations!

    :param tensorflow.Tensor pooled: Max pooled output tensor.
    :param tensorflow.Tensor ind: Argmax indices.
    :param str/tuple output_size: Output size.
    :param tuple ksize: Kernel size, should be the same as what was used
        during pooling.
    :param tuple strides: Stride of the max pooling window.
    :param tuple padding: Padding that was added to the input.
    :return: The tensor after unpooling.
    :rtype: tensorflow.Tensor
    """
    # Get the the shape of the tensor in the form of a list
    input_shape = pooled.get_shape().as_list()
    n = input_shape[0]
    c = input_shape[1]
    h_out = (input_shape[2] - 1) * strides[0] - 2 * padding[0] + ksize[0]
    w_out = (input_shape[3] - 1) * strides[1] - 2 * padding[1] + ksize[1]

    # Determine the output shape
    if output_size == 'NCHW':
        output_shape = (n, c, h_out, w_out)
    elif output_size == 'NHWC':
        output_shape = (n, h_out, w_out, c)
    else:
        output_shape = output_size

    # Reshape into one giant tensor for better workability
    pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])

    # The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes
    # flattened index ((b * height + y) * width + x) * channels + c
    # Create a single unit extended cuboid of length bath_size populating it with continuous natural
    # number from zero to batch_size
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.concat([b_, ind_],1)
    ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))

    # Update the sparse matrix with the pooled values , it is a batch wise operation
    unpooled_ = tf.tensor_scatter_nd_update(ref, ind_, pooled_)

    # Reshape the vector to get the final result
    unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
    return unpooled


def propagate_three(a, b, c, activation):
    a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)


# propagate tanh nonlinearity
def propagate_tanh_two(a, b):
    return 0.5 * (tanh(a) + (tanh(a + b) - tanh(b))), 0.5 * (tanh(b) + (tanh(a + b) - tanh(a)))


# propagate convolutional or linear layer
def propagate_conv_linear(relevant, irrelevant, module):
    bias = module(tf.zeros(irrelevant.shape))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = tf.abs(rel)
    prop_irrel = tf.abs(irrel)
    prop_sum = prop_rel + prop_irrel + STABILIZING_CONSTANT

    prop_rel = tf.divide(prop_rel, prop_sum)
    prop_irrel = tf.divide(prop_irrel, prop_sum)
    return rel + tf.multiply(prop_rel, bias), irrel + tf.multiply(prop_irrel, bias)


def propagate_AdaptiveAvgPool2d(relevant, irrelevant, module):
    rel = module(relevant)
    irrel = module(irrelevant)
    return rel, irrel


# propagate ReLu nonlinearity
def propagate_relu(relevant, irrelevant, activation):
    # swap_inplace = False
    # try:  # handles inplace
    #     if activation.inplace:
    #         swap_inplace = True
    #         activation.inplace = False
    # except:
    #     pass
    # zeros = tf.zeros(relevant.size())
    rel_score = activation(relevant)
    irrel_score = activation(relevant + irrelevant) - activation(relevant)
    # if swap_inplace:
    #     activation.inplace = True
    return rel_score, irrel_score


# propagate maxpooling operation
def propagate_pooling(relevant, irrelevant, model_type='mnist', pooler=None):
    window_size = 4

    # get both indices
    temp = relevant + irrelevant
    temp = tf.transpose(temp, perm=[0, 2, 3, 1])
    both, both_ind = tf.nn.max_pool_with_argmax(temp, ksize=2, strides=2, padding='VALID', include_batch_in_index=True)
    both = tf.transpose(both, perm=[0, 3, 1, 2])
    both_ind = tf.transpose(both_ind, perm=[0, 3, 1, 2])
    ones_out = tf.ones_like(both)
    size1 = relevant.shape

    if model_type == 'mnist':
        mask_both = unpool(ones_out, both_ind, output_size=size1, ksize=(2, 2), strides=(2, 2))
        # relevant
        rel = mask_both * relevant
        rel = tf.nn.avg_pool2d(rel, ksize=2, strides=2, padding='VALID', data_format='NCHW') * window_size
        # irrelevant
        irrel = mask_both * irrelevant
        irrel = tf.nn.avg_pool2d(irrel, ksize=2, strides=2, padding='VALID', data_format='NCHW') * window_size
        return rel, irrel

    elif model_type == 'vgg':
        mask_both = unpool(ones_out, both_ind, output_size=size1, ksize=pooler.get_config().pool_size, strides=pooler.get_config().strides)
        # relevant
        rel = mask_both * relevant
        rel = tf.nn.avg_pool2d(rel, ksize=pooler.get_config().pool_size, strides=pooler.get_config().strides, padding='VALID', data_format='NCHW') * window_size
        # irrelevant
        irrel = mask_both * irrelevant
        irrel = tf.nn.avg_pool2d(irrel, ksize=pooler.get_config().pool_size, strides=pooler.get_config().strides, padding='VALID', data_format='NCHW') * window_size
        return rel, irrel


# propagate dropout operation
def propagate_dropout(relevant, irrelevant, dropout):
    return dropout(relevant, training=True), dropout(irrelevant, training=True)


# get contextual decomposition scores for blob
def cd(blob, im_torch, model, model_type='mnist'):
    # set up model
    # model.eval()

    # set up blobs
    blob = tf.constant(blob)
    relevant = blob * im_torch
    irrelevant = (1 - blob) * im_torch

    if model_type == 'mnist':
        # scores = []
        mods = list(model.submodules)
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
        relevant, irrelevant = propagate_pooling(relevant, irrelevant, model_type='mnist')
        relevant, irrelevant = propagate_relu(relevant, irrelevant, layers.ReLU())

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[1])
        relevant, irrelevant = propagate_pooling(relevant, irrelevant, model_type='mnist')
        relevant, irrelevant = propagate_relu(relevant, irrelevant, layers.ReLU())

        relevant = tf.reshape(relevant, [-1, 800])
        irrelevant = tf.reshape(irrelevant, [-1, 800])

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[2])

        relevant, irrelevant = propagate_relu(relevant, irrelevant, layers.ReLU())
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[3])

    # else:
    #     mods = list(model.submodules)
    #     for _, mod in enumerate(mods):
    #         t = str(type(mod))
    #         if 'Conv2d' in t or 'Linear' in t:
    #             if 'Linear' in t:
    #                 relevant = relevant.view(relevant.size(0), -1)
    #                 irrelevant = irrelevant.view(irrelevant.size(0), -1)
    #             relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
    #         elif 'ReLU' in t:
    #             relevant, irrelevant = propagate_relu(relevant, irrelevant, mod)
    #         elif 'MaxPool2d' in t:
    #             relevant, irrelevant = propagate_pooling(relevant, irrelevant, model_type=model_type)
    #         elif 'Dropout' in t:
    #             relevant, irrelevant = propagate_dropout(relevant, irrelevant, mod)

    return relevant, irrelevant
