from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow import sigmoid, tanh
from tensorflow.keras import layers

from utils import check_and_convert_to_NHWC

STABILIZING_CONSTANT = 10e-20


def unpool(pooled, ind, ksize=(2, 2), strides=(2, 2), padding=(0, 0), output_size=None):
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
    :param tuple ksize: Kernel size, should be the same as what was used
        during pooling.
    :param tuple strides: Stride of the max pooling window.
    :param tuple padding: Padding that was added to the input.
    :return: The tensor after unpooling.
    :rtype: tensorflow.Tensor
    """
    # Get the the shape of the tensor in the form of a list
    input_shape = pooled.get_shape().as_list()

    # Determine the output shape
    if output_size is None:
        n = input_shape[0]
        c = input_shape[3]
        h_out = (input_shape[1] - 1) * strides[0] - 2 * padding[0] + ksize[0]
        w_out = (input_shape[2] - 1) * strides[1] - 2 * padding[1] + ksize[1]
        output_shape = (n, h_out, w_out, c)
    else:
        output_shape = output_size

    # Reshape into one giant tensor for better workability
    pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])

    # The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes
    # flattened index ((b * height + y) * width + x) * channels + c
    # Create a single unit extended cuboid of length bath_size populating it with continuous natural
    # number from zero to batch_size
    # batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
    # b = tf.ones_like(ind) * batch_range
    # b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    # ind_ = tf.concat([b_, ind_], 1)
    ref = tf.Variable(tf.zeros([output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]]))

    # Update the sparse matrix with the pooled values , it is a batch wise operation
    unpooled_ = tf.tensor_scatter_nd_update(ref, ind_, pooled_)

    # Reshape the vector to get the final result
    unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
    # import pdb
    # pdb.set_trace()
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


def propagate_relu_vgg(relevant, irrelevant):
    # rel_score = activation(relevant)
    # irrel_score = activation(relevant + irrelevant) - activation(relevant)
    rel_score= tf.nn.relu(relevant)
    irrel_score = tf.nn.relu(relevant + irrelevant) - tf.nn.relu(relevant)
    return rel_score, irrel_score


# propagate maxpooling operation
def propagate_pooling(relevant, irrelevant, model_type='mnist', pooler=None):
    window_size = 4

    # get both indices
    temp = relevant + irrelevant
    if model_type == 'mnist':
        temp = tf.transpose(temp, perm=[0, 2, 3, 1])
    both, both_ind = tf.nn.max_pool_with_argmax(temp, ksize=2, strides=2, padding='VALID', include_batch_in_index=True)
    ones_out = tf.ones_like(both)
    size1 = relevant.shape

    if model_type == 'mnist':
        mask_both = unpool(ones_out, both_ind, ksize=(2, 2), strides=(2, 2))
        mask_both = tf.transpose(mask_both, perm=[0, 3, 1, 2])
        # relevant
        rel = mask_both * relevant
        rel = tf.nn.avg_pool2d(rel, ksize=2, strides=2, padding='VALID', data_format='NCHW') * window_size
        # irrelevant
        irrel = mask_both * irrelevant
        irrel = tf.nn.avg_pool2d(irrel, ksize=2, strides=2, padding='VALID', data_format='NCHW') * window_size
        return rel, irrel

    elif model_type == 'vgg':
        mask_both = unpool(ones_out, both_ind, ksize=pooler.get_config()['pool_size'], strides=pooler.get_config()['strides'], output_size=size1)
        # relevant
        rel = mask_both * relevant
        rel = tf.nn.avg_pool2d(rel, ksize=pooler.get_config()['pool_size'], strides=pooler.get_config()['strides'], padding='VALID', data_format='NHWC') * window_size
        # irrelevant
        irrel = mask_both * irrelevant
        irrel = tf.nn.avg_pool2d(irrel, ksize=pooler.get_config()['pool_size'], strides=pooler.get_config()['strides'], padding='VALID', data_format='NHWC') * window_size
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


def cd_inefficient(blob, im_torch, model, model_type='mnist'):
    relevant_batch = []
    irrelevant_batch = []

    for i in im_torch:
        rel, irrel = cd(blob, tf.expand_dims(i, axis=0), model, model_type='mnist')
        relevant_batch.append(rel)
        irrelevant_batch.append(irrel)

    relevant_batch = tf.concat(relevant_batch, axis=0)
    irrelevant_batch = tf.concat(irrelevant_batch, axis=0)
    return (relevant_batch, irrelevant_batch)


def cd_vgg_features(blob, img, model, model_type='vgg'):
    relevant = tf.where(blob, img, tf.zeros(img.shape))
    irrelevant = tf.where(1-blob, img, tf.zeros(img.shape))

    relevant = check_and_convert_to_NHWC(relevant)
    irrelevant = check_and_convert_to_NHWC(irrelevant)

    '''
    input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    '''
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant,model.get_layer('block1_conv1'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block1_conv2'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    # pdb.set_trace()
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, model_type=model_type,pooler= model.get_layer('block1_pool'))

    '''
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0  
    '''
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant,model.get_layer('block2_conv1'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block2_conv2'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, pooler=model.get_layer('block2_pool'), model_type=model_type)
    '''
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    '''
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block3_conv1'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block3_conv2'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block3_conv3'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, pooler=model.get_layer('block3_pool'), model_type=model_type)

    '''
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0    
    '''
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block4_conv1'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block4_conv2'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block4_conv3'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, pooler=model.get_layer('block4_pool'), model_type=model_type)

    '''
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0       
    '''
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block5_conv1'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block5_conv2'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block5_conv3'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, pooler=model.get_layer('block5_pool'), model_type=model_type)

    # relevant, irrelevant = propagate_AdaptiveAvgPool2d(relevant, irrelevant, mods[31]) #CHECKTHIS

    # relevant = relevant.reshape(relevant.size(0), -1)
    # irrelevant = irrelevant.reshape(irrelevant.size(0), -1)
    relevant = tf.reshape(relevant, (relevant.shape[0],-1))
    irrelevant = tf.reshape(irrelevant, (irrelevant.shape[0],-1))

    # exit(0);

    return relevant, irrelevant


def cd_vgg_classifier(relevant, irrelevant, im_torch, model, model_type='vgg'): #CHECK WITH AZ,M
    # set up model
    # model.eval()
    '''
    fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312
    _________________________________________________________________
    predictions (Dense)          (None, 1000)              4097000
    '''
    # NOTE: SKIPPING DROPOUT AS KERAS_APPLICATIONS MODEL DOES NOT HAVE DROPOUT LAYERS
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('fc1') )
    # print(relevant.shape)
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    # relevant, irrelevant = propagate_dropout(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('fc2'))
    relevant, irrelevant = propagate_relu_vgg(relevant, irrelevant)
    # relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[5])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant,model.get_layer('predictions'))
    # only interested in not cancer, which is class 0
    # model.train()

    return relevant, irrelevant


def cd_text_irreg_scores(batch_text, model, start, stop):
    weights = model.layers[1].get_weights()
    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = tf.split(tf.transpose(weights[0]), 4, 0)
    W_hi, W_hf, W_hg, W_ho = tf.split(tf.transpose(weights[1]), 4, 0)
    b_i, b_f, b_g, b_o = tf.split(weights[2], 4, 0)
    word_vecs = tf.transpose(model.layers[0](batch_text), perm=[1, 2, 0])  # change: we take all check this BxTxEd -> TxEdxB
    T = word_vecs.shape[0]
    batch_size = word_vecs.shape[2]
    hidden_dim = W_hi.shape[0]
    relevant_h = tf.zeros(( hidden_dim,batch_size))
    irrelevant_h = tf.zeros(( hidden_dim,batch_size))
    prev_rel = tf.zeros((  hidden_dim,batch_size))
    prev_irrel = tf.zeros((  hidden_dim,batch_size))

    for i in range(T):
        prev_rel_h = relevant_h
        prev_irrel_h = irrelevant_h
        rel_i = tf.matmul(W_hi, prev_rel_h)
        rel_g = tf.matmul(W_hg, prev_rel_h)
        rel_f = tf.matmul(W_hf, prev_rel_h)
        rel_o = tf.matmul(W_ho, prev_rel_h)
        irrel_i = tf.matmul(W_hi, prev_irrel_h)
        irrel_g = tf.matmul(W_hg, prev_irrel_h)
        irrel_f = tf.matmul(W_hf, prev_irrel_h)
        irrel_o = tf.matmul(W_ho, prev_irrel_h)

        w_ii_contrib = tf.matmul(W_ii, word_vecs[i])
        w_ig_contrib = tf.matmul(W_ig, word_vecs[i])
        w_if_contrib = tf.matmul(W_if, word_vecs[i])
        w_io_contrib = tf.matmul(W_io, word_vecs[i])

        # pdb.set_trace()
        is_in_relevant = tf.cast((start <= i), dtype=tf.float32) * tf.cast((i <= stop), dtype=tf.float32)
        is_not_in_relevant = 1 - is_in_relevant

        rel_i = rel_i + is_in_relevant * w_ii_contrib
        rel_g = rel_g + is_in_relevant * w_ig_contrib
        rel_f = rel_f + is_in_relevant * w_if_contrib
        rel_o = rel_o + is_in_relevant * w_io_contrib

        irrel_i = irrel_i + is_not_in_relevant * w_ii_contrib
        irrel_g = irrel_g + is_not_in_relevant * w_ig_contrib
        irrel_f = irrel_f + is_not_in_relevant * w_if_contrib
        irrel_o = irrel_o + is_not_in_relevant * w_io_contrib

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i[:,None], sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g[:,None], tanh)

        relevant = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g
        bias_contrib = bias_contrib_i * bias_contrib_g

        is_in_relevant_bias = tf.cast((start <= i), dtype=tf.float32) * tf.cast((i < stop), dtype=tf.float32)
        is_not_in_relevant_bias = 1- is_in_relevant_bias
        relevant = relevant + is_in_relevant_bias*bias_contrib

        irrelevant = irrelevant + is_not_in_relevant_bias*bias_contrib

        if i > 0:
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f[:,None], sigmoid)
            relevant = relevant +(rel_contrib_f + bias_contrib_f) * prev_rel
            irrelevant = irrelevant+(rel_contrib_f + irrel_contrib_f + bias_contrib_f) * prev_irrel + irrel_contrib_f *  prev_rel

        o = sigmoid(tf.matmul(W_io, word_vecs[i]) + tf.matmul(W_ho, prev_rel_h + prev_irrel_h) + b_o[:,None])

        new_rel_h, new_irrel_h = propagate_tanh_two(relevant, irrelevant)

        relevant_h = o * new_rel_h
        irrelevant_h = o * new_irrel_h
        prev_rel = relevant
        prev_irrel = irrelevant

    W_out = model.layers[2].get_weights()[0]
    W_out = tf.transpose(W_out)
    # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias

    scores = tf.matmul(W_out, relevant_h)
    irrel_scores = tf.matmul(W_out, irrelevant_h)

    return scores, irrel_scores


def softmax_out(output):
    return tf.nn.softmax(tf.stack((output[0],output[1]), axis=1), axis = 1)


def cd_penalty_for_one_decoy_all(batch_text, batch_label, model1, start, stop):
    mask_exists = (start!=-1)
    # pdb.set_trace()
    batch_label_alt = list(batch_label.numpy())
    if tf.experimental.numpy.any(mask_exists):
        model1_output = cd_text_irreg_scores(batch_text, model1, start, stop)
        # pdb.set_trace()
        correct_idx = list(zip(batch_label_alt, list(range(batch_label.shape[0]))))  # only use the correct class
        # wrong_idx = (1-batch_label, tf.range(batch_label.shape[0]))
        output = (tf.gather_nd(model1_output[0], correct_idx), tf.gather_nd(model1_output[1], correct_idx))
        # model1_softmax = softmax_out((model1_output[0][correct_idx],model1_output[1][correct_idx])) #+ softmax_out((model1_output[0][wrong_idx],model1_output[1][wrong_idx]))
        model1_softmax = softmax_out(output)
        # output = (tf.log(model1_softmax[:,1])).masked_select(mask_exists)
        # pdb.set_trace()
        output = tf.boolean_mask(tf.math.log(model1_softmax[:,1]), mask_exists)
        # return -output.mean()
        # pdb.set_trace()
        return -tf.reduce_mean(output)
    else:
        return tf.zeros(1)
