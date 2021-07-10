import numpy as np
import tensorflow as tf


STABILIZING_CONSTANT = 10e-20

def get_layer_without_activation(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        config = layer.get_config()
        config["activation"] = "linear"
        new_layer = tf.keras.layers.Conv2D.from_config(config)

    elif isinstance(layer, tf.keras.layers.Dense):
        config = layer.get_config()
        config["activation"] = "linear"
        new_layer = tf.keras.layers.Dense.from_config(config)

    return new_layer


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

def propagate_relu(relevant, irrelevant, activation):

    # rel_score = activation(relevant)
    # irrel_score = activation(relevant + irrelevant) - activation(relevant)
    rel_score= tf.nn.relu(relevant)
    irrel_score = tf.nn.relu(relevant + irrelevant) - tf.nn.relu(relevant)
    return rel_score, irrel_score


def propagate_conv_linear(relevant, irrelevant, module):

    module = get_layer_without_activation(module)
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



def cd_vgg_features(blob,img, model, model_type='vgg'):
    relevant = blob*img
    irrelevant = (1-blob)*img
    
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
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block1_conv2'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, model.get_layer('block1_pool'), model_type=model_type)

    '''
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0  
    '''
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant,model.get_layer('block2_conv1'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block2_conv2'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, model.get_layer('block2_pool'), model_type=model_type)
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
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block3_conv2'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block3_conv3'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, model.get_layer('block3_pool'), model_type=model_type)

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
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block4_conv2'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block4_conv3'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, model.get_layer('block4_pool'), model_type=model_type)

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
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block5_conv2'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('block5_conv3'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, model.get_layer('block5_pool'), model_type=model_type)

    relevant, irrelevant = propagate_AdaptiveAvgPool2d(relevant, irrelevant, mods[31]) #CHECKTHIS


    relevant = relevant.reshape(relevant.size(0), -1)
    irrelevant = irrelevant.reshape(irrelevant.size(0), -1)

    return relevant,irrelevant


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
    #NOTE: SKIPPING DROPOUT AS KERAS_APPLICATIONS MODEL DOES NOT HAVE DROPOUT LAYERS
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('fc1') )
    # print(relevant.shape)
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    # relevant, irrelevant = propagate_dropout(relevant, irrelevant)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, model.get_layer('fc2'))
    relevant, irrelevant = propagate_relu(relevant, irrelevant)
    # relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[5])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant,model.get_layer('predictions'))
    # only interested in not cancer, which is class 0
    #model.train()
    
    return relevant, irrelevant














    




