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











    




