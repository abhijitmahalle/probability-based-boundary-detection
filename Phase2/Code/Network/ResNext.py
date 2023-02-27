"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True


def path(x, num_filters1, num_filters2, kernal_size1, kernal_size2, b, n,c):
    x = tf.layers.conv2d(x, name= 'conv1_'+str(b)+str(n)+str(c), padding='same',filters = num_filters1, kernel_size = kernal_size1, activation = None)
    x = tf.layers.batch_normalization(x,axis = -1, center = True, scale = True, name = 'bn1_'+str(b)+str(n)+str(c))
    x = tf.nn.relu(x, name = 'reu1'+str(b)+str(n)+str(c))

    x = tf.layers.conv2d(x, name= 'conv2_'+str(b)+str(n)+str(c), padding='same',filters = num_filters2, kernel_size = kernal_size2, activation = None)
    x = tf.layers.batch_normalization(x,axis = -1, center = True, scale = True, name ='bn2_'+str(b)+str(n)+str(c))
    return x

def residualBlock(input_image, num_filters, num_filters1, num_filters2, kernal_size, kernal_size1, kernal_size2, cardinality, block_number, layer_number):

    x = tf.layers.conv2d(inputs =input_image, name='conv_'+str(block_number)+str(layer_number), padding='same',filters = num_filters, kernel_size = kernal_size, activation = None)
    x  = tf.layers.batch_normalization(inputs = x ,axis = -1, center = True, scale = True, name = 'bn_'+str(block_number)+str(layer_number))
    x_store = x

    x_merged = x_store

    for c in range(cardinality):
        x_path = path(x, num_filters1, num_filters2, kernal_size1, kernal_size2, block_number, layer_number, c)
        x_merged = tf.math.add(x_merged, x_path)

    x = tf.nn.relu(x_merged, name='relu_'+str(block_number)+str(layer_number))
    return x


def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    ip = Img
    
    ip = residualBlock(input_image = ip, num_filters = 16, num_filters1 = 16, num_filters2 = 16, kernal_size=5, kernal_size1=5, kernal_size2=5, cardinality=6, block_number=1, layer_number=1)
    ip = residualBlock(input_image = ip, num_filters = 16, num_filters1 = 16, num_filters2 = 16, kernal_size=5, kernal_size1=5, kernal_size2=5, cardinality=6, block_number=2, layer_number=2)

    ip = tf.layers.flatten(ip)
    ip = tf.layers.dense(ip, name='fc1', units = 100, activation = None)
    ip = tf.layers.dense(ip, name='fc2', units = 10, activation = None)
    
    prLogits = ip
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax




