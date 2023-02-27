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

def residualBlock(input_image, num_filters, kernel_size, block_number, layer_number):

    x = tf.layers.conv2d(inputs =input_image, name=str(layer_number)+'conv', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    x  = tf.layers.batch_normalization(inputs = x ,axis = -1, center = True, scale = True, name = str(layer_number)+'bn')

    f_x = tf.layers.conv2d(input_image, name=str(block_number)+'block_conv1', padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = None)
    f_x = tf.layers.batch_normalization(f_x, name=str(block_number)+'block_batchnorm1')
    f_x = tf.nn.relu(f_x, name=str(block_number)+'block_relu1')

    f_x = tf.layers.conv2d(f_x, name=str(block_number)+'block_conv2', padding = 'same',filters = num_filters, kernel_size = kernel_size, activation = None)
    f_x = tf.layers.batch_normalization(f_x, name=str(block_number)+'block_batchnorm2')

    h_x = tf.math.add(x, f_x)
    h_x = tf.nn.relu(h_x, name='relu'+str(layer_number))
    return h_x


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
    
    ip = residualBlock(ip, num_filters=32, kernel_size=5, block_number=1, layer_number=2)
    ip = residualBlock(ip, num_filters=64, kernel_size=5, block_number=2, layer_number=3)

    ip = tf.layers.flatten(ip)
    ip = tf.layers.dense(ip, name='fc1', units = 100, activation = None)
    ip = tf.layers.dense(ip, name='fc2', units = 10, activation = None)
    
    prLogits = ip
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax

