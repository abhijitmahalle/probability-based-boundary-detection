"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s): 
Sakshi Kakde
M.Eng. Robotics,University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

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
    ip = tf.layers.conv2d(ip, name='conv1', filters = 16, kernel_size = 5, activation = None)
    ip = tf.nn.relu(ip, name='relu1')


    ip = tf.layers.conv2d(ip, name='conv2', filters = 32, kernel_size = 3, activation = None)
    ip = tf.nn.relu(ip, name='relu2')

    ip = tf.layers.flatten(ip)
    ip = tf.layers.dense(ip, name='fc1', units = 100, activation = None)

    ip = tf.layers.dense(ip, name='fc2', units = 10, activation = None)

    prLogits = ip
    prSoftMax = tf.nn.softmax(logits = prLogits)  

    return prLogits, prSoftMax

