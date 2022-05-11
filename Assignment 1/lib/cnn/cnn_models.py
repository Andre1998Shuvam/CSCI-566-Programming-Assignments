from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels = 3,kernel_size = 3, number_filters = 3,name = 'conv1'),
            MaxPoolingLayer(2,2,'mp1'),
            flatten("flatten"),
            fc(27,5,0.02,'fc1')
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels = 3, kernel_size = 3, padding = 1,number_filters = 16, name = 'conv1'),
            leaky_relu(name='lrelu1'),
            ConvLayer2D(input_channels = 16, kernel_size = 3, stride=2, number_filters = 32, name = 'conv2'),
            leaky_relu(name='lrelu2'),
            MaxPoolingLayer(2,2,'mp1'),
            flatten("flatten"),
            fc(1568,100,5e-2,'fc1'),
            leaky_relu(name='lrelu3'),
            fc(100,10,5e-2,'fc4')            
            ########### END ###########
        )
