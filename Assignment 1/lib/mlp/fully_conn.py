from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from unicodedata import name

from numpy import outer

from lib.mlp.layer_utils import *


""" Super Class """


class Module(object):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, feat, is_training=True, seed=None):
        output = feat
        for layer in self.net.layers:
            if isinstance(layer, dropout):
                output = layer.forward(output, is_training, seed)
            else:
                output = layer.forward(output)
        self.net.gather_params()
        return output

    def backward(self, dprev):
        for layer in self.net.layers[::-1]:
            dprev = layer.backward(dprev)
        self.net.gather_grads()
        return dprev


""" Classes """


class TestFCReLU(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            flatten(name="flatten1"),
            fc(input_dim=15, output_dim=5, name="fc1"),
            leaky_relu(name="lrelu1")
            ########### END ###########
        )


class SmallFullyConnectedNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            fc(input_dim=4, output_dim=30, init_scale=0.02, name='fc1'),
            leaky_relu(name='lrelu1'),
            fc(input_dim=30, output_dim=7, init_scale=0.02,  name='fc2')
            ########### END ###########
        )


class DropoutNet(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.dropout = dropout
        self.seed = seed
        self.net = sequential(
            flatten(name="flat"),
            fc(15, 20, 5e-2, name="fc1"),
            leaky_relu(name="relu1"),
            fc(20, 30, 5e-2, name="fc2"),
            leaky_relu(name="relu2"),
            fc(30, 10, 5e-2, name="fc3"),
            leaky_relu(name="relu3"),
            dropout(keep_prob, seed=seed)
        )


class TinyNet(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.net = sequential(
            ########## TODO: ##########
            flatten(),
            fc(input_dim=3072, output_dim=500, init_scale=5e-2, name='fc1'),
            leaky_relu(),
            fc(500, 10, 5e-2, name='fc2')
            ########### END ###########
        )


class DropoutNetTest(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.dropout = dropout
        self.seed = seed
        self.net = sequential(
            flatten(name="flat"),
            fc(3072, 500, 1e-2, name="fc1"),
            dropout(keep_prob, seed=seed),
            leaky_relu(name="relu1"),
            fc(500, 500, 1e-2, name="fc2"),
            leaky_relu(name="relu2"),
            fc(500, 10, 1e-2, name="fc3"),
        )


class FullyConnectedNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.net = sequential(
            flatten(name="flat"),
            fc(3072, 100, 5e-2, name="fc1"),
            leaky_relu(name="relu1"),
            fc(100, 100, 5e-2, name="fc2"),
            leaky_relu(name="relu2"),
            fc(100, 100, 5e-2, name="fc3"),
            leaky_relu(name="relu3"),
            fc(100, 100, 5e-2, name="fc4"),
            leaky_relu(name="relu4"),
            fc(100, 100, 5e-2, name="fc5"),
            leaky_relu(name="relu5"),
            fc(100, 10, 5e-2, name="fc6")
        )
