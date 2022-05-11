from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from scipy.io import loadmat


def SVHN(data_dir):
    """ Load SVHN data """
    train_data_dict = loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    test_data_dict = loadmat(os.path.join(data_dir, 'test_32x32.mat'))
    data_train = train_data_dict['X'].transpose(3, 1, 0, 2)
    labels_train = train_data_dict['y'].flatten()
    data_test = test_data_dict['X'].transpose(3, 1, 0, 2)
    labels_test = test_data_dict['y'].flatten()
    return data_train, labels_train, data_test, labels_test


def SVHN_data(num_training=70000, num_validation=3257):
    # Load the raw SVHN data
    data_train, labels_train, data_test, labels_test = SVHN('data')

    # convert to float and rescale
    data_train = data_train.astype(np.float32) / 255
    data_test = data_test.astype(np.float32) / 255
    # convert labels to zero-indexed
    labels_train -= 1
    labels_test -= 1

    # Subsample the data
    data_val = data_train[range(num_training, num_training+num_validation)]
    labels_val = labels_train[range(num_training, num_training+num_validation)]
    data_train = data_train[range(num_training)]
    labels_train = labels_train[range(num_training)]

    # Normalize the data: subtract the images mean
    mean_image = np.mean(data_train, axis=0)
    data_train -= mean_image
    data_val -= mean_image
    data_test -= mean_image

    # return a data dict
    return {
      'data_train': data_train, 'labels_train': labels_train,
      'data_val': data_val, 'labels_val': labels_val,
      'data_test': data_test, 'labels_test': labels_test,
    }
