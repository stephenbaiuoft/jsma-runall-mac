from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import *
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.attacks import SaliencyMapMethod
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

# quickly test data samples
# wanno make sure Training Data is larger than
# 20 * 20 ==> 400

def evaluate_pca(train_start=0, train_end=3000, test_start=0,
                   test_end=500, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   nb_filters=64):

    # to CSC2515 Files
    relative_path_2515 = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/' \
                      'bai work/CSC2515 Files/'

    # Create TF session
    sess = tf.Session()



    # >>> data = np.load('/tmp/123.npz')
    # >>> data['a']
    # load JSMA numpy array
    np_jsma_data_path = 'saver/numpy_jsma_x_data/'

    file = 'jsma_testing_x.npz'
    data = np.load(relative_path_2515 + np_jsma_data_path + file)
    X_jsma_testing  = data['testing_jsma_x']

    # Temporary solution: ===> terminal to compute PCA...
    X_jsma_testing_pca_0 = pca_filter(X_jsma_testing, 5)
    print("finished!!!! ")



# apply PCA filtering
def pca_filter(X, n_components):
    # expand out to features
    X_flat = X.reshape(-1, 28*28)

    n_components_total = int(n_components * n_components)
    pca_tmp = PCA(n_components = n_components_total)

    s_fit_transform = pca_tmp.fit_transform(X_flat)
    # inverse back to original planes
    X_inverse_flat = pca_tmp.inverse_transform(s_fit_transform)

    # back to X_filter
    X_filter = X_inverse_flat.reshape(-1, 28, 28, 1)

    return X_filter


def main(argv=None):
    # test ==> we will try various n_components
    evaluate_pca()


if __name__ == '__main__':


    tf.app.run()

