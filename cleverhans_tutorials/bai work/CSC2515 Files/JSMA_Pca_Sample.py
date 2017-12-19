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

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)



    # Create TF session
    sess = tf.Session()
    print("\n\nRunning JSMA PCA Evaluation Test\n")


    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    rng = np.random.RandomState([2017, 8, 30])
    # rng_0 = np.random.RandomState([2017, 12, 12])
    # rng_1 = np.random.RandomState([2017, 11, 11])

    f_out = open(relative_path_2515 + "PCA_JSMA_evaluation_SAMPLE.log", "w")


    ##########################################################
    ###Common Training Parameters
    ##########################################################
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    ###########################################################
    #####BASELINE
    ###########################################################
    # Redefine TF model graph
    model = make_basic_cnn(nb_filters=nb_filters)
    preds = model(x)

    # pre-process input shape None, 28, 28
    # in this case, for simplicity--> we always choose
    # number of components equal to square roots
    ###########################################################
    #####BASELINE 4 models of 25, 36, 49, 100,
    ###########################################################
    np_jsma_data_path = 'saver/numpy_jsma_x_data/'

    # Training Data is Not Used
    X_training_saved = np.load(relative_path_2515 + np_jsma_data_path + 'jsma_training_x.npz')
    X_jsma_training = X_training_saved['training_jsma_x']

    X_testing_saved = np.load(relative_path_2515 + np_jsma_data_path + 'jsma_testing_x.npz')
    # bug in the previous saving script lol
    X_jsma_testing = X_testing_saved['training_jsma_x']


    # this corresponds to Y_testing

    n_components = 5

    # shape should be None x 28 x 28
    X_jsma_testing_pca_0 = pca_filter(X_test, n_components)


    # evaluation function for other comparisons
    def evaluate(model_name=None, model_pred=None):
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}

        accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                              args=eval_params)
        print(model_name + '\nTest accuracy on legitimate examples: %0.4f' % accuracy)
        f_out.write('\n\n' + model_name + '\nTest accuracy on legitimate examples: '+str(accuracy) + '\n')

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds, X_jsma_testing,
                              Y_test, args=eval_params)
        print('Test accuracy on unfiltered adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on unfiltered adversarial examples:  ' + str(accuracy) + '\n')

        # Accuracy of the adversarially filtered trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds, X_jsma_testing_pca_0 ,
                              Y_test, args=eval_params)
        print('Test accuracy on PCA filtered adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on PCA filtered adversarial examples:  ' + str(accuracy) + '\n')

    eva_p0 = "Model 0, with PCA components = 5 x 5"
    # now train the model
    model_train(sess, x, y, preds, X_train, Y_train,
                evaluate=evaluate,
                eva_p0=eva_p0, eva_p1=None,
                args=train_params, rng=rng)


    n_components = 6
    X_train_pca_1 = pca_filter(X_train, n_components)
    X_test_pca_1 = pca_filter(X_test,n_components)


    n_components = 7
    X_train_pca_2 = pca_filter(X_train, n_components)
    X_test_pca_2 = pca_filter(X_test,n_components)



    n_components = 10
    X_train_pca_3 = pca_filter(X_train, n_components)
    X_test_pca_3 = pca_filter(X_test,n_components)



    # >>> data = np.load('/tmp/123.npz')
    # >>> data['a']
    np_pca_data_path = 'saver/numpy_pca_data'

    np.savez(relative_path_2515 + np_jsma_data_path + 'X_jsma_testing_pca_0.npz',
             X_jsma_testing_pca_0=X_jsma_testing_pca_0 )

    # f_out.write('\nJSMA Tensor saved in: ' + jsma_save_path + '\n')
    f_out.write('\n\nPCA data saved in file: ' + relative_path_2515 + np_pca_data_path +'\n')

    # close the file
    f_out.close()


# apply PCA filtering
def pca_filter(X, n_components):
    # expand out to features
    X_flat = X.reshape(-1, 28*28)
    n_components_total = n_components * n_components
    pca = PCA(n_components= n_components_total)

    X_filter_flat = pca.fit_transform(X_flat)

    # inverse back to original planes
    X_inverse_flat = pca.inverse_transform(X_filter_flat)

    # back to X_filter
    X_filter = X_inverse_flat.reshape(-1, 28, 28, 1)

    return X_filter


def main(argv=None):
    # test ==> we will try various n_components
    evaluate_pca()


if __name__ == '__main__':


    tf.app.run()

