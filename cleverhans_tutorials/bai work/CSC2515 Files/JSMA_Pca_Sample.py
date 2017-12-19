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

FLAGS = flags.FLAGS

# quickly test data samples
# wanno make sure Training Data is larger than
# 20 * 20 ==> 400

def evaluate_pca(train_start=0, train_end=1, test_start=0,
                   test_end=1, nb_epochs=1, batch_size=128,
                   learning_rate=0.001,
                   nb_filters=64):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """
    model_save_path = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/' \
                      'bai work/CSC2515 Files/tmp/'
    # to CSC2515 Files
    relative_path_2515 = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/' \
                      'bai work/CSC2515 Files/tmp/'

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
    model_baseline = make_basic_cnn_pca(nb_filters=nb_filters)
    preds_baseline = model_baseline(x)

    jsma = SaliencyMapMethod(model_baseline, back='tf', sess=sess)
    adv_x_2 = jsma.generate(x, **jsma_params)
    # test to get np array

    preds_2_adv = model_baseline(adv_x_2)


    # pre-process input shape None, 28, 28
    # in this case, for simplicity--> we always choose
    # number of components equal to square roots
    ###########################################################
    #####BASELINE 4 models of 25, 36, 49, 100,
    ###########################################################
    n_components = 5
    X_train_pca_0 = pca_filter(X_train, n_components)
    X_test_pca_0 = pca_filter(X_test,n_components)
    model_0 = make_basic_cnn_pca(nb_filters=nb_filters)
    x_0 = tf.placeholder(tf.float32, shape=(None, n_components, n_components, 1))
    preds_0 = model_0(x_0)

    jsma_0 = SaliencyMapMethod(model_0, back='tf', sess=sess)
    adv_x_0 = jsma_0.generate(x, **jsma_params)
    # generate the preds_adv_0
    preds_adv_0 = model_0(adv_x_0)


    n_components = 6
    X_train_pca_1 = pca_filter(X_train, n_components)
    X_test_pca_1 = pca_filter(X_test,n_components)
    model_1 = make_basic_cnn_pca(nb_filters=nb_filters)
    x_1 = tf.placeholder(tf.float32, shape=(None, n_components, n_components, 1))
    preds_1 = model_1(x_1)

    jsma_1 = SaliencyMapMethod(model_0, back='tf', sess=sess)
    adv_x_1 = jsma_1.generate(x, **jsma_params)
    # generate the preds_adv_0
    preds_adv_1 = model_1(adv_x_1)


    n_components = 7
    X_train_pca_2 = pca_filter(X_train, n_components)
    X_test_pca_2 = pca_filter(X_test,n_components)
    # change # of perceptons in a model_01
    model_2 = make_basic_cnn_pca(nb_filters=nb_filters)
    x_2 = tf.placeholder(tf.float32, shape=(None, n_components, n_components, 1))
    preds_2 = model_2(x_2)

    jsma_2 = SaliencyMapMethod(model_0, back='tf', sess=sess)
    adv_x_2 = jsma_2.generate(x, **jsma_params)
    # generate the preds_adv_0
    preds_adv_2 = model_2(adv_x_2)


    n_components = 10
    X_train_pca_3 = pca_filter(X_train, n_components)
    X_test_pca_3 = pca_filter(X_test,n_components)
    # change filter size and c effect
    model_3 = make_basic_cnn_pca(nb_filters=nb_filters)
    x_3 = tf.placeholder(tf.float32, shape=(None, n_components, n_components, 1))
    preds_3 = model_3(x_3)

    jsma_3 = SaliencyMapMethod(model_0, back='tf', sess=sess)
    adv_x_3 = jsma_3.generate(x, **jsma_params)
    # generate the preds_adv_0
    preds_adv_3 = model_3(adv_x_3)



    def evaluate_baseline(eva_p0, eva_p1):
        f_out.write("Baseline model\n")
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_baseline, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on legitimate examples: '+str(accuracy) + '\n')

        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on adversarial examples: '+str(accuracy) + '\n')

        report.adv_train_adv_eval = accuracy

    model_train(sess, x, y, preds_baseline, X_train, Y_train,
                predictions_adv=preds_2_adv,
                evaluate=evaluate_baseline,
                eva_p0=None, eva_p1=None,
                args=train_params, rng=rng)

    # after training evaluating
    feed_dict = {x: X_train}
    r = sess.run(adv_x_2, feed_dict = feed_dict)

    # evaluation function for other comparisons
    def evaluate_other(model_name=None, model_pred=None):
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}

        accuracy = model_eval(sess, x_0, y, model_pred, X_test_pca_0, Y_test,
                              args=eval_params)
        print(model_name + '\nTest accuracy on legitimate examples: %0.4f' % accuracy)
        f_out.write('\n\n' + model_name + '\nTest accuracy on legitimate examples: '+str(accuracy) + '\n')


        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test_pca_0,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on adversarial examples:  ' + str(accuracy) + '\n')



    # Training Session!!!
    eva_p0 = 'model 0: PAC of Dimension 8 X 8'
    eva_p1 = preds_0
    model_train(sess, x, y, preds_0, X_train, Y_train,
                evaluate=evaluate_other, eva_p0=eva_p0, eva_p1=eva_p1,
                args=train_params)

    eva_p0 = 'model 1: make_5_cnn_large'
    eva_p1 = preds_1
    model_train(sess, x, y, preds_1, X_train, Y_train,
                evaluate=evaluate_other, eva_p0=eva_p0, eva_p1=eva_p1,
                args=train_params)


    eva_p0 = 'model 2: change # of perceptons'
    eva_p1 = preds_2
    model_train(sess, x, y, preds_2, X_train, Y_train,
                evaluate=evaluate_other, eva_p0=eva_p0, eva_p1=eva_p1,
                args=train_params)


    eva_p0 = 'model 3: change filter size'
    eva_p1 = preds_3
    model_train(sess, x, y, preds_3, X_train, Y_train,
                evaluate=evaluate_other, eva_p0=eva_p0, eva_p1=eva_p1,
                args=train_params)


    # >>> data = np.load('/tmp/123.npz')
    # >>> data['a']
    np_data_path = 'saver/PCA_numpy_data/'

    # f_out.write('\nJSMA Tensor saved in: ' + jsma_save_path + '\n')
    f_out.write('\n\nModel saved in file: ' + relative_path_2515 + np_data_path +'\n')

    # close the file
    f_out.close()


# pca filter
def pca_filter(X, n_components):
    # expand out to features
    X_flat = X.reshape(-1, 28*28)
    n_components_total = n_components * n_components
    pca = PCA(n_components= n_components_total)

    X_filter_flat = pca.fit_transform(X_flat)

    # back to X_filter
    X_filter = X_filter_flat.reshape(-1, n_components, n_components, 1)

    return X_filter


def main(argv=None):
    # test ==> we will try various n_components
    evaluate_pca()


if __name__ == '__main__':


    tf.app.run()

