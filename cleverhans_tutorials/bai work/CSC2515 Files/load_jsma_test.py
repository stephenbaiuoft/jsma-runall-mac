# this file is for testing model weights
"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
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

FLAGS = flags.FLAGS

# This is for loading JSMA model
def evaluate_weight(train_start=0, train_end=2000, test_start=0,
                   test_end=300, nb_epochs=3, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
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

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    model_save_path = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/bai work/saver/'
    model_sess_name = 'cnn_percepton_num.ckpt'

    # Create TF session
    sess = tf.Session()




    print("\n\nRunning JSMA Evaluation Test V2\n")
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

    f_out = open("JSMA_weight_changedModel.log", "w")

    # this is for running 5 cnn?
    model_0 = make_basic_cnn(nb_filters=nb_filters)
    preds_0 = model_0(x)

    model_1 = make_5_cnn(nb_filters=nb_filters)
    preds_1 = model_1(x)

    # Redefine TF model graph
    model_2 = make_basic_cnn(nb_filters=nb_filters)
    preds_2 = model_2(x)

    jsma = SaliencyMapMethod(model_2, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    adv_x_2 = jsma.generate(x, **jsma_params)
    preds_2_adv = model_2(adv_x_2)

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on legitimate examples: '+str(accuracy) + '\n')



        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on adversarial examples: '+str(accuracy) + '\n')


    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, rng=rng)

    def evaluate_0():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_0, X_test, Y_test,
                              args=eval_params)
        print('Model 0\nTest accuracy on legitimate examples: %0.4f' % accuracy)
        f_out.write('Model 0\nTest accuracy on legitimate examples: '+str(accuracy) + '\n')


        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on adversarial examples:  ' + str(accuracy) + '\n')


    def evaluate_1():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_1, X_test, Y_test,
                              args=eval_params)
        print('Model 1\nTest accuracy on legitimate examples: %0.4f' % accuracy)
        f_out.write('Model 1\nTest accuracy on legitimate examples:  ' + str(accuracy) + '\n')

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on adversarial examples: ' + str(accuracy) + '\n')


    # training model_0 and model_1
    model_train(sess, x, y, preds_0, X_train, Y_train, evaluate=evaluate_0,
                args=train_params)

    model_train(sess, x, y, preds_1, X_train, Y_train, evaluate=evaluate_1,
                args=train_params)



    # Save entire sess to disk.
    save_path = saver.save(sess, model_save_path + model_sess_name)
    print("Model saved in file: %s" % save_path)
    f_out.write('\n\nModel saved in file: ' + save_path +'\n')
    # close the file
    f_out.close()


def main(argv=None):
    evaluate_weight()


if __name__ == '__main__':


    tf.app.run()

