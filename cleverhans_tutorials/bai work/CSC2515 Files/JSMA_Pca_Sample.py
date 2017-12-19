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

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)





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




    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    adv_x_2 = jsma.generate(x, **jsma_params)
    preds_2_adv = model(adv_x_2)

    # evaluation function for other comparisons
    def evaluate(model_name=None, model_pred=None):
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                              args=eval_params)
        print(model_name + '\nTest accuracy on legitimate examples: %0.4f' % accuracy)
        f_out.write('\n\n' + model_name + '\nTest accuracy on legitimate examples: '+str(accuracy) + '\n')

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_jsma_testing,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on adversarial examples:  ' + str(accuracy) + '\n')


    eva_p0 = "Model 0, with PCA components = 5 x 5"
    # now train the model
    model_train(sess, x, y, preds, X_train, Y_train,
                evaluate=None,
                eva_p0=eva_p0, eva_p1=None,
                args=train_params, rng=rng)

    # saving model testing_jsma_samples
    np_jsma_data_path = 'saver/numpy_jsma_x_data/'

    feed_dict = {x: X_test}
    X_jsma_testing = sess.run(adv_x_2, feed_dict = feed_dict)
    print("X_jsma_testing shape is:\n", X_jsma_testing.shape)

    # For Saving ==> Not Required in this case
    # Later for drawing purposes
    print("testing JSMA examples --> shape is:", X_jsma_testing.shape)
    f_out.write("testing JSMA examples --> shape is: " + str(X_jsma_testing.shape) + '\n')
    np.savez(relative_path_2515 + np_jsma_data_path + 'jsma_testing_x.npz', testing_jsma_x=X_jsma_testing)

    # file = 'jsma_testing_x.npz'
    # data = np.load(relative_path_2515 + np_jsma_data_path + file)
    # X_jsma_testing = data['testing_jsma_x']

    # Temporary solution: ===> terminal to compute PCA...
    X_jsma_testing_pca_0 = pca_filter(X_jsma_testing, 5)
    X_jsma_testing_pca_1 = pca_filter(X_jsma_testing, 7)
    X_jsma_testing_pca_2 = pca_filter(X_jsma_testing, 9)
    X_jsma_testing_pca_3 = pca_filter(X_jsma_testing, 10)

    X_test_pca_0 = pca_filter(X_test, 5)
    X_test_pca_1 = pca_filter(X_test, 7)
    X_test_pca_2 = pca_filter(X_test, 9)
    X_test_pca_3 = pca_filter(X_test, 10)

    eval_params = {'batch_size': batch_size}

    accuracy = model_eval(sess, x, y, preds, X_test,
                          Y_test, args=eval_params)
    print('[Unfiltered + CLEAN] Test accuracy on CLEAN examples: %0.4f' % accuracy)
    f_out.write('[Unfiltered + CLEAN] Test accuracy on CLEAN examples:  ' + str(accuracy) + '\n')

    accuracy = model_eval(sess, x, y, preds, X_jsma_testing,
                          Y_test, args=eval_params)
    print('[Unfiltered + Adversarial] Test accuracy on adversarial examples: %0.4f' % accuracy)
    f_out.write('[Unfiltered + Adversarial] Test accuracy on adversarial examples:  ' + str(accuracy) + '\n\n\n')

    ########################################    ########################################
    accuracy = model_eval(sess, x, y, preds, X_test_pca_0,
                          Y_test, args=eval_params)
    print('[Model_0 NP + Filtered + CLEAN] Test accuracy on CLEAN examples: %0.4f' % accuracy)
    f_out.write('[Model_0 NP + Filtered + CLEAN] Test accuracy on CLEAN examples:  ' + str(accuracy) + '\n')
    accuracy = model_eval(sess, x, y, preds, X_jsma_testing_pca_0,
                          Y_test, args=eval_params)
    print('[Model_0 NP + Filtered + Adversarial] Test accuracy on adversarial examples: %0.4f' % accuracy)
    f_out.write('[Model_0 NP + Filtered + Adversarial] Test accuracy on adversarial examples:  ' + str(accuracy) + '\n\n\n')

    ########################################    ########################################
    accuracy = model_eval(sess, x, y, preds, X_test_pca_1,
                          Y_test, args=eval_params)
    print('[Model_1 NP + Filtered + CLEAN] Test accuracy on CLEAN examples: %0.4f' % accuracy)
    f_out.write('[Model_1 NP + Filtered + CLEAN] Test accuracy on CLEAN examples:  ' + str(accuracy) + '\n')
    accuracy = model_eval(sess, x, y, preds, X_jsma_testing_pca_1,
                          Y_test, args=eval_params)
    print('[Model_1 NP + Filtered + Adversarial] Test accuracy on adversarial examples: %0.4f' % accuracy)
    f_out.write('[Model_1 NP + Filtered + Adversarial] Test accuracy on adversarial examples:  ' + str(accuracy) + '\n\n\n')

    ########################################    ########################################
    accuracy = model_eval(sess, x, y, preds, X_test_pca_2,
                          Y_test, args=eval_params)
    print('[Model_2 NP + Filtered + CLEAN] Test accuracy on CLEAN examples: %0.4f' % accuracy)
    f_out.write('[Model_2 NP + Filtered + CLEAN] Test accuracy on CLEAN examples:  ' + str(accuracy) + '\n')
    accuracy = model_eval(sess, x, y, preds, X_jsma_testing_pca_2,
                          Y_test, args=eval_params)
    print('[Model_2 NP + Filtered + Adversarial] Test accuracy on adversarial examples: %0.4f' % accuracy)
    f_out.write('[Model_2 NP + Filtered + Adversarial] Test accuracy on adversarial examples:  ' + str(accuracy) + '\n\n\n')

    ########################################    ########################################
    accuracy = model_eval(sess, x, y, preds, X_test_pca_3,
                          Y_test, args=eval_params)
    print('[Model_3 NP Evaluation Filtered + CLEAN] Test accuracy on CLEAN examples: %0.4f' % accuracy)
    f_out.write('[Model_3 NP Evaluation Filtered + CLEAN] Test accuracy on CLEAN examples:  ' + str(accuracy) + '\n')
    accuracy = model_eval(sess, x, y, preds, X_jsma_testing_pca_3,
                          Y_test, args=eval_params)
    print('[Model_3 NP Evaluation Filtered + Adversarial] Test accuracy on adversarial examples: %0.4f' % accuracy)
    f_out.write('[Model_3 NP Evaluation Filtered + Adversarial] Test accuracy on adversarial examples:  ' + str(accuracy) + '\n')

    # np_jsma_pca_data_path ='saver/numpy_pca_data/'
    # filename_25 = 'pca25.npz'
    # data_25 = np.load(relative_path_2515 + np_jsma_pca_data_path + filename_25)
    # X_jsma_testing_pca_0 = data_25['ift25']
    # X_jsma_testing_pca_0 = X_jsma_testing_pca_0.reshape(-1,28, 28,1)




    # close the file
    f_out.close()


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

