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


def evaluate_weight(train_start=0, train_end=3000, test_start=0,
                    test_end=500, nb_epochs=6, batch_size=128,
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

    # to CSC2515 Files
    relative_path_2515 = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/' \
                         'bai work/CSC2515 Files/'

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    sess = tf.Session()
    print("\n\nRunning JSMA Evaluation Test\n")

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

    # Redefine TF model graph
    model_baseline = make_basic_cnn(nb_filters=nb_filters)
    preds_baseline = model_baseline(x)

    jsma = SaliencyMapMethod(model_baseline, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    adv_x_2 = jsma.generate(x, **jsma_params)
    preds_2_adv = model_baseline(adv_x_2)


    # Perform baseline simple training
    eva_p0 = "Baseline Evaluation"
    eva_p1 = preds_baseline
    model_train(sess, x, y, preds_baseline, X_train, Y_train,
                evaluate=None,
                eva_p0=eva_p0, eva_p1=eva_p1,
                args=train_params, rng=rng)

    # after training the baseline model ==> save the adversarial examples

    np_jsma_data_path = 'saver/numpy_jsma_x_data/'

    feed_dict = {x: X_test}
    testing_jsma_x = sess.run(adv_x_2, feed_dict=feed_dict)
    print("testing JSMA examples --> shape is:", testing_jsma_x.shape)

    # Bug here ===> will fix later.... --> but will not regenerate the code
    # Only affects restoring lol
    np.savez(relative_path_2515 + np_jsma_data_path + 'jsma_testing_x.npz', testing_jsma_x=testing_jsma_x)



# evaluates model tensor and return the concatenated weight set
def evaluate_model_tensor(model, sess_default, f_out):
    # get model layers
    model_layers = model.layers
    weight_set = []
    for l in model_layers:
        # only Con2D im interested in learning its weight
        if type(l) is Conv2D:
            np_weight = l.kernels.eval(session=sess_default)
            # sanity check
            print("shape is: ", np_weight.shape)
            f_out.write("shape is: " + str(np_weight.shape) + '\n')

            weight_set.append(np_weight)

    print("*****End of Model Weight********\n")
    f_out.write("*****End of Model Weight********\n")

    return weight_set


def main(argv=None):
    evaluate_weight()


if __name__ == '__main__':
    tf.app.run()

