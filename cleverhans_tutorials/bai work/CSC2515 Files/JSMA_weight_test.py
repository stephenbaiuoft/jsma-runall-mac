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
    model_save_path = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/bai work/CSC2515 Files/saver/'

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
    # rng_0 = np.random.RandomState([2017, 12, 12])
    # rng_1 = np.random.RandomState([2017, 11, 11])

    f_out = open("JSMA_weight_evaluation.log", "w")

    # this is for running 5 cnn?
    model_0 = make_5_cnn_small(nb_filters=nb_filters)
    preds_0 = model_0(x)

    model_1 = make_5_cnn_large(nb_filters=nb_filters)
    preds_1 = model_1(x)

    # change # of perceptons in a model_01 ==>larger kernel
    model_2 = make_cnn_percepton(nb_filters=nb_filters)
    preds_2 = model_2(x)

    # change filter size and c effect
    model_3 = make_cnn_filter(nb_filters=20)
    preds_3 = model_3(x)

    # larger kernel size!!!
    model_4 = make_cnn_large_kernel(nb_filters=nb_filters)
    preds_4 = model_4(x)


    # Redefine TF model graph
    model_baseline = make_basic_cnn(nb_filters=nb_filters)
    preds_baseline = model_baseline(x)

    jsma = SaliencyMapMethod(model_baseline, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    adv_x_2 = jsma.generate(x, **jsma_params)
    preds_2_adv = model_baseline(adv_x_2)

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

    # Perform baseline simple training
    eva_p0 = "Baseline Evaluation"
    eva_p1 = preds_baseline
    model_train(sess, x, y, preds_baseline, X_train, Y_train,
                evaluate=evaluate_baseline,
                eva_p0=eva_p0, eva_p1=eva_p1,
                args=train_params, rng=rng)

    # after training the baseline model ==> save the adversarial examples
    feed_dict = {x: X_train}
    training_jsma_x = sess.run(adv_x_2, feed_dict = feed_dict)
    print("trained JSMA examples --> shape is:", training_jsma_x.shape)
    f_out.write("trained JSMA examples --> shape is: " + str(training_jsma_x.shape) +'\n')

    np_jsma_data_path = 'saver/numpy_jsma_x_data/'
    np.savez(relative_path_2515 + np_jsma_data_path + 'jsma_training_x.npz', training_jsma_x=training_jsma_x)

    feed_dict = {x: X_test}
    testing_jsma_x = sess.run(adv_x_2, feed_dict = feed_dict)
    print("testing JSMA examples --> shape is:", testing_jsma_x.shape)
    f_out.write("testing JSMA examples --> shape is: " + str(testing_jsma_x.shape) + '\n')
    # Bug here ===> will fix later.... --> but will not regenerate the code
    # Only affects restoring lol
    np.savez(relative_path_2515 + np_jsma_data_path + 'jsma_testing_x.npz', testing_jsma_x=testing_jsma_x)


    # evaluation function for other comparisons
    def evaluate_other(model_name=None, model_pred=None):
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, model_pred, X_test, Y_test,
                              args=eval_params)
        print(model_name + '\nTest accuracy on legitimate examples: %0.4f' % accuracy)
        f_out.write('\n\n' + model_name + '\nTest accuracy on legitimate examples: '+str(accuracy) + '\n')


        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        f_out.write('Test accuracy on adversarial examples:  ' + str(accuracy) + '\n')



    # Training Session!!!
    eva_p0 = 'model 0: make_5_cnn_small'
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

    eva_p0 = 'model 4: change kernel size to a larger kernel!!'
    eva_p1 = preds_4
    model_train(sess, x, y, preds_4, X_train, Y_train,
                evaluate=evaluate_other, eva_p0=eva_p0, eva_p1=eva_p1,
                args=train_params)

    # Add ops to save and restore all the variables.
    # use version I as V2 u may not recover!
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    # Save entire sess to disk.

    model_sess_name = 'cnn_weight_test.ckpt'
    save_path = saver.save(sess, model_save_path + model_sess_name)
    print("Model saved in file: %s" % save_path)



    # to CSC2515 Files
    relative_path_2515 = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/' \
                      'bai work/CSC2515 Files/'
    # saves the output so later no need to re-fun file
    np_data_path = 'saver/numpy_data/'

    # baseline weights
    weight_set_baseline = evaluate_model_tensor(model_baseline, sess, f_out)
    np.savez(relative_path_2515 + np_data_path + 'model_baseline_weights.npz',
             conv_weights= weight_set_baseline)

    weight_set_0 = evaluate_model_tensor(model_0, sess, f_out)
    np.savez(relative_path_2515 + np_data_path + 'model_0_weights.npz', conv_weights= weight_set_0)


    weight_set_1 = evaluate_model_tensor(model_1, sess, f_out)
    # saves the output so later no need to re-fun file
    np.savez(relative_path_2515 + np_data_path + 'model_1_weights.npz', conv_weights= weight_set_1)

    weight_set_2 = evaluate_model_tensor(model_2, sess, f_out)
    np.savez(relative_path_2515 + np_data_path + 'model_2_weights.npz', conv_weights= weight_set_2)


    weight_set_3 = evaluate_model_tensor(model_3, sess, f_out)
    np.savez(relative_path_2515 + np_data_path + 'model_3_weights.npz', conv_weights= weight_set_3)

    weight_set_4 = evaluate_model_tensor(model_4, sess, f_out)
    np.savez(relative_path_2515 + np_data_path + 'model_4_weights.npz', conv_weights= weight_set_4)

    # f_out.write('\nJSMA Tensor saved in: ' + jsma_save_path + '\n')
    f_out.write('\n\nModel saved in file: ' + save_path +'\n')




    # close the file
    f_out.close()


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
            f_out.write("shape is: " + str( np_weight.shape) + '\n')

            weight_set.append(np_weight)

    print("*****End of Model Weight********\n")
    f_out.write("*****End of Model Weight********\n")

    return weight_set


def main(argv=None):
    evaluate_weight()


if __name__ == '__main__':


    tf.app.run()

