from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans_tutorials.tutorial_models import make_basic_cnn

FLAGS = flags.FLAGS


def mnist_tutorial_jsma(train_start=0, train_end=100, test_start=0,
                        test_end=10, viz_enabled=True, nb_epochs=2,
                        batch_size=128, nb_classes=10, source_samples=10,
                        learning_rate=0.001):
    """
    MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2017, 8, 30])

    print("x_train shape: ", X_train.shape)
    print("y_train shape: ", Y_train.shape)

    model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                rng=rng)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')


    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    # randomly select for testing
    adv_random = jsma.generate(x, **jsma_params)
    preds_adv_random = model.get_probs(adv_random)

    # ==> generate 10 targeted classes for every train data regardless
    # This call runs the Jacobian-based saliency map approach
    # Loop over the samples we want to perturb into adversarial examples

    X_train_adv_set = []
    Y_train_adv_set = []
    for index in range(X_train.shape[0]):
        print('--------------------------------------')
        x_val = X_train[index:(index+1)]
        y_val = Y_train[index]


        # add normal sample in!!!!
        X_train_adv_set.append(x_val)
        Y_train_adv_set.append(y_val)

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(y_val))
        target_classes = other_classes(nb_classes, current_class)
        # Loop over all target classes
        for target in target_classes:
            # print('Generating adv. example for target class %i' % target)
            # This call runs the Jacobian-based saliency map approach

            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            adv_x = jsma.generate_np(x_val, **jsma_params)

            # append to X_train_adv_set and Y_train_adv_set
            X_train_adv_set.append(adv_x)
            Y_train_adv_set.append(y_val)

            # shape is: (1, 28, 28, 1)
            # print("adv_x shape is: ", adv_x.shape)

            # check for success rate
            # res = int(model_argmax(sess, x, preds, adv_x) == target)

    print('-------------Finished Generating Np Adversarial Data-------------------------')

    X_train_data = np.concatenate(X_train_adv_set, axis=0)
    Y_train_data = np.stack(Y_train_adv_set, axis=0)
    print("X_train_data shape is: ", X_train_data.shape)
    print("Y_train_data shape is: ", Y_train_data.shape)


    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
            eval_params = {'batch_size': batch_size}

            print("\nStart model_val Clean Data ")
            accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                                  args=eval_params)
            print('Legitimate accuracy: %0.4f' % accuracy)
            print("\nEnd of model_eval Clean Data \n\n\n\n")
            report.adv_train_clean_eval = accuracy

            # Accuracy of the adversarially trained model on adversarial examples
            accuracy = model_eval(sess, x, y, preds_adv_random, X_test,
                                  Y_test, args=eval_params)

            print('Adversarial accuracy: %0.4f' % accuracy)
        # report.adv_train_adv_eval = accuracy

    print("About to train the model: modified X_train and Y_train")
    model_train(sess, x, y, preds, X_train_data, Y_train_data, evaluate=evaluate_2,
                args=train_params, rng=rng)


    # Close TF session
    sess.close()


    return report


def main(argv=None):
    mnist_tutorial_jsma(viz_enabled=FLAGS.viz_enabled,
                        nb_epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        nb_classes=FLAGS.nb_classes,
                        source_samples=FLAGS.source_samples,
                        learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 3, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    tf.app.run()
