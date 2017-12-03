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
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ElasticNetMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import CarliniWagnerL2


FLAGS = flags.FLAGS


def mnist_tutorial_jsma(train_start=0, train_end=5, test_start=0,
                        test_end=2, viz_enabled=True, nb_epochs=8,
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

    # do not log
    model_train(sess, x, y, preds, X_train, Y_train, args=train_params,verbose=False,
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


    # 1. Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    # randomly select for testing
    adv_random = jsma.generate(x, **jsma_params)
    preds_adv_random = model.get_probs(adv_random)

    # 2. Instantiate FGSM attack
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x_fgsm = fgsm.generate(x, **fgsm_params)
    preds_adv_fgsm = model.get_probs(adv_x_fgsm)


    # 3. Instantiate Elastic net attack
    en_params = {'binary_search_steps': 10,
                 # 'y': None,
                 'max_iterations': 1000,
                 'learning_rate': 0.01,
                 'batch_size': source_samples,
                 'initial_const': 10,
                 'beta': 0}
    enet = ElasticNetMethod(model, sess=sess)
    adv_x_en = enet.generate(x, **en_params)
    preds_adv_elastic_net = model.get_probs(adv_x_en)

    # 4. Deepfool
    deepfool_params = {'nb_candidate':10,
                       'overshoot':0.02,
                       'max_iter': 50,
                       'clip_min': 0.,
                       'clip_max': 1.}
    deepfool = DeepFool(model, sess=sess)
    adv_x_df = deepfool.generate(x, **deepfool_params)
    preds_adv_deepfool = model.get_probs(adv_x_df)

    # 5. Base Iterative
    bim_params = {'eps': 0.3,
                  'eps_iter': 0.01,
                  'nb_iter': 100,
                  'clip_min': 0.,
                  'clip_max': 1.}
    base_iter = BasicIterativeMethod(model, sess=sess)
    adv_x_bi = base_iter.generate(x, **bim_params)
    preds_adv_base_iter = model.get_probs(adv_x_bi)

    # 6. C & W Attack
    cw = CarliniWagnerL2(model, back='tf', sess=sess)
    cw_params = {'binary_search_steps': 1,
                 # 'y': None,
                 'max_iterations': 100,
                 'learning_rate': 0.1,
                 'batch_size': source_samples,
                 'initial_const': 10}
    adv_x_cw = cw.generate(x, **cw_params)
    preds_adv_cw = model.get_probs(adv_x_cw)


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

    # saves the output so later no need to re-fun file
    np.savez("jsma_training_data.npz", x_train=X_train_data
             , y_train=Y_train_data)

    # >>> data = np.load('/tmp/123.npz')
    # >>> data['a']

    f_out = open("jsma_elastic_against5.log", "w")

    # evaluate the function against 5 attacks
    # fgsm, base iterative, jsma, elastic net, and deepfool
    def evaluate_against_all():
            # 1 Clean Data
            eval_params = {'batch_size': batch_size}
            accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                                  args=eval_params)
            print('Legitimate accuracy: %0.4f' % accuracy)

            tmp = 'Legitimate accuracy: '+ str(accuracy + "\n")
            f_out.write(tmp)


            # 2 JSMA
            accuracy = model_eval(sess, x, y, preds_adv_random, X_test,
                                  Y_test, args=eval_params)

            print('JSMA accuracy: %0.4f' % accuracy)
            tmp = 'JSMA accuracy:'+ str(accuracy + "\n")
            f_out.write(tmp)


            # 3 FGSM
            accuracy = model_eval(sess, x, y, preds_adv_fgsm, X_test,
                                  Y_test, args=eval_params)

            print('FGSM accuracy: %0.4f' % accuracy)
            tmp = 'FGSM accuracy:' + str(accuracy + "\n")
            f_out.write(tmp)

            # 4 Base Iterative
            accuracy = model_eval(sess, x, y, preds_adv_base_iter, X_test,
                                  Y_test, args=eval_params)

            print('Base Iterative accuracy: %0.4f' % accuracy)
            tmp = 'Base Iterative accuracy:' + str(accuracy + "\n")
            f_out.write(tmp)

            # 5 Elastic Net
            accuracy = model_eval(sess, x, y, preds_adv_elastic_net, X_test,
                                  Y_test, args=eval_params)

            print('Elastic Net accuracy: %0.4f' % accuracy)
            tmp = 'Elastic Net accuracy:' + str(accuracy + "\n")
            f_out.write(tmp)

            # 6 DeepFool
            accuracy = model_eval(sess, x, y, preds_adv_deepfool, X_test,
                                  Y_test, args=eval_params)
            print('DeepFool accuracy: %0.4f' % accuracy)
            tmp = 'DeepFool accuracy:' + str(accuracy + "\n")
            f_out.write(tmp)

            # 7 C & W Attack
            accuracy = model_eval(sess, x, y, preds_adv_cw, X_test,
                                  Y_test, args=eval_params)
            print('C & W accuracy: %0.4f' % accuracy)
            tmp = 'C & W  accuracy:' + str(accuracy + "\n")
            f_out.write(tmp)

            f_out.write("*******End of Epoch***********\n\n")

        # report.adv_train_adv_eval = accuracy

    print("Now Adversarial Training with Elastic Net  + modified X_train and Y_train")
    # trained_model.out
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': '/home/stephen/PycharmProjects/jsma-runall-mac/',
        'filename': 'trained_model.out'
    }
    model_train(sess, x, y, preds, X_train_data, Y_train_data,
                save=True, predictions_adv=preds_adv_elastic_net,
                evaluate=evaluate_against_all, verbose=False,
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
    flags.DEFINE_integer('nb_epochs', 8, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    tf.app.run()
