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

# import the trained jsma trained graphs
def load_jsma_model():
    model_save_path = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/bai work/CSC2515 Files/saver/'
    model_save_dir = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/bai work/CSC2515 Files/saver'
    model_sess_meta = 'cnn_weight_test.ckpt.meta'
    model_data = 'cnn_weight_test.ckpt.data-00000-of-00001'

    model_data_t = 'cnn_weight_test.ckpt'

    restore_dir = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/' \
                  'bai work/CSC2515 Files/saver/cnn_weight_test.ckpt'

    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(model_save_path+model_sess_meta)

    with tf.Session() as sess:
        # restore to sess
        imported_meta.restore(sess, restore_dir)
        # model data failure
        #imported_meta.restore(sess, model_save_path + model_data)
        print("successfully restored/loaded back to sess")







def main(argv=None):
    load_jsma_model()


if __name__ == '__main__':


    tf.app.run()

