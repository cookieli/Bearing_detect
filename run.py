from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import autoencoder as ae
import classification as cf
import autoencoder_depart as aed
from input import Bearing_dataset
init = tf.global_variables_initializer()
saver = tf.train.Saver()
M_train_data = ae.mnist.train.images
train_data = Bearing_dataset.train_data
train_labels = Bearing_dataset.train_label
train_labels = tf.one_hot(train_labels, 10)
eval_labels = Bearing_dataset.eval_label
eval_labels = tf.one_hot(eval_labels, 10)
eval_data = Bearing_dataset.eval_data
#train_labels = np.asarray(train_labels, dtype =np.int32 )
with tf.Session() as sess:
    sess.run(init)
    print("Pretrain layer_1:")
    #ae.run_autoencoder(sess, ae.loss, train_data, ae.y_pred, 0.4,30000, 256, 1000, ae.optimizer, False)
    ae.run_autoencoder(sess, aed.layer_1_para['loss'], M_train_data, aed.layer_1_para['y_pred'], 1, 400, 256, 100, aed.layer_1_para['optimizer'], False)
    print("train:")
    cf.run_model(sess, cf.out, cf.loss, train_data, train_labels.eval(), 0.8, 20, 256, 100, cf.train_step, False)
    print("test:")
    cf.run_model(sess, cf.out, cf.loss, eval_data, eval_labels.eval(), 1, 1, 256)