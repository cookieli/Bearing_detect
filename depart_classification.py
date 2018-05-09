import numpy as np
import tensorflow as tf
from autoencoder import num_hidden_3, num_classes
from autoencoder_depart import Encoder, scope_name

W1 = tf.get_variable('W1', shape = [num_hidden_3, num_classes], dtype = tf.float32, initializer = tf.random_normal_initializer(stddev = 0.1))
b1 = tf.get_variable('b1', shape = [num_classes], dtype = tf.float32, initializer = tf.random_normal_initializer)

#X = tf.placeholder(tf.float32, [None, num_input], "dcf")
#drop_out_prob = tf.placeholder(tf.float32)
#training_now = tf.placeholder(tf.bool)


def classification_model(x, drop_out_prob = 1.0, train = False):
    x_norm = tf.layers.batch_normalization(x, center = True, scale = True, training = train)
    layer_1 = Encoder(x_norm, scope_name[0], need_reuse = True)
    layer_1_norm = tf.layers.batch_normalization(x, center = True, scale = True, training = train)
    layer_1_drop = tf.nn.dropout(layer_1_norm, keep_prob = drop_out_prob)
    layer_2 = Encoder(layer_1_drop, scope_name[1], need_reuse = True)
    layer_2_norm = tf.layers.batch_normalization(x, center = True, scale = True, training = train)
    layer_2_drop = tf.nn.dropout(layer_2_norm, keep_prob = drop_out_prob)
    layer_3 = Encoder(layer_2_drop, scope_name[2], need_reuse = True)
    layer_3_norm = tf.layers_batch_normalization(x, center = True, scale = True, training = train)
    layer_3_drop = tf.nn.dropout(layer_3_norm, keep_prob = drop_out_prob)
    scores = tf.add(tf.matmul(layer_3_drop, W1), b1)
    prediction = tf.nn.softmax(scores)
    return prediction

out = classification_model(X, drop_out_prob)

loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(y, tf.float32) * tf.log(out), axis = 1)) + 0.0 * tf.reduce_sum(tf.multiply(W1, W1))

optimizer = tf.train.AdamOptimizer(5e-4)

train_step = optimizer.minimize(loss)
