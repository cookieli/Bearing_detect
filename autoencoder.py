from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Training parameters
learning_rate = 2.5e-4
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

#Network Parameters

num_hidden_1 = 512
num_hidden_2 = 256
num_hidden_3 = 128
num_input = 1200

X = tf.placeholder("float", [None, num_input])
is_training = tf.placeholder(tf.bool)
drop_out_prob = tf.placeholder(tf.float32)
weights = {
    'encoder_h1': tf.get_variable("encoder_h1",
                                  shape = [num_input, num_hidden_1],
                                  initializer = tf.random_normal_initializer(stddev = 0.1)),
    'encoder_h2': tf.get_variable("encoder_h2",
                                  shape = [num_hidden_1, num_hidden_2],
                                  initializer = tf.random_normal_initializer(stddev = 0.1)),
    'encoder_h3': tf.get_variable("encoder_h3",
                                  shape = [num_hidden_2, num_hidden_3],
                                  initializer = tf.random_normal_initializer(stddev = 0.1)),
    'decoder_h1': tf.get_variable("decoder_h1",
                                  shape = [num_hidden_3, num_hidden_2],
                                  initializer = tf.random_normal_initializer(stddev = 0.1)),
    'decoder_h2': tf.get_variable("decoder_h2",
                                  shape = [num_hidden_2, num_hidden_1],
                                  initializer = tf.random_normal_initializer(stddev = 0.1)),
    'decoder_h3': tf.get_variable("decoder_h3",
                                  shape = [num_hidden_1, num_input],
                                  initializer = tf.random_normal_initializer(stddev = 0.1))
}
biases = {
    'encoder_b1': tf.get_variable("encoder_b1",
                                  shape = [num_hidden_1],
                                  initializer = tf.random_normal_initializer),
    'encoder_b2': tf.get_variable("encoder_b2",
                                  shape = [num_hidden_2],
                                  initializer = tf.random_normal_initializer),
    'encoder_b3': tf.get_variable("encoder_b3",
                                  shape = [num_hidden_3],
                                  initializer = tf.random_normal_initializer),
    'decoder_b1': tf.get_variable("decoder_b1",
                                  shape = [num_hidden_2],
                                  initializer = tf.random_normal_initializer),
    'decoder_b2': tf.get_variable("decoder_b2",
                                  shape = [num_hidden_1],
                                  initializer = tf.random_normal_initializer),
    'decoder_b3': tf.get_variable("decoder_b3",
                                  shape = [num_input],
                                  initializer = tf.random_normal_initializer)
}

#Building the encoder
def encoder(x, train = False, drop_prob = 0.4):
    x_norm = tf.layers.batch_normalization(x, center = True, scale = True, training = train)
    x_drop = tf.nn.dropout(x_norm, keep_prob = drop_prob)
    layer_1 = tf.tanh(tf.add(tf.matmul(x_drop, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_1_norm = tf.layers.batch_normalization(layer_1, center = True, scale = True, training = train)
    layer_1_drop = tf.nn.dropout(layer_1_norm, keep_prob = drop_prob)
    layer_2 = tf.tanh(tf.add(tf.matmul(layer_1_drop, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_2_norm = tf.layers.batch_normalization(layer_2, center = True, scale = True, training = train)
    layer_2_drop = tf.nn.dropout(layer_2_norm, keep_prob = drop_prob)
    layer_3 = tf.tanh(tf.add(tf.matmul(layer_2_drop, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3
def decoder(x, train = False, drop_prob = 0.4):
    x_norm = tf.layers.batch_normalization(x, center = True, scale = True, training = train)
    x_drop = tf.nn.dropout(x_norm, keep_prob = drop_prob)
    layer_1 = tf.tanh(tf.add(tf.matmul(x_drop, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_1_norm = tf.layers.batch_normalization(layer_1, center = True, scale = True, training = train)
    layer_1_drop = tf.nn.dropout(layer_1_norm, keep_prob = drop_prob)
    layer_2 = tf.tanh(tf.add(tf.matmul(layer_1_drop, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_2_norm = tf.layers.batch_normalization(layer_2, center = True, scale = True, training = train)
    layer_2_drop = tf.nn.dropout(layer_2_norm, keep_prob = drop_prob)
    layer_3 = tf.tanh(tf.add(tf.matmul(layer_2_drop, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3

#encoder_op = encoder(X)
#decoder_op = decoder(encoder_op)
#y_pred = decoder_op
#y_true = X
def autoencoder(x, train = False, drop_prob = 0.4):
    encoder_op = encoder(x,train, drop_prob)
    decoder_op = decoder(encoder_op,train, drop_prob)
    return decoder_op, encoder_op
y_true = X
y_pred, features = autoencoder(X, is_training, drop_out_prob)
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

def run_autoencoder(session, loss_val, Xd, predict,drop_prob = 0.4,
                    num_steps = 40, batch_size=256, print_every = 100,
                    training = None, plot_losses = False):
    correct_prediction = y_true - y_pred
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    train_indices = np.arange(Xd.shape[0])
    np.random.shuffle(train_indices)

    variables = [loss_val, correct_prediction, training, extra_update_ops]
   
    iter_cnt = 0
    correct = 0
    losses = []
    for i in range(num_steps):
        start_idx = (i * batch_size)%Xd.shape[0]
        idx = train_indices[start_idx: start_idx + batch_size]

        feed_dict = {X: Xd[idx, :], is_training: training is not None}
        actual_batch_size = Xd[idx].shape[0]
        loss, corr, _, _ = session.run(variables, feed_dict = feed_dict)
        losses.append(loss * actual_batch_size)
        correct += np.sum(corr)

        if (iter_cnt%print_every) == 0:
            print("Iteration{0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt, loss, np.sum(corr)))
        iter_cnt += 1
    total_loss = np.sum(losses)/Xd.shape[0]
    print("Overall loss = {0:.3g} ".format(total_loss))
    if plot_losses:
        plt.plot(losses)
        plt.grid(True)
        plt.title('Loss')
        plt.xlabel("minibatch number")
        plt.ylabel('minibatch loss')
        plt.show()
    return total_loss







