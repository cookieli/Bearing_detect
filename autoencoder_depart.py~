from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import autoencoder as ae

X = tf.placeholder(dtype = tf.float32, shape =[None, ae.num_input])
is_training = tf.placeholder(tf.bool)
learning_rate = 0.01

layer_shape = {
    'layer_1': {
        'encoder': {'weights': [ae.num_input, ae.num_hidden_1], 'bias': [ae.num_hidden_1]},
        'decoder': {'weights': [ae.num_hidden_1, ae.num_input], 'bias': [ae.num_input]},
    },
    'layer_2': {
        'encoder': {'weights': [ae.num_hidden_1, ae.num_hidden_2], 'bias': [ae.num_hidden_2]},
        'decoder': {'weights': [ae.num_hidden_2, ae.num_hidden_1], 'bias': [ae.num_hidden_1]},
    },
    'layer_3': {
        'encoder': {'weights': [ae.num_hidden_2, ae.num_hidden_3], 'bias': [ae.num_hidden_3]},
        'decoder': {'weights': [ae.num_hidden_3, ae.num_hidden_2], 'bias': [ae.num_hidden_2]},
    }
}

scope_name = ['layer_1', 'layer_2', 'layer_3']
def encoder(x, kernel_shape, bias_shape, train = False):
    # create weights
    weights = tf.get_variable("weights", kernel_shape,
                              initializer = tf.random_normal_initializer(stddev = 0.1))
    # create bias
    bias = tf.get_variable("bias", bias_shape,
                           initializer = tf.random_normal_initializer)
    # x_norm = tf.layers_batch_normalization(x, center = True, scale = True, training = train)
    out = tf.tanh(tf.add(tf.matmul(x, weights),
                         bias))
    return out

def Encoder(x, layer_name, train = False, need_reuse = True):
    with tf.variable_scope(layer_name, reuse = tf.AUTO_REUSE):
        with tf.variable_scope('Encoder'):
            return encoder(x,
                           layer_shape[layer_name]['encoder']['weights'],
                           layer_shape[layer_name]['encoder']['bias'],
                           train)

def decoder(x, kernel_shape, bias_shape, train = False):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer = tf.random_normal_initializer(stddev = 0.1))
    bias = tf.get_variable("bias", bias_shape,
                           initializer = tf.random_normal_initializer(stddev = 0.1))
    out = tf.tanh(tf.add(tf.matmul(x, weights),
                         bias))
    return out
def Decoder(x, layer_name, train = False, need_reuse = True):
    with tf.variable_scope(layer_name, reuse = tf.AUTO_REUSE):
        with tf.variable_scope('Decoder'):
            return decoder(x,
                           layer_shape[layer_name]['decoder']['weights'],
                           layer_shape[layer_name]['decoder']['bias'],
                           train)

def autoencoder(x, layer_name, e_kernel_shape, e_bias_shape, d_kernel_shape, d_bias_shape, train = False):
    x_norm = tf.layers_batch_normalization(x, center = True, scale = True, training = train)
    with tf.variable_scope(layer_name):
        encoder_out = encoder(x_norm, e_kernel_shape, e_bias_shape, train)
        encoder_norm = tf.layers_batch_normalization(encoder_out,center = True, scale = True, training = train)
        decoder_out = encoder(encoder_norm, d_kernel_shape, d_kernel_shape, train)
        return (encoder_norm, decoder_out)
def Autoencoder(x, layer_name, train = False):
    x_norm = tf.layers.batch_normalization(x, center = True, scale = True, training = train)
    encoder_out = Encoder(x_norm, layer_name, train)
    encoder_norm = tf.layers.batch_normalization(encoder_out, center = True, scale = True, training = train)
    decoder_out = Decoder(encoder_norm, layer_name, train)
    return decoder_out


def layer_para(x, layer_name, train = False):
    dic ={}
    dic['y_true'] = x
    y_true = dic['y_true']
    dic['y_pred'] = Autoencoder(x, layer_name, train)
    y_pred = dic['y_pred']
    with tf.variable_scope(layer_name, reuse = tf.AUTO_REUSE):
        with tf.variable_scope('layer_para'):
            dic['loss'] = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            loss = dic['loss']
            dic['optimizer'] = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    return dic

#layer_1_para = layer_para(X, scope_name[0], is_training)

#layer_2_para = layer_para(x = Encoder(X, scope_name[0], False, True), layer_name = scope_name[1], train = is_training)

#layer_3_para = layer_para(x = Encoder(X, scope_name[1], False, True), layer_name = scope_name[2], train = is_training)

