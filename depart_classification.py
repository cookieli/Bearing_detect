import numpy as np
import tensorflow as tf
from autoencoder import num_hidden_3, num_hidden_4, num_input
from autoencoder_depart import Encoder, scope_name
import math
import matplotlib.pyplot as plt

num_classes = 10

W1 = tf.get_variable('W1', shape = [num_hidden_4, num_classes], dtype = tf.float32, initializer = tf.random_normal_initializer(stddev = 0.1))
b1 = tf.get_variable('b1', shape = [num_classes], dtype = tf.float32, initializer = tf.random_normal_initializer)

X = tf.placeholder(tf.float32, [None, num_input], "dcf")
y = tf.placeholder(tf.int64, [None, num_classes], "labels")
drop_out = tf.placeholder(tf.float32, name = "droping")
#training = tf.placeholder(tf.bool, name = "training")


def classification_model(x, drop_out_prob = 1.0, train = False):
    x_norm = tf.layers.batch_normalization(x, center = True, scale = True, training = train)
    layer_1 = Encoder(x_norm, scope_name[0], need_reuse = True)
    layer_1_norm = tf.layers.batch_normalization(layer_1, center = True, scale = True, training = train)
    layer_1_drop = tf.nn.dropout(layer_1_norm, keep_prob = drop_out_prob)
    layer_2 = Encoder(layer_1_drop, scope_name[1], need_reuse = True)
    layer_2_norm = tf.layers.batch_normalization(layer_2, center = True, scale = True, training = train)
    layer_2_drop = tf.nn.dropout(layer_2_norm, keep_prob = drop_out_prob)
    layer_3 = Encoder(layer_2_drop, scope_name[2], need_reuse = True)
    layer_3_norm = tf.layers.batch_normalization(layer_3, center = True, scale = True, training = train)
    layer_3_drop = tf.nn.dropout(layer_3_norm, keep_prob = drop_out_prob)
    layer_4 = Encoder(layer_3_drop, scope_name[3], need_reuse = True)
    layer_4_norm = tf.layers.batch_normalization(layer_4, center = True, scale = True, training = train)
    layer_4_drop = tf.nn.dropout(layer_4_norm, keep_prob = drop_out_prob)
    scores = tf.add(tf.matmul(layer_4_drop, W1), b1)
    prediction = tf.nn.softmax(scores)
    return prediction
#with tf.name_scope(" softmax scores"):
out = classification_model(X, drop_out)
#0/7
with tf.name_scope("loss"):
    loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(y, tf.float32) * tf.log(out), axis = 1)) 
    #loss = tf.losses.softmax_cross_entropy(y, out)
optimizer = tf.train.AdamOptimizer(1.8e-3)

train_step = optimizer.minimize(loss)


def run_model(session, predict, loss_val, Xd, yd, drop_prob = 1.0,
              epochs=1, batch_size=256, print_every = 100,
              training = None, plot_losses=False):
    correct_prediction = tf.equal(tf.argmax(predict, 1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #shuffle indices
    train_indices = np.arange(Xd.shape[0])
    np.random.shuffle(train_indices)

    training_now = training is not None

    variables = [loss_val, correct_prediction,  accuracy]
    if training_now:
        variables[-1] = training

    #counter
    iter_cnt = 0
    total_losses = []
    total_accs = []
    for e in range(epochs):
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indices[start_idx: start_idx+batch_size]

            feed_dict = {
                X:Xd[idx, :],
                y: yd[idx],
                drop_out: drop_prob,
               }
            actual_batch_size = yd[idx].shape[0]
            loss, corr,_= session.run(variables, feed_dict=feed_dict)

            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)

            if training_now and (iter_cnt%print_every) == 0:
                print("Iteration{0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        total_losses.append(total_loss)
        total_accs.append(total_correct)
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss, total_correct, e+1))
    if plot_losses:
        plt.plot(total_accs)
        plt.grid(True)
        plt.title('Epoch {} accuracy'.format(e+1))
        plt.xlabel('minibatch number')
        plt.ylabel('minibatch accuracy')
        plt.show()
    return total_loss, total_correct



