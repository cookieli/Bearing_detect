from __future__ import division, print_function, absolute_import
from autoencoder import mnist, num_hidden_1, num_hidden_2, num_hidden_3, num_input,features, encoder, weights
import tensorflow as tf
import numpy as np
import math
num_classes = 10
learning_rate = 0.01
X = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.int64, [None, num_classes])
W1 = tf.get_variable('W1', shape = [num_hidden_3, num_classes], dtype = tf.float32, initializer = tf.random_normal_initializer)
b1 = tf.get_variable("b1", shape = [num_classes], dtype = tf.float32, initializer = tf.random_normal_initializer)
def classification_model(X):
    layer1 = encoder(X)
    layer_1_norm = tf.layers.batch_normalization(X, center = True, scale = True)
    scores = tf.add(tf.matmul(layer1, W1), b1)
    prediction = tf.nn.softmax(scores)
    return prediction

out = classification_model(X)

loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(y,tf.float32)*tf.log(out),axis = 1))
optimizer = tf.train.AdamOptimizer(5e-4)
train_step = optimizer.minimize(loss)


train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=256, print_every = 100,
              training = None, plot_losses=False):
    correct_prediction = tf.equal(tf.argmax(predict, 1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #shuffle indices
    train_indices = np.arange(Xd.shape[0])
    np.random.shuffle(train_indices)

    training_now = training is not None

    variables = [loss_val, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    #counter
    iter_cnt = 0
    for e in range(epochs):
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indices[start_idx: start_idx+batch_size]

            feed_dict = {
                X:Xd[idx, :],
                y:yd[idx],
            }
            actual_batch_size = yd[idx].shape[0]
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)

            if training_now and (iter_cnt%print_every) == 0:
                print("Iteration{0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss, total_correct, e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct



