# https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/

import tensorflow as tf
from tensorflow.contrib import rnn


# import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

time_steps = 28 # unrolled through 28 steps
num_units = 128 # 128 lstm units
n_input = 28 # 28 pixels
learning_rate = 0.001
n_classes = 10 # prediction class count
batch_size = 128


# placeholders (input/target)
x = tf.placeholder("float", [None, time_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# variables
weights = tf.Variable(tf.random_normal([num_units, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

# TODO: explore below step
#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input = tf.unstack(x, time_steps, 1)

# definte network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

# get output at last time step for prediction
pred = tf.matmul(outputs[-1], weights) + biases

# define loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while iter < 800:
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
        batch_x = batch_x.reshape((batch_size, time_steps, n_input)) # TODO: what does this do??

        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

        # log every 10th run
        if iter % 10 == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_x, y:batch_y})
            los = sess.run(loss,feed_dict={x:batch_x, y:batch_y})
            print("For iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")

        iter=iter+1
