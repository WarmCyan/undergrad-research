import tensorflow as tf
from tensorflow.contrib import rnn

import collections

# function to build a dataset (returns a unique id associated with each word)
# this particular example mapping is built by frequency of each word. (So the most common word is id 0, next most common is 1, etc)
def build_dictionary(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


def build_dataset(dictionary, data):
    entries = []
    for i in range(0, len(data)-3):
        entry = []
        entry.append([])
        entry[0].append(dictionary[data[i]])
        entry[0].append(dictionary[data[i+1]])
        entry[0].append(dictionary[data[i+2]])
        entry.append([])
        entry[1].append(dictionary[data[i+3]])

        entries.append(entry)
    return entries 


# build dataset from one of Aesop's short stoies
data = "long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies ."
data = data.split(' ')

dictionary, reverse_dictionary = build_dictionary(data)
feed = build_dataset(dictionary, data)
print(feed)


sess = tf.InteractiveSession()

# set up network variables
vocab_size = len(dictionary)

n_input = 3 # TODO: grab 3 words at a time as input?
n_hidden = 512 # number of hidden units in the RNN cell
batch_size = 1

weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
biases = tf.Variable(tf.random_normal([vocab_size]))

x = tf.placeholder(tf.int32, [None, n_input])
y_ = tf.placeholder(tf.int32, [None, 1])

# 1-layer LSTM with n-hidden units
lstm = rnn.BasicLSTMCell(n_hidden)

#currentState = tf.zeros(lstm.state_size)
#hiddenState = tf.zeros(batch_size, lstm.state_size)
#initial_state = state = tf.zeros([batch_size, lstm.state_size[1]])
initial_state = state = tf.zeros([batch_size, n_hidden])

#probabilities = []

output, state = lstm(x, state)

logits = tf.matmul(output, weights) + biases
#y = tf.nn.softmax(logits)
#probabilities.append(tf.nn.softmax(logits))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

for entry in feed:
    print("Running")
    sess.run([ ], feed_dict={x: entry[0], y_: entry[1]})




# reshape to [1, n_input]
#x = tf.reshape(x, [-1, n_input]) # TODO: why -1? What does that do?

# Generate an n_input-element sequence of inputs
# (eg. [had] [a] [general] -> [20] [6] [33])
#x = tf.split(x,n_input,1)

# generate prediction
#outputs, states = rnn.static_rnn(lstm, x, dtype=tf.float32)

# there are n_input outputs but we only want the last output
#rnn_out = tf.matmul(outputs[-1], weights) + biases




print("Done")
