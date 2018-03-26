import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')



# https://github.com/code-terminator/DilatedRNN/blob/master/models/drnn.py


EMBEDDING_DIMENSIONALITY = 16
DILATION_RADIUS = 10




def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


def dRNN(cell, inputs, rate, initial_state):
    n_steps = len(inputs)

    if rate < 0 or rate >= n_steps:
        raise ValueError("bad rate variable")

    # zero pad to make sure the number of inputs can be evenly divided by the rate
    if (n_steps % rate) != 0:
        zero_tensor = tf.zeros_like(inputs[0])
        dilated_n_steps = n_steps // rate + 1 # NOTE: in python // is division with automatic floor

        for i_pad in range(dilated_n_steps * rate - n_stpes):
            inputs.append(zero_tensor)

    else:
        dilated_n_steps = n_stepsps // rate
        
    # Example:
    # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
    # zero-padding --> [x1, x2, x3, x4, x5, 0]
    # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
    # which the length is the ceiling of n_steps/rate
    dilated_inputs = [tf.concat(inputs[i * rate(i+1) * rate], axis=0) for i in range(dilated_n_steps)]

    dilated_outputs, final_state = rnn.static_rnn(cell, dilated_inputs, dtype=tf.float32, initial_state=initial_state) # TODO: don't know if this being static is going to cause issues or not, and it's not lstm? But lstm can be passed in?
    # TODO: is final_state dilated too? Do I have to handle this similar to how dilated_outputs are handled?

    splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
    unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]

    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]

    return outputs, final_state


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])

        size = 256
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


class FuNPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = tf.placeholder(tf.float32, [None] + list(ob_space))

        self.z = tf.expand_dims(flatten(tf.nn.elu(conv2d(self.x, 32,  "l{}".format(i+1), [3,3], [2,2]))), [0])


        size = 256
        
        scope_name = tf.get_variable_scope().name

        # MANAGER NETWORK
        with tf.variable_scope(scope_name + "_m"):

            #self.s = tf.nn.elu(linear(self.z, EMBEDDING_DIMENSIONALITY, "mspace", normalized_columns_initializer(0.01))) # TODO: almost positive this is incorrect, supposed to be size?
            self.s = tf.nn.elu(linear(self.z, size, "mspace", normalized_columns_initializer(0.01))) # TODO: almost positive this is incorrect, supposed to be size?

            
            # TODO: dilated lstm
            
            m_lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True) # TODO: is tuple state going to be an issue?
            self.m_state_size = m_lstm.state_size
            m_c_in = tf.placeholder(tf.float32, [1, m_lstm.state_size.c])
            m_h_in = tf.placeholder(tf.float32, [1, m_lstm.state_size.h])
            self.m_state_in = [m_c_in, m_h_in]

            m_state_in = rnn.rnn_cell.LSTMStateTuple(m_c_in, m_h_in)
            m_lstm_outputs, m_lstm_state = dRNN(m_lstm, self.s, DILATION_RADIUS, m_state_in)
            
            
            #m_lstm_outputs = None # TODO: might have to reshape?

            g_ = m_lstm_outputs # NOTE: again, not sure if this is true, also may need to reshape? TODO: only take the last 16? (would have to match EMBEDDING_DIMENSIONALITY in order for cosine similarity to work
            g_norm = tf.sqrt(tf.reduce_sum(tf.square(g_), 1)) # TODO: don't know if reduce_sum dim of 1 is correct?
            self.g = g_ / g_norm
            
            self.m_vf = tf.reshape(linear(m_lstm_outputs, 1, "m_value", normalized_columns_initializer(1.0)), [-1])


            # TODO: stop gradient?
            self.pooled_goals = reduce_sum(self.g, 0) # TODO: which dimension are these pooled???
            
            # NOTE: keep in mind phi is technically trained as part of worker
            # NOTE: not using linear because no bias
            self.phi_w = tf.get_variable("phi/w", [self.pooled_goals.get_shape()[1], size], initializer=normalized_columns_initializer(1.0))
            self.w = matmul(self.pooled_goals, self.phi_w)



        self.var_list_m = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name + "_m")

        # WORKER NETWORK
        with tf.variable_scope(scope_name + "_w"):

            w_lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
            self.w_state_size = w_lstm.state_size
            stepsize = tf.shape(self.z)[:1]

            w_c_init = np.zeros((1, w_lstm.state_size.c), np.float32)
            w_h_init = np.zeros((1, w_lstm.state_size.h), np.float32)
            self.w_state_init = [w_c_init, w_h_init]

            w_c_in = tf.placeholder(tf.float32, [1, w_lstm.state_size.c])
            w_h_in = tf.placeholder(tf.float32, [1, w_lstm.state_size.h])
            self.w_state_in = [w_c_in, w_h_in]

            w_state_in = rnn.rnn_cell.LSTMStateTuple(w_c_in, w_h_in)
            w_lstm_outputs, w_lstm_state = tf.nn.dynamic_rnn(
                w_lstm, self.z, initial_state=w_state_in, sequence_length=w_step_size,
                time_major=False)

            w_lstm_c, w_lstm_h = w_lstm_state
            w_lstm_outputs = tf.reshape(w_lstm_outputs, [-1, size])
            self.w_state_out = [w_lstm_c[:1, :], w_lstm_h[:1, :]]

            # NOTE: is U the direct lstm output? or is there in fact a linear layer inbetween? Idt there is
            self.U = linear(w_lstm_outputs, ac_space, "U", normalized_columns_initializer(0.01))

            # NOTE: I assume this is where value is calculated, but I don't actually know
            self.w_vf = tf.reshape(linear(w_lstm_outputs, 1, "w_value", normalized_columns_initializer(1.0)), [-1])
            
            #self.action = tf.softmax(tf.matmul(self.U, self.w))
            self.pi = tf.matmul(self.U, self.w)
            
        self.var_list_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name + "_w")
