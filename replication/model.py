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
    print("x shape for " + str(name))
    print(x.get_shape())
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take.

    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = tf.transpose(x, [1, 0, 2])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)

    return x_reformat

def dRNN(cell, inputs, rate, initial_state):
    n_steps = len(inputs)

    print("nsteps:",n_steps)
    print("rate:",rate)
    if rate < 0 or rate >= n_steps:
        raise ValueError("bad rate variable")

    # zero pad to make sure the number of inputs can be evenly divided by the rate
    if (n_steps % rate) != 0:
        print("PADDING")
        zero_tensor = tf.zeros_like(inputs[0])
        dilated_n_steps = n_steps // rate + 1 # NOTE: in python // is division with automatic floor

        for i_pad in range(dilated_n_steps * rate - n_stpes):
            inputs.append(zero_tensor)

    else:
        print("No padding necessary")
        dilated_n_steps = n_steps // rate

    print("Dilated:")
    print(dilated_n_steps)
        
    # Example:
    # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
    # zero-padding --> [x1, x2, x3, x4, x5, 0]
    # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
    # which the length is the ceiling of n_steps/rate
    dilated_inputs = [tf.concat(inputs[i * rate:(i+1) * rate], axis=0) for i in range(dilated_n_steps)]
    #for i in range(dilated_n_steps):
        #thing = tf.concat(inputs[i*rate:(i+1)*rate], axis=0)
        #print("thing:")
        #print(thing)

    print("inputs:")
    print(dilated_inputs)

    #dilated_outputs, final_state = rnn.static_rnn(cell, dilated_inputs, dtype=tf.float32, initial_state=initial_state) # TODO: don't know if this being static is going to cause issues or not, and it's not lstm? But lstm can be passed in?
    # TODO: is final_state dilated too? Do I have to handle this similar to how dilated_outputs are handled?

    # NOTE: the problem is initial state!!!! Original drnn didn't have initial state passed in either (unsure if this is the case for FuN or not
    
    dilated_outputs, final_state = rnn.static_rnn(cell, dilated_inputs, dtype=tf.float32) # TODO: don't know if this being static is going to cause issues or not, and it's not lstm? But lstm can be passed in?

    print("\nOutputs:")
    print(dilated_outputs)

    print("\nOutput Shapes:")
    for output in dilated_outputs:
        print(output.shape)

    splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
    unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]

    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]

    print("Actual outputs:")
    print(outputs)

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
    def __init__(self, ob_space, ac_space, local_steps, horizen):
        print("Ob space:")
        print(ob_space)
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3,3], [2,2]))

        

        print("x:")
        print(x.shape)
        #x = tf.expand_dims(flatten(x), [0])            
        #self.z = tf.reshape(x, [-1, 256])
        self.z = tf.reshape(x, [-1, 288])
        print("z:")
        print(self.z.shape)


        size = 256
        
        scope_name = tf.get_variable_scope().name

        # MANAGER NETWORK
        with tf.variable_scope(scope_name + "_m"):

            #self.s = tf.nn.elu(linear(self.z, EMBEDDING_DIMENSIONALITY, "mspace", normalized_columns_initializer(0.01))) # TODO: almost positive this is incorrect, supposed to be size?
            self.s = tf.nn.elu(linear(self.z, size, "mspace", normalized_columns_initializer(0.01))) 
            self.s_expanded = tf.expand_dims(self.s, 2) # NOTE: this used to be 0, not 2

            
            # dilated lstm
            
            m_lstm = rnn.BasicLSTMCell(size, state_is_tuple=True) # TODO: is tuple state going to be an issue?
            
            m_c_init = np.zeros((1, m_lstm.state_size.c), np.float32)
            m_h_init = np.zeros((1, m_lstm.state_size.h), np.float32)
            self.m_state_init = [m_c_init, m_h_init]

            
            self.m_state_size = m_lstm.state_size
            m_c_in = tf.placeholder(tf.float32, [1, m_lstm.state_size.c])
            m_h_in = tf.placeholder(tf.float32, [1, m_lstm.state_size.h])
            self.m_state_in = [m_c_in, m_h_in]

            m_state_in = rnn.LSTMStateTuple(m_c_in, m_h_in)
            print("M_state_in:")
            print(m_state_in)

            # TODO: there's no state_init for the manager network?

            # NOTE: expanded needs to be in shape (batch_size [?], n_steps [10{DILATION_RADIUS}], input_dims [1??])
            # (?, 256, 1)
            print("regular size:")
            print(self.s.shape)
            print("expanded size:")
            print(self.s_expanded.shape)
            reformatted = _rnn_reformat(self.s_expanded, 1, local_steps)
            #reformatted = _rnn_reformat(self.s_expanded, 256, local_steps) # TODO: no idea if 256 is correct. Default is 1?
            print("reformatted: ")
            print(reformatted)
            m_lstm_outputs, m_lstm_state = dRNN(m_lstm, reformatted, DILATION_RADIUS, m_state_in) # TODO: dunno if the input_dims of 1 is correct? (it was the default from the sample code)

            m_lstm_c, m_lstm_h = m_lstm_state
            self.m_state_out = [m_lstm_c[:1,:], m_lstm_h[:1,:]]
            
            
            # TODO: TODO: TODO: TODO: TODO: TODO: TODO: figure out below line!!!!!
            #m_lstm_outputs = m_lstm_outputs[-1] # NOTE: this gets only the last output from the last core. Is this correct assumption?

            g_ = m_lstm_outputs 
            g_norm = tf.sqrt(tf.reduce_sum(tf.square(g_), 0)) # TODO: don't know if reduce_sum dim of 1 is correct?
            self.g = g_ / g_norm
            
            self.g = tf.transpose(self.g, [1, 0, 2]) # (40, ?, 256) -> (?, 40, 256)
            
            self.m_vf = tf.reshape(linear(m_lstm_outputs[-1], 1, "m_value", normalized_columns_initializer(1.0)), [-1]) # NOTE: I think value calculated only from the last g_t ([-1])


            # TODO: stop gradient?
            print("goal shape?")
            print(self.g) # NOTE: (40, ?, 256) ?????? this doesn't seem correct
                    
            print("Weird slicing of g:")
            print(self.g[:,-horizen:].shape)
            self.pooled_goals = tf.reduce_sum(self.g[:,-horizen:], 1) # NOTE: not sure if just getting the last -horizen ACTUALLY gives us everything from this core or not, still not sure if drnn is returning the right thing
            print("Pooled goals shape:")
            print(self.pooled_goals.shape)
            
            # NOTE: keep in mind phi is technically trained as part of worker
            # NOTE: not using linear because no bias
            #self.phi_w = tf.get_variable("phi/w", [self.pooled_goals.get_shape()[1], size], initializer=normalized_columns_initializer(1.0))
            self.phi_w = tf.get_variable("phi/w", [self.pooled_goals.get_shape()[1], EMBEDDING_DIMENSIONALITY], initializer=normalized_columns_initializer(1.0))
            self.w = tf.matmul(self.pooled_goals, self.phi_w)
            self.w = tf.expand_dims(self.w, 2)



        self.var_list_m = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name + "_m")

        # WORKER NETWORK
        with tf.variable_scope(scope_name + "_w"):

            w_lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
            self.w_state_size = w_lstm.state_size
            stepsize = tf.shape(self.z)[:1]

            w_c_init = np.zeros((1, w_lstm.state_size.c), np.float32)
            w_h_init = np.zeros((1, w_lstm.state_size.h), np.float32)
            self.w_state_init = [w_c_init, w_h_init]

            w_c_in = tf.placeholder(tf.float32, [1, w_lstm.state_size.c])
            w_h_in = tf.placeholder(tf.float32, [1, w_lstm.state_size.h])
            self.w_state_in = [w_c_in, w_h_in]

            self.z_alt = tf.expand_dims(flatten(self.x), [0])
            self.w_step_size = tf.shape(self.x)[:1] # TODO: this might actually be self.x?

            w_state_in = rnn.LSTMStateTuple(w_c_in, w_h_in)
            w_lstm_outputs, w_lstm_state = tf.nn.dynamic_rnn(
                w_lstm, self.z_alt, initial_state=w_state_in, sequence_length=self.w_step_size,
                time_major=False)

            w_lstm_c, w_lstm_h = w_lstm_state
            print("original outputs shape:")
            print(w_lstm_outputs.shape)
            #w_lstm_outputs = tf.reshape(w_lstm_outputs, [-1, size])
            self.w_state_out = [w_lstm_c[:1, :], w_lstm_h[:1, :]]

            # NOTE: is U the direct lstm output? or is there in fact a linear layer inbetween? Idt there is
            #self.U = linear(w_lstm_outputs, ac_space, "U", normalized_columns_initializer(0.01))

            self.U = tf.reshape(w_lstm_outputs, [-1, 6, 16]) # TODO: is this cheating?? Does this just work?

            
            print("Ac space:")
            print(ac_space)
            print("outputs shape:")
            print(w_lstm_outputs.shape)
            print("U shape:")
            print(self.U.shape)
            print("w shape:")
            print(self.w.shape)

            # NOTE: I assume this is where value is calculated, but I don't actually know
            w_lstm_outputs = tf.reshape(w_lstm_outputs, [-1, size])
            self.w_vf = tf.reshape(linear(w_lstm_outputs, 1, "w_value", normalized_columns_initializer(1.0)), [-1])
            
            #self.action = tf.softmax(tf.matmul(self.U, self.w))
            #self.pi = tf.matmul(self.U, self.w)
            self.logits = tf.matmul(self.U, self.w)
            self.logits = tf.transpose(self.logits, [0, 2, 1])
            self.logits = tf.reshape(self.logits, [-1, 6])
            self.sample = categorical_sample(self.logits, ac_space)[0, :]
            
        self.var_list_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name + "_w")
        
    def get_initial_features(self):
        return self.w_state_init[0], self.w_state_init[1], self.m_state_init[0], self.m_state_init[1]

    # returns action, worker value, manager value, goal, worker state out, manager state out, latent state
    def act(self, ob, w_c, w_h, m_c, m_h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.w_vf, self.m_vf, self.g, self.w_state_out, self.m_state_out, self.s],
                {self.x: [ob], self.w_state_in[0]: w_c, self.w_state_in[1]: w_h, self.m_state_in[0]: m_c, self.m_state_in[1]: m_h})

    def value(self, ob, w_c, w_h, m_c, m_h):
        sess = tf.get_default_session()
        values = sess.run([self.w_vf, self.m_vf], {self.x: [ob], self.w_state_in[0]: w_c, self.w_state_in[1]: w_h, self.m_state_in[0]: m_c, self.m_state_in[1]: m_h})
        return values[1][0] # NOTE: only returning manager because that is prediction only of environment reward?
