import tensorflow as tf
import gym
import numpy as np


# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/ 
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/ 
# https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

class Environment:
    def __init__(self):
        pass

class Agent:
    def __init__(self, numPossibleActions):
        self.sess = tf.Session()

        self.numPossibleActions = numPossibleActions

        print("Agent initialized")


    def buildGraph(self):
        
        # NOTE: same preprocessing as in DQN paper

        self.input = tf.placeholder(tf.float32, shape=(1, 84,84,4), name='input') # TODO: pretty sure that shape isn't right
        
        # (32 filters, kernel size of 8, stride of 4)
        # 16 filters, kernel size of 8, stride of 4
        with tf.name_scope('conv1'):
            self.w1 = tf.Variable(tf.random_normal([8, 8, 4, 16]), name='weights1')
            self.b1 = tf.Variable(tf.random_normal([16]), name='bias1')
            self.conv1 = tf.nn.conv2d(self.input, self.w1, [1, 4, 4, 1], "VALID", name='conv1') 
            self.conv1_relu = tf.nn.relu(tf.nn.bias_add(self.conv1, self.b1))
            
        # 64 filters, kernel size of 4, stride of 2
        # 32 filters, kernel size of 4, stride of 2
        with tf.name_scope('conv2'):
            self.w2 = tf.Variable(tf.random_normal([4, 4, 16, 32]), name='weights2')
            self.b2 = tf.Variable(tf.random_normal([32]), name='bias2')
            self.conv2 = tf.nn.conv2d(self.conv1_relu, self.w2, [1, 2, 2, 1], "VALID", name='conv2') 
            self.conv2_relu = tf.nn.relu(tf.nn.bias_add(self.conv2, self.b2))

            # flattened size is 9*9*32 = 2592
            self.conv2_out = tf.reshape(self.conv2_relu, [-1, 2592], name='conv2_flatten') 
            

        with tf.name_scope('fully_connected'):
            self.fc_w = tf.Variable(tf.random_normal([2592, 256]), name='fc_weights') 
            self.fc_b = tf.Variable(tf.random_normal([256]), name='fc_biases') # fully connected biases

            self.fc_out = tf.nn.relu_layer(self.conv2_out, self.fc_w, self.fc_b, name='fc_out')


        with tf.name_scope('policy'):
            self.policy_w = tf.Variable(tf.random_normal([256, self.numPossibleActions]), name='policy_w')
            
            # TODO: do we need biases as well?
            self.policy_out = tf.nn.softmax(tf.matmul(self.fc_out, self.policy_w))

        # TODO: is this supposed to be a single output node, or one for each action?
        # (is it a global value, or value-action pairs?
        with tf.name_scope('value'):
            #self.value_w = tf.Variable(tf.random_normal([256, 1]), name='value_w')
            self.value_w = tf.Variable(tf.random_normal([256, self.numPossibleActions]), name='value_w')

            self.value_out = tf.matmul(self.fc_out, self.value_w)


        #self.testOutput = self.conv2_relu + 1


        self.merged_summaries = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        
        self.train_writer = tf.summary.FileWriter('../tensorboard_data/a3c' , self.sess.graph)
        self.train_writer.add_graph(self.sess.graph)


a = Agent(6)
a.buildGraph()
