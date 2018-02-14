import tensorflow as tf
import gym
import numpy as np
#from scipy.misc import imresize
from skimage.transform import resize


# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py 


# create an environment

class Environment:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        #state_0 = env.reset()
        print("Environment initialized")

    def preprocessFrame(self, frame):
        greyFrame = np.dot(frame[...,:3], [.3, .6, .1]) # convert to grey scale? TODO: don't think that's actually what it does?
        return resize(greyFrame, (84,84))
        

class Agent:
    def __init__(self):
        self.sess = tf.Session()
        print("agent initialized")


    def buildGraph(self):
        #self.input = tf.placeholder(tf.float32, shape=(84,84,4)) # TODO: pretty sure that shape isn't right
        self.input = tf.placeholder(tf.float32, shape=(1, 84,84,4)) # TODO: pretty sure that shape isn't right

        # convolutional layers

        # TODO: need weights and biases between each layer right?? Or is this automatically handled?
        # NOTE: I assume yes, tf docs mentione "trainable" variable which adds variables to graph?
        
        # 32 filters, kernel size of 8, stride of 4
        self.conv1 = tf.layers.conv2d(self.input, 32, 8, 4, activation=tf.nn.relu, name='conv1')
        
        # 64 filters, kernel size of 4, stride of 2
        self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, 2, activation=tf.nn.relu, name='conv2')
        
        # 64 filters, kernel size of 3, stride of 1
        self.conv3 = tf.layers.conv2d(self.conv2, 64, 3, 1, activation=tf.nn.relu, name='conv3')

        # NOTE: think I might need to flatten output of conv3?

        self.conv3_out = tf.reshape([-1])

        # fully conected layer
        self.fc_w = tf.Variable(tf.random_normal([64, 512]), name='fc_weights') # fully connected weights TODO: no idea if size is right
        #self.fc_w = tf.Variable(tf.random_normal([64, 512])) # fully connected weights TODO: no idea if size is right
        self.fc_b = tf.Variable(tf.random_normal([512]), name='fc_biases') # fully connected biases

        #fc_out = tf.matmul(self.conv3, fc_w) + fc_b
        self.fc_out = tf.nn.relu_layer(self.conv3, self.fc_w, self.fc_b)

        #  
        #self.out_w = tf.Variable(tf.random_normal([512, 6]))
        #self.out_b = tf.Variable(tf.random_normal(6))

        #self.choice = tf.matmul(self.fc_out, self.out_w) + self.out_b
        
        
        # NOTE: 6 possible outputs for space invaders
        # final 


        self.merged_summaries = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter('../tensorboard_data/cnn_atari2' , self.sess.graph)
        #self.test_writer = tf.summary.FileWriter(self.logdir + self.name)
        self.train_writer.add_graph(self.sess.graph)
        #self.test_writer.add_graph(self.session.graph) 
        

a = Agent()
a.buildGraph()
print("done")
