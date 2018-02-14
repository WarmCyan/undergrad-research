import tensorflow as tf
import gym
import numpy as np
#from scipy.misc import imresize
from skimage.transform import resize


# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py 


# create an environment

class Environment:
    def __init__():
        self.env = gym.make('SpaceInvaders-v0')
        #state_0 = env.reset()

    def preprocessFrame(self, frame):
        greyFrame = np.dot(frame[...,:3], [.3, .6, .1]) # convert to grey scale? TODO: don't think that's actually what it does?
        return resize(greyFrame, (84,84))
        

class Agent:
    def __init__():
        self.sess = tf.Session()


    def buildGraph():

        self.input = tf.placeholder(tf.float32, shape=(84,84,4)) # TODO: pretty sure that shape isn't right

        # convolutional layers
        

        # TODO: need weights and biases between each layer right?? Or is this automatically handled?
        # NOTE: I assume yes, tf docs mentione "trainable" variable which adds variables to graph?
        
        # 32 filters, kernel size of 8, stride of 4
        self.conv1 = tf.layers.conv2d(self.input, 32, 8, 4, activation=tf.nn.relu)
        
        # 64 filters, kernel size of 4, stride of 2
        self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, 2, activation=tf.nn.relu)
        
        # 64 filters, kernel size of 3, stride of 1
        self.conv3 = tf.layers.conv2d(self.conv2, 64, 3, 1, activation=tf.nn.relu)

        # NOTE: think I might need to flatten output of conv3?

        # fully conected layer
        fc_w = tf.Variable(tf.random_normal([64, 512]) # fully connected weights TODO: no idea if size is right
        fc_b # fully connected biases
        
        # final 
