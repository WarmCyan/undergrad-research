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


        self.input = tf.placeholder(tf.float32, shape=(84,84,4)) # TODO: pretty sure that shape isn't right

        # 32 filters
        # TODO: kernel size (specified 8x8) is 8?
        # stride of 4
        self.conv1 = tf.layers.conv2d(self.input, 32, 8, 4, activation=tf.nn.relu)
        
        pass

