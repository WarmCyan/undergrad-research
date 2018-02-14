import tensorflow as tf
import gym
import numpy as np
#from scipy.misc import imresize
from skimage.transform import resize

# create an environment

class Environment:
    def __init__():
        self.env = gym.make('SpaceInvaders-v0')
        #state_0 = env.reset()

    def preprocessFrame(self, frame):

        greyFrame = np.dot(frame[...,:3], [.3, .6, .1]) # convert to grey scale? TODO: don't think that's actually what it does?
        
        return resize(greyFrame, (64,64))
        

class Agent:
    def __init__():
