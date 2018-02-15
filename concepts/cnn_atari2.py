import tensorflow as tf
import gym
import numpy as np
#from scipy.misc import imresize
from skimage.transform import resize
from skimage.color import rgb2grey
from skimage.io import imsave

import random


# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py 
# https://github.com/devsisters/DQN-tensorflow/tree/master/dqn 
# http://cs231n.github.io/convolutional-networks/#conv 


# https://ai.intel.com/demystifying-deep-reinforcement-learning/ 
# https://danieltakeshi.github.io/2016/12/01/going-deeper-into-reinforcement-learning-understanding-dqn/ 


# create an environment

class Environment:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        #state_0 = env.reset()
        print("Environment initialized")

    def getInitialState(self):
        return self.preprocessFrame(self.env.reset())

    def preprocessFrame(self, frame):
        frame = resize(frame, (110,84))
        frame = frame[18:102,0:84]
        frame = rgb2grey(frame)
        frame = np.reshape(frame, (1,84,84,1))
        return frame

    def act(self, action):
        observation, reward, done, info = self.env.step(action)
        if done: print("DONE")

        observationFrame = self.preprocessFrame(observation)
        return observationFrame, reward, done


class Agent:
    def __init__(self):
        self.sess = tf.Session()
        self.epsilon = 1.0
        #self.epsilon = 0.1
        print("agent initialized")


    def buildGraph(self):
        #self.input = tf.placeholder(tf.float32, shape=(1, 84,84,4)) # TODO: pretty sure that shape isn't right
        self.input = tf.placeholder(tf.float32, shape=(1,84,84,1)) # TODO: pretty sure that shape isn't right

        # convolutional layers

        # 32 filters, kernel size of 8, stride of 4
        self.conv1 = tf.layers.conv2d(self.input, 32, 8, 4, activation=tf.nn.relu, name='conv1')
        
        # 64 filters, kernel size of 4, stride of 2
        self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, 2, activation=tf.nn.relu, name='conv2')
        
        # 64 filters, kernel size of 3, stride of 1
        self.conv3 = tf.layers.conv2d(self.conv2, 64, 3, 1, activation=tf.nn.relu, name='conv3')
        self.conv3_out = tf.reshape(self.conv3, [-1, 3136], name='conv3_flatten')


        # fully conected layer
        with tf.name_scope('fully_connected'):
            self.fc_w = tf.Variable(tf.random_normal([3136, 512]), name='fc_weights') 
            self.fc_b = tf.Variable(tf.random_normal([512]), name='fc_biases') # fully connected biases
            #self.output = self.conv3_out + 1

            self.fc_out = tf.nn.relu_layer(self.conv3_out, self.fc_w, self.fc_b, name='fc_out')

        # output layer (for space invaders, 6 possible outputs)
        with tf.name_scope('output'):
            self.out_w = tf.Variable(tf.random_normal([512, 6]), name='out_weights')
            self.out_b = tf.Variable(tf.random_normal([6]), name='out_biases')

            self.output = tf.matmul(self.fc_out, self.out_w) + self.out_b

        #self.summary_op = tf.summary.tensor_summary("output", self.output)

        #for i in range(self.output.get_shape()[0]):
        #    for j in range(self.output.get_shape()[1]):
        #        tf.summary.scalar("output_" + str(i) + "_" + str(j), self.output[i, j])

        self.merged_summaries = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter('../tensorboard_data/cnn_atari2' , self.sess.graph)
        self.train_writer.add_graph(self.sess.graph)
        #self.train_writer.add_summary(self.merged_summaries)
        #self.train_writer.add_summary(self.sess.run(summary_op), 1))

    # if none is returned, take random action
    def act(self, frameInput):
        action = None
        
        exploreOrNo = random.uniform(0,1)
        if exploreOrNo > self.epsilon: action = np.argmax(self.sess.run([self.output], feed_dict={self.input: frameInput}))
        if self.epsilon > .1: self.epsilon -= .000009

        return action

a = Agent()
a.buildGraph()
print("done")

e = Environment()
state = e.getInitialState()

#a.sess.run([a.output, a.summary_op], feed_dict={a.input: state})
#result, summary = a.sess.run([a.output, a.summary_op], feed_dict={a.input: state})
#result = a.sess.run([a.output], feed_dict={a.input: state})
#result = a.act(state)
#action = np.argmax(result)
#print(result)
#print(action)

rewards = []

done = False
frameNum = 0
while not done:
    
    frameNum += 1
    #if frameNum % 4 != 0:
        #newstate, reward, done = e.act(0)
    #else:
        #action = a.act(state)
        #if action == None: action = e.env.action_space.sample()
        #newstate, reward, done = e.act(action) 

        #rewards.append(reward)
        ##print(reward)
        ##action = np.argmax(a.act(newstate))

        
    action = a.act(state)
    if action == None: action = e.env.action_space.sample()
    newstate, reward, done = e.act(action) 

    rewards.append(reward)
    e.env.render()
    state = newstate

avg = np.average(rewards)
#print(avg)

#a.train_writer.add_summary(summaries)


#frame0 = e.env.reset()
#imsave('frame0.png', frame0)
#frame1 = e.preprocessFrame(frame0)
#imsave('frame1.png', frame1)

