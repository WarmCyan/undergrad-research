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

        self.seqSize = 4 # how many frames to include in a sequence
        self.frameSeq = []
        
        print("Environment initialized")

    def getInitialState(self):
        frame = self.preprocessFrame(self.env.reset())
        self.frameSeq.append(frame) # TODO: make this based off of self.seqsize
        self.frameSeq.append(frame)
        self.frameSeq.append(frame)
        self.frameSeq.append(frame)

        #state = np.dstack(frameSeq)
        state = np.dstack(self.frameSeq)
        
        return state
        #return self.preprocessFrame(self.env.reset())

    def preprocessFrame(self, frame):
        frame = resize(frame, (110,84))
        frame = frame[18:102,0:84]
        frame = rgb2grey(frame)
        #frame = np.reshape(frame, (1,84,84,1))
        return frame

    def act(self, action):
        observation, reward, done, info = self.env.step(action)
        if done: print("DONE")

        observationFrame = self.preprocessFrame(observation)

        self.frameSeq.pop(0)
        self.frameSeq.append(observationFrame)
        state = np.dstack(self.frameSeq)
        
        return state, reward, done


class Agent:
    def __init__(self):
        self.sess = tf.Session()
        self.epsilon = 1.0



        self.replayMemory = []


        
        #self.epsilon = 0.1
        print("agent initialized")


    def buildGraph(self):

        ## -- Q NETWORK --
        self.input = tf.placeholder(tf.float32, shape=(1, 84,84,4), name='input') # TODO: pretty sure that shape isn't right
        # convolutional layers


        # filter: [filter_height, filter_width, in_channels, out_channels ]
        # stride (NHWC): [batchsize, stride_height, stride_width, channels]

        # 32 filters, kernel size of 8, stride of 4
        #self.conv1 = tf.layers.conv2d(self.input, 32, 8, 4, activation=tf.nn.relu, name='conv1')
        with tf.name_scope('conv1'):
            self.w1 = tf.Variable(tf.random_normal([8, 8, 4, 32]), name='weights1')
            self.b1 = tf.Variable(tf.random_normal([32]), name='bias1')
            self.conv1 = tf.nn.conv2d(self.input, self.w1, [1, 4, 4, 1], "VALID", name='conv1') 
            self.conv1_relu = tf.nn.relu(tf.nn.bias_add(self.conv1, self.b1))
        
        # 64 filters, kernel size of 4, stride of 2
        #self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, 2, activation=tf.nn.relu, name='conv2')
        with tf.name_scope('conv2'):
            self.w2 = tf.Variable(tf.random_normal([4, 4, 32, 64]), name='weights2')
            self.b2 = tf.Variable(tf.random_normal([64]), name='bias2')
            self.conv2 = tf.nn.conv2d(self.conv1_relu, self.w2, [1, 2, 2, 1], "VALID", name='conv2') 
            self.conv2_relu = tf.nn.relu(tf.nn.bias_add(self.conv2, self.b2))
            
        # 64 filters, kernel size of 3, stride of 1
        #self.conv3 = tf.layers.conv2d(self.conv2_relu, 64, 3, 1, activation=tf.nn.relu, name='conv3')
        #self.conv3_out = tf.reshape(self.conv3, [-1, 3136], name='conv3_flatten')
        with tf.name_scope('conv3'):
            self.w3 = tf.Variable(tf.random_normal([3, 3, 64, 64]), name='weights3')
            self.b3 = tf.Variable(tf.random_normal([64]), name='bias3')
            self.conv3 = tf.nn.conv2d(self.conv2_relu, self.w3, [1, 1, 1, 1], "VALID", name='conv3') 
            self.conv3_relu = tf.nn.relu(tf.nn.bias_add(self.conv3, self.b3))
            
        self.conv3_out = tf.reshape(self.conv3_relu, [-1, 3136], name='conv3_flatten')

        # fully conected layer
        with tf.name_scope('fully_connected'):
            self.fc_w = tf.Variable(tf.random_normal([3136, 512]), name='fc_weights') 
            self.fc_b = tf.Variable(tf.random_normal([512]), name='fc_biases') # fully connected biases

            self.fc_out = tf.nn.relu_layer(self.conv3_out, self.fc_w, self.fc_b, name='fc_out')

        # output layer (for space invaders, 6 possible outputs)
        with tf.name_scope('output'):
            self.out_w = tf.Variable(tf.random_normal([512, 6]), name='out_weights')
            self.out_b = tf.Variable(tf.random_normal([6]), name='out_biases')

            self.output = tf.matmul(self.fc_out, self.out_w) + self.out_b


        ## -- Q-hat NETWORK -- (target network)
            
        self.t_input = tf.placeholder(tf.float32, shape=(1, 84,84,4), name='t_input') # TODO: pretty sure that shape isn't right
        # convolutional layers


        # filter: [filter_height, filter_width, in_channels, out_channels ]
        # stride (NHWC): [batchsize, stride_height, stride_width, channels]

        # 32 filters, kernel size of 8, stride of 4
        #self.conv1 = tf.layers.conv2d(self.input, 32, 8, 4, activation=tf.nn.relu, name='conv1')
        with tf.name_scope('t_conv1'):
            self.t_w1 = tf.Variable(tf.random_normal([8, 8, 4, 32]), name='t_weights1')
            self.t_b1 = tf.Variable(tf.random_normal([32]), name='t_bias1')
            self.t_conv1 = tf.nn.conv2d(self.t_input, self.t_w1, [1, 4, 4, 1], "VALID", name='t_conv1') 
            self.t_conv1_relu = tf.nn.relu(tf.nn.bias_add(self.t_conv1, self.t_b1))
        
        # 64 filters, kernel size of 4, stride of 2
        #self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, 2, activation=tf.nn.relu, name='conv2')
        with tf.name_scope('t_conv2'):
            self.t_w2 = tf.Variable(tf.random_normal([4, 4, 32, 64]), name='t_weights2')
            self.t_b2 = tf.Variable(tf.random_normal([64]), name='t_bias2')
            self.t_conv2 = tf.nn.conv2d(self.t_conv1_relu, self.t_w2, [1, 2, 2, 1], "VALID", name='t_conv2') 
            self.t_conv2_relu = tf.nn.relu(tf.nn.bias_add(self.t_conv2, self.t_b2))
            
        # 64 filters, kernel size of 3, stride of 1
        #self.conv3 = tf.layers.conv2d(self.conv2_relu, 64, 3, 1, activation=tf.nn.relu, name='conv3')
        #self.conv3_out = tf.reshape(self.conv3, [-1, 3136], name='conv3_flatten')
        with tf.name_scope('t_conv3'):
            self.t_w3 = tf.Variable(tf.random_normal([3, 3, 64, 64]), name='t_weights3')
            self.t_b3 = tf.Variable(tf.random_normal([64]), name='t_bias3')
            self.t_conv3 = tf.nn.conv2d(self.t_conv2_relu, self.t_w3, [1, 1, 1, 1], "VALID", name='t_conv3') 
            self.t_conv3_relu = tf.nn.relu(tf.nn.bias_add(self.t_conv3, self.t_b3))
            
        self.t_conv3_out = tf.reshape(self.t_conv3_relu, [-1, 3136], name='t_conv3_flatten')

        # fully conected layer
        with tf.name_scope('t_fully_connected'):
            self.t_fc_w = tf.Variable(tf.random_normal([3136, 512]), name='t_fc_weights') 
            self.t_fc_b = tf.Variable(tf.random_normal([512]), name='t_fc_biases') # fully connected biases

            self.t_fc_out = tf.nn.relu_layer(self.t_conv3_out, self.t_fc_w, self.t_fc_b, name='t_fc_out')

        # output layer (for space invaders, 6 possible outputs)
        with tf.name_scope('t_output'):
            self.t_out_w = tf.Variable(tf.random_normal([512, 6]), name='t_out_weights')
            self.t_out_b = tf.Variable(tf.random_normal([6]), name='t_out_biases')

            self.t_output = tf.matmul(self.t_fc_out, self.t_out_w) + self.t_out_b



            
        #self.t_input = tf.placeholder(tf.float32, shape=(1, 84,84,4)) # TODO: pretty sure that shape isn't right
        # convolutional layers

        # 32 filters, kernel size of 8, stride of 4
        #self.t_conv1 = tf.layers.conv2d(self.t_input, 32, 8, 4, activation=tf.nn.relu, name='t_conv1')
        
        # 64 filters, kernel size of 4, stride of 2
        #self.t_conv2 = tf.layers.conv2d(self.t_conv1, 64, 4, 2, activation=tf.nn.relu, name='t_conv2')
        
        # 64 filters, kernel size of 3, stride of 1
        #self.t_conv3 = tf.layers.conv2d(self.t_conv2, 64, 3, 1, activation=tf.nn.relu, name='t_conv3')
        #self.t_conv3_out = tf.reshape(self.t_conv3, [-1, 3136], name='t_conv3_flatten')


        # fully conected layer
        #with tf.name_scope('t_fully_connected'):
            #self.t_fc_w = tf.Variable(tf.random_normal([3136, 512]), name='t_fc_weights') 
            #self.t_fc_b = tf.Variable(tf.random_normal([512]), name='t_fc_biases') # fully connected biases

            #self.t_fc_out = tf.nn.relu_layer(self.t_conv3_out, self.t_fc_w, self.t_fc_b, name='t_fc_out')
            

        # output layer (for space invaders, 6 possible outputs)
        #with tf.name_scope('t_output'):
            #self.t_out_w = tf.Variable(tf.random_normal([512, 6]), name='t_out_weights')
            #self.t_out_b = tf.Variable(tf.random_normal([6]), name='t_out_biases')

            #self.t_output = tf.matmul(self.t_fc_out, self.t_out_w) + self.t_out_b
            


        ## -- RESET OPERATIONS --
        #with tf.name_scope('reset'):
        self.reset_conv1_w = tf.assign(self.t_w1, self.w1)
        self.reset_conv1_b = tf.assign(self.t_b1, self.b1)
        self.reset_conv2_w = tf.assign(self.t_w2, self.w2)
        self.reset_conv2_b = tf.assign(self.t_b2, self.b2)
        self.reset_conv3_w = tf.assign(self.t_w3, self.w3)
        self.reset_conv3_b = tf.assign(self.t_b3, self.b3)
        self.reset_fc_w = tf.assign(self.t_fc_w, self.fc_w)
        self.reset_fc_b = tf.assign(self.t_fc_b, self.fc_b)
        self.reset_out_w = tf.assign(self.t_out_w, self.out_w)
        self.reset_out_b = tf.assign(self.t_out_b, self.out_b)

        self.reset_op = tf.group(self.reset_conv1_w, self.reset_conv1_b, self.reset_conv2_w, self.reset_conv3_w, self.reset_conv3_b, self.reset_fc_w, self.reset_fc_b, self.reset_out_w, self.reset_out_b)
            

        self.merged_summaries = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())

        #print("fc:")
        #print(self.fc_w.eval(self.sess))
        #print("t_fc:")
        #print(self.t_fc_w.eval(self.sess))
        
        self.sess.run([self.reset_op])
        #print("t_fc after reset:")
        #print(self.t_fc_w.eval(self.sess))
        
        self.train_writer = tf.summary.FileWriter('../tensorboard_data/cnn_atari3' , self.sess.graph)
        self.train_writer.add_graph(self.sess.graph)


    # if none is returned, take random action
    def act(self, state):
        action = None
        
        exploreOrNo = random.uniform(0,1)
        if exploreOrNo > self.epsilon: action = np.argmax(self.sess.run([self.output], feed_dict={self.input: [state]}))
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

exit()

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

