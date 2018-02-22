import tensorflow as tf
import gym
import numpy as np

from skimage.transform import resize
from skimage.color import rgb2grey
from skimage.io import imsave

import random

import multiprocessing
import threading


# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/ 
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/ 
# https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2 


# returns a set of operations to set all weights of destination scope to values of weights from source scope
def getWeightChangeOps(scopeSrc, scopeDest):
    srcVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scopeSrc)
    destVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scopeDest)

    assignOps = []
    for srcVar, destVar in zip(srcVars, destVars):
        assignOps.append(tf.assign(srcVar, destVar))

    return assignOps

# calculates discounted return TODO: figure out why this actually works?
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Manager:

    def __init__(self):

        self.learningRate = .0001
        
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        self.globalNetwork = Network('global', self.optimizer)
        self.numWorkers = multiprocessing.cpu_count()
        self.workers = []

        for i in range(self.numWorkers):
            self.workers.append(Worker("worker_" + str(i), self.optimizer))


    def run(self):
        with tf.Session() as sess:
            coordinator = tf.train.Coordinator()
            sess.run(tf.global_variables_initializer())

            # create worker threads
            worker_threads = []
            for worker in self.workers:
                worker_function = lambda: worker.work(sess, coordinator)
                t = threading.Thread(target=worker_function)
                t.start()
                worker_threads.append(t)

            coordinator.join(worker_threads)
            

class Worker:

    def __init__(self, name, optimizer):
        self.name = name
        #self.optimizer = optimizer
        
        self.network = Network(self.name, optimizer)
        self.network.buildGraph()

        self.resetWeights = getWeightChangeOps("global", self.name)

        self.t_max = 4000
        self.GAMMA = .99

    def train(self, history):
        history = np.array(history)
        states = history[:,0]
        actions = history[:,1]
        rewards = history[:,2]
        states_next = history[:,3]
        values = history[:,4]

        discountedRewards = discount(rewards, self.GAMMA)

        # NOTE: values[1:] = the next state, values[:-1] = the previous state
        # A = Q - V(s)
        # Q = r + yV(s')
        # A = r + yV(s') - V(S)
        advantages = rewards + self.GAMMA*values[1:] - self.values[:-1]

        # TODO: supposedly we have to discount advantages, I don't know if that is correct or not (shouldn't we just use discounted rewards?)



        # apply gradients to global network
        p_loss, v_loss = sess.run([self.network.policy_loss, self.network.value_loss, self.network.apply_gradients], feed_dict={self.network.input: states, self.network.actions: actions, self.network.target_v: discountedRewards, self.network.advantages: advantages})

        print("Policy loss:",p_loss,"Value loss:",v_loss)


        

        
        
        
    
    def work(self, session, coordinator):
        while not coordinator.should_stop():

            # reset ops
            session.run(self.resetWeights)

            # get an environment instance
            self.env = Environment()

            history = []

            t = 0

            # get state s_t
            s_t = env.getInitialState()
            terminal = False

            # repeat until terminal state
            while not terminal and t < self.t_max:
                # perform a_t according to policY9a_t|s_t; theta_)
                policyVec, v = session.run([self.network.policy_out, self.network.value_out], feed_dict={self.network.input: [state]})
                a_t = np.argmax(policyVec)

                # receive reward r_t and new state s_{t+1}
                #a_t = a.act(s_t)
                s_t1, r_t, terminal = self.env.act(a_t)

                self.history.append([s_t, a_t, r_t, s_t1, v])

                t += 1

            R = 0
            #if not terminal: R = 


            for i in range(0, t):
                transition = history[i - t_start]
                s_i = transition[0]
                a_i = transition[1]
                r_i = transition[2]

                R = r_i + self.GAMMA*R

            
                
                
                
            
            
    

    
    


class Network:
    def __init__(self, scope, optimizer):
        self.scope = scope
        self.optimizer = optimizer
        
        

    def buildGraph(self):
        self.input = tf.placeholder(tf.float32, shape=(1,84,84,4), name='input') # TODO: pretty sure that shape isn't right
        
        # 16 filters, kernel size of 8, stride of 4
        with tf.name_scope('conv1'):
            self.w1 = tf.Variable(tf.random_normal([8, 8, 4, 16]), name='weights1')
            self.b1 = tf.Variable(tf.random_normal([16]), name='bias1')
            self.conv1 = tf.nn.conv2d(self.input, self.w1, [1, 4, 4, 1], "VALID", name='conv1') 
            self.conv1_relu = tf.nn.relu(tf.nn.bias_add(self.conv1, self.b1))
            
        # 32 filters, kernel size of 4, stride of 2
        with tf.name_scope('conv2'):
            self.w2 = tf.Variable(tf.random_normal([4, 4, 16, 32]), name='weights2')
            self.b2 = tf.Variable(tf.random_normal([32]), name='bias2')
            self.conv2 = tf.nn.conv2d(self.conv1_relu, self.w2, [1, 2, 2, 1], "VALID", name='conv2') 
            self.conv2_relu = tf.nn.relu(tf.nn.bias_add(self.conv2, self.b2))

            # flattened size is 9*9*32 = 2592
            self.conv2_out = tf.reshape(self.conv2_relu, [-1, 2592], name='conv2_flatten') 
            

        # fully connected layer with 256 hidden units
        with tf.name_scope('fully_connected'):
            self.fc_w = tf.Variable(tf.random_normal([2592, 256]), name='fc_weights') 
            self.fc_b = tf.Variable(tf.random_normal([256]), name='fc_biases') # fully connected biases

            self.fc_out = tf.nn.relu_layer(self.conv2_out, self.fc_w, self.fc_b, name='fc_out')

        # policy output, policy = distribution of probabilities over actions, use softmax to choose highest probability action
        with tf.name_scope('policy'):
            self.policy_w = tf.Variable(tf.random_normal([256, 6]), name='policy_w')
            
            # TODO: do we need biases as well?
            self.policy_out = tf.nn.softmax(tf.matmul(self.fc_out, self.policy_w))
            
            ## NOTE: used for gradient calculations
            #self.policy_log_prob = tf.log(self.policy_out)

        # Only a SINGLE output, just a single linear value
        with tf.name_scope('value'):
            self.value_w = tf.Variable(tf.random_normal([256, 1]), name='value_w')

            # TODO: do we need a bias for this? (edit: I'm pretty sure since it's a single linear value, there's no point in having a bias value?)

            self.value_out = tf.matmul(self.fc_out, self.value_w)



        # policy gradient calculation
        #self.R = tf.placeholder(tf.float32, shape=(1), name='reward_input')

        #self.entropy = tf.reduce_sum(self.policy_out * self.policy_log_prob, name='entropy')
        
        #with tf.name_scope('advantage'):
            #self.A = self.R - self.value_out

        #with tf.name_scope("policy_loss"):
            #self.policy_loss = self.policy_log_prob*self.A # NOTE: the graph of this doesn't look right...the mul term doesn't go into the gradient at all, is that correct?
            #self.dtheta_ = tf.gradients(self.policy_gradient_term, [self.policy_w, self.fc_w, self.w2, self.w1]) # TODO: no idea if this is correct at all

        # TODO: add an entropy term to the gradient


        #with tf.name_scope('value_loss'):
            #self.value_loss = tf.square(self.A)
            #self.dtheta_v_ = tf.gradients(self.A, [self.value_w]) # TODO: does this still apply to all weights or only value weights?
            

        #with tf.name_scope("objective"):
            #self.full_objective = self.policy_loss + self.value_loss_weight*self.value_loss + self.entropy*self.BETA
            #localvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # TODO: eventually need a scope variable passed in as well I think? (once go to multithreading)
            #self.gradients = tf.gradients(self.full_objective, localvars)

            # TODO: still doesn't look right, the full_objective stuff doesn't actually lead into the gradients variable??

        if self.scope != 'global':
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
            
            self.actions_onehot = tf.one_hot(self.actions, 6, dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(self.policy_out * self.actions_onehot, [1])
            
            # losses
            # NOTE: .5's seem arbitrary, these should be set as hyperparameters
            self.value_loss = .5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value_out, [-1])))
            self.entropy = -tf.reduce_sum(self.policy_out * self.actions_onehot, [1])
            self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
            self.loss = .5 * self.value_loss + self.policy_loss - self.entropy * .01 # NOTE: .01 should also be a hyperparameter

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 40.0) # TODO: where is 40 coming from???
            
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_gradients = self.optimizer.apply_gradients(zip(self.clipped_gradients, global_vars))
            
            
            


        if self.scope == 'global':
            self.merged_summaries = tf.summary.merge_all()
            self.sess.run(tf.global_variables_initializer())
            
            self.train_writer = tf.summary.FileWriter('../tensorboard_data/a3c' , self.sess.graph)
            self.train_writer.add_graph(self.sess.graph)




    




class Environment:
    def __init__(self):
        self.env = gym.make("SpaceInvaders-v0")
        
        self.seqSize = 4
        self.frameSeq = []
        
        print("Environment initialized")

    def getInitialState(self):
        frame = self.preprocessFrame(self.env.reset())
        self.frameSeq.append(frame) # TODO: make this based off of self.seqsize
        self.frameSeq.append(frame)
        self.frameSeq.append(frame)
        self.frameSeq.append(frame)

        state = np.dstack(self.frameSeq)
        
        return state

    def preprocessFrame(self, frame):
        frame = resize(frame, (110,84))
        frame = frame[18:102,0:84]
        frame = rgb2grey(frame)
        return frame


    def act(self, action):
        observation, reward, terminal, info = self.env.step(action)
        if terminal: print("TERMINAL STATE REACHED")

        observationFrame = self.preprocessFrame(observation)
        
        self.frameSeq.pop(0)
        self.frameSeq.append(observationFrame)
        state = np.dstack(self.frameSeq)
        
        return state, reward, terminal




m = Manager()
m.run()




'''
class Agent:
    def __init__(self, numPossibleActions):
        self.sess = tf.Session()

        self.numPossibleActions = numPossibleActions


        self.BETA = .01 # NOTE: this is the entropy regularization term
        self.value_loss_weight = .4

        self.epsilon = 1.0
        self.epsilon_minimum = .1
        self.epsilon_time = 4000000 # number of frames before fully annealed
        self.epsilon_anneal_rate = (self.epsilon - self.epsilon_minimum) / self.epsilon_time

        print("Agent initialized")


    def buildGraph(self):
        
        # NOTE: same preprocessing as in DQN paper

        self.input = tf.placeholder(tf.float32, shape=(1,84,84,4), name='input') # TODO: pretty sure that shape isn't right
        
        # 16 filters, kernel size of 8, stride of 4
        with tf.name_scope('conv1'):
            self.w1 = tf.Variable(tf.random_normal([8, 8, 4, 16]), name='weights1')
            self.b1 = tf.Variable(tf.random_normal([16]), name='bias1')
            self.conv1 = tf.nn.conv2d(self.input, self.w1, [1, 4, 4, 1], "VALID", name='conv1') 
            self.conv1_relu = tf.nn.relu(tf.nn.bias_add(self.conv1, self.b1))
            
        # 32 filters, kernel size of 4, stride of 2
        with tf.name_scope('conv2'):
            self.w2 = tf.Variable(tf.random_normal([4, 4, 16, 32]), name='weights2')
            self.b2 = tf.Variable(tf.random_normal([32]), name='bias2')
            self.conv2 = tf.nn.conv2d(self.conv1_relu, self.w2, [1, 2, 2, 1], "VALID", name='conv2') 
            self.conv2_relu = tf.nn.relu(tf.nn.bias_add(self.conv2, self.b2))

            # flattened size is 9*9*32 = 2592
            self.conv2_out = tf.reshape(self.conv2_relu, [-1, 2592], name='conv2_flatten') 
            

        # fully connected layer with 256 hidden units
        with tf.name_scope('fully_connected'):
            self.fc_w = tf.Variable(tf.random_normal([2592, 256]), name='fc_weights') 
            self.fc_b = tf.Variable(tf.random_normal([256]), name='fc_biases') # fully connected biases

            self.fc_out = tf.nn.relu_layer(self.conv2_out, self.fc_w, self.fc_b, name='fc_out')

        # policy output, policy = distribution of probabilities over actions, use softmax to choose highest probability action
        with tf.name_scope('policy'):
            self.policy_w = tf.Variable(tf.random_normal([256, self.numPossibleActions]), name='policy_w')
            
            # TODO: do we need biases as well?
            self.policy_out = tf.nn.softmax(tf.matmul(self.fc_out, self.policy_w))
            
            # NOTE: used for gradient calculations
            self.policy_log_prob = tf.log(self.policy_out)

        # Only a SINGLE output, just a single linear value
        with tf.name_scope('value'):
            self.value_w = tf.Variable(tf.random_normal([256, 1]), name='value_w')

            # TODO: do we need a bias for this? (edit: I'm pretty sure since it's a single linear value, there's no point in having a bias value?)

            self.value_out = tf.matmul(self.fc_out, self.value_w)



        # policy gradient calculation
        self.R = tf.placeholder(tf.float32, shape=(1), name='reward_input')

        self.entropy = tf.reduce_sum(self.policy_out * self.policy_log_prob, name='entropy')
        
        with tf.name_scope('advantage'):
            self.A = self.R - self.value_out

        with tf.name_scope("policy_loss"):
            self.policy_loss = self.policy_log_prob*self.A # NOTE: the graph of this doesn't look right...the mul term doesn't go into the gradient at all, is that correct?
            #self.dtheta_ = tf.gradients(self.policy_gradient_term, [self.policy_w, self.fc_w, self.w2, self.w1]) # TODO: no idea if this is correct at all

        # TODO: add an entropy term to the gradient


        with tf.name_scope('value_loss'):
            self.value_loss = tf.square(self.A)
            #self.dtheta_v_ = tf.gradients(self.A, [self.value_w]) # TODO: does this still apply to all weights or only value weights?
            

        with tf.name_scope("objective"):
            self.full_objective = self.policy_loss + self.value_loss_weight*self.value_loss + self.entropy*self.BETA
            localvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # TODO: eventually need a scope variable passed in as well I think? (once go to multithreading)
            self.gradients = tf.gradients(self.full_objective, localvars)

            # TODO: still doesn't look right, the full_objective stuff doesn't actually lead into the gradients variable??




        self.merged_summaries = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        
        self.train_writer = tf.summary.FileWriter('../tensorboard_data/a3c' , self.sess.graph)
        self.train_writer.add_graph(self.sess.graph)



    def act(self, state):
        action = None
        
        #exploreOrNo = random.uniform(0, 1)
        #if exploreOrNo > self.epsilon: action = I

        actionVec = self.sess.run([self.policy_out], feed_dict={self.input: [state]})

        return np.argmax(actionVec)

        #if self.epsilon > self.epsilon_minimum: self.epsilon -= self.epsilon_anneal_rate


agent = Agent(6)
agent.buildGraph()

e = Environment()
#in_0 = e.getInitialState()
#action = a.act(in_0)
#print(action)

GAMMA = .99
T_MAX = 40000 # TODO: no idea what this actually was in study, find out (num episodes?) edit: no, this is not number of episode, as it also increases each action
t_max = 4000 # TODO: also no idea what this should be (maximum frames in a game)


history = []

T = 0

# initialize thread step counter t <- 1
t = 1

'''
'''
# repeat until T > T_MAX
for T in range(T_MAX):
    
    # reset gradients dtheta <- 0 and dtheta_v <- 0
    dtheta = 0
    dtheta_v = 0

    # synchronize thread-specific parameters theta_ = theta and theta_v_ <- theta_v # NOTE: this is resetting the local thread graph to the global graph

    # t_start = t
    t_start = t

    # get state s_t
    s_t = e.getInitialState()
    terminal = False

    # repeat until terminal state or t-t_start == t_max
    while not terminal: # TODO: and t stuff

        # perform a_t according to policy(a_t|s_t;theta_)
        a_t = a.act(s_t)

        # receive reward r_t and new state s_{t+1}
        s_t1, r_t, terminal = e.act(a_t) # TODO: specify target network? That should probably go in act function in agent

        # TODO: I assume (s_t, a_t, r_t, s_t1) needs to be stored?
        history.append((s_t, a_t, r_t, s_t1))
        
        # t <- t + 1
        t += 1
        
        # T <- T + 1
        T += 1
    
    # NOTE: R = 0 (if terminal), V(s_t, theta_v_)
    R = 0
    if not terminal: R = somvaluefuncion() 

    # for i in {t - 1, ..., t_start}
    for i in range(t-1, t_start, -1):
        transition = history[i - t_start]
        s_i = transition[0]
        a_i = transition[1]
        r_i = transition[2]
        
        # R <- r_i + GAMMA*R
        R = r_i + GAMMA*R

        # Accumulate gradients wrt theta_: dtheta <- dtheta + {nabla_theta_}(log(policy(a_i|s_i;theta_)(R-V(s_i;theta_v_))

        # Accumulate gradients wrt theta_v_: dtheta_v <- dtheta_v + {partial derivative}(R-V(s_i;theta_v_))^2 / {partial derivative}theta_v_

    # perform asynchronous update of theta using dtheta and of theta_v using dtheta_v
        
'''
