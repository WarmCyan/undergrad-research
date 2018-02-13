import tensorflow as tf
import gym
import time
import random

# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/ 
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/ 
# https://github.com/NVlabs/GA3C/tree/master/ga3c 

#THREAD_DELAY = 
NUM_ACTIONS = 1

class Environment:
    def __init__():
        self.env = gym.make('CartPole-v1')
        
    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def runEpisode():
        state = self.env.reset() # initial state
        while True:
            #time.sleep()
            action = self.agent.act(s)
            state_, reward, done, info = self.env.step(a) # TODO: what is info?

            if done: # terminal state
                state__ = none 

            self.agent.train(state, action, reward, state_)

            state = state_

            if done or self.stop_signal:
                break
            

class Agent:
    def __init__():
        pass

    def act(self, state):
        # as in q-learning, add a random chance for exploring
        if random.random() < epsilon:
            return random.randint(0, NUM_ACTIONS-1)
        else:
            p = brain.predict_p(state) # NOTE: don't actually think p is p here?
            return np.random.choice(NUM_ACTIONS, p=p)

    def train(self, state, action, reward, state):
        action_cats = np.zeros(NUM_ACTIONS) # TODO: what is cats???
        action_cats[a] = 1

        # store experience samples
        self.memory.append((state, action_cats, reward, state_))


class Brain:
    def __init__():
        pass
