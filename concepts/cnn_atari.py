import tensorflow as tf
import gym
import time
import random

# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/ 
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/ 
# https://github.com/NVlabs/GA3C/tree/master/ga3c 
# https://github.com/devsisters/DQN-tensorflow/tree/master/dqn

#THREAD_DELAY = 
NUM_ACTIONS = 1
N_STEP_RETURN = 10
GAMMA = .99
GAMMA_N = GAMMA ** N_STEP_RETURN

class Environment:
    def __init__():
        self.env = gym.make('SpaceInvaders-v0')
        
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
        def get_sample(memory, n):
            reward = 0.0
            for i in range(n):
                r += memory[i][2] * (GAMMA ** i)

            # TODO: what on earth do the '_''s do?
            state, action, _, _ = memory[0]
            _, _, _, state_ = memory[n-1]

            return state, action, self.R, state_

        # make a one-hot representation of action
        action_cats = np.zeros(NUM_ACTIONS) 
        action_cats[action] = 1

        # store experience samples
        self.memory.append((state, action_cats, reward, state_))

        self.R = (self.R + r * GAMMA_N) / GAMMA # TODO: I remember this being a thing, but I don't remember exactly what it does
        
        # if we hit terminating state (wind down/gradually clear memory)
        if state_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                state, action, reward, state_ = get_sample(self.meory, n)
                brain.train_push # TODO: where is brain coming from again?

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0 # TODO: what??????

        # only store n samples though, so clean up as needed
        if len(self.memory) >= N_STEP_RETURN:
            state, action, reward, state_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(state, action, reward, state_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)
            



class Brain:
    def __init__():
        pass
    

    def train_push(self, state, action, reward, state_):
        with self.lock_queue: 
            self.train_queue[0].append(state)
            self.train_queue[1].append(action)
            self.train_queue[2].append(reward)

            if state_ is None:
                self.train_queue[3].append(NONE_STATE) # TODO: where is this coming from?
                self.train_queue[4].append(0.0)
            else:
                self.train_queue[3].append(state_) # TODO: where is this coming from?
                self.train_queue[4].append(1.0)

                

