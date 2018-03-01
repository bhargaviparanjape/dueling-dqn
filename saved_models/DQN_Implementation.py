#!/usr/bin/env python
from __future__ import print_function
import os
import gym
import sys
import copy
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import random
import math
from itertools import count
import torch.optim as optim
import torch

class baseQNetwork(nn.Module):
    def __init__(self, env):
        super(baseQNetwork, self).__init__()
        self.env = env
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
    
    def forward(self):
        pass

    def load_model(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save_model(self, file_name):
        torch.save(self.state_dict(), file_name)
    
class linearQNetwork(baseQNetwork):
    def __init__(self, env):
        super(linearQNetwork, self).__init__(env)
        self.linear = nn.Linear(self.input_size, self.output_size)
    
    def forward(self, input_state):
        output = self.linear(input_state)
        return output
    
class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in 
        # the memory are replaced. A simple (if not the most efficient) was to implement the memory 
        # is as a list of transitions. 
        pass

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, 
        # next state, terminal flag tuples. You will feed this to your model to train.
        pass

    def append(self, transition):
        # Appends transition to the memory.     
        pass

class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env_name, network='linear', render = False, gamma=1, num_episodes = 5000): 
        
        self.env = gym.make(env_name)
        self.num_episodes = num_episodes
        self.num_iter = 0
        self.gamma = gamma
        
        if network=='linear':
            self.model = linearQNetwork(self.env)

        #if torch.cuda.is_available():
        #    self.model.cuda()
        
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def epsilon_greedy_policy(self, qvalues, eps_start = 0.5, eps_end = 0.05, eps_decay = 1e6):
        # Epsilon greedy probabilities to sample from.
        
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * self.num_iter / eps_decay)
        self.num_iter += 1
        if sample > eps_threshold:
            print 
            action = qvalues.data.max(0)[1]
        else:
            action = torch.LongTensor([random.randrange(self.model.output_size)])
        return action[0]

    def greedy_policy(self, qvalues):
        # Greedy policy for test time.
        
        action = qvalues.data.max(0)[1]
        return action[0]

    def train(self, exp_replay=False, model_file=None):
        
        print ('*'*80)
        print ('Training.......')
        print ('*'*80)
        for episode in range(1,self.num_episodes+1):
            cur_state = self.env.reset()
            for t in count():
                state_var = Variable(torch.FloatTensor(cur_state))
                #if torch.cuda.is_available:
                #    state_var.cuda()
                qvalues = self.model(state_var)
                action = self.epsilon_greedy_policy(qvalues)
                prediction = qvalues.max(0)[0].view(1, 1)
                next_state, reward, done, _ = self.env.step(action)
                
                next_state_var = Variable(torch.FloatTensor(next_state),volatile=True)
                #if torch.cuda.is_available:
                #    next_state_var.cuda()
                nqvalues = self.model(next_state_var)
                target = reward + self.gamma* nqvalues.max(0)[0].view(1, 1)
                
                loss = self.loss_function(prediction, target)
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                cur_state = next_state
                if done:
                    print('Episode %d : Loss = %f, Steps = %d' %(episode,loss,t))
                    break
        print ('*'*80)
        print ('Training Complete')
        print ('*'*80)
        return

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards 
        # for the 100 episodes. Here you need to interact with the environment, irrespective of whether
        # you are using a memory.
        pass
        
    def burn_in_memory():
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='MountainCar-v0')
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--load', dest='model_load', type=str, default=None)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env

    # You want to create an instance of the DQN_Agent class here, and then train / test it
    usenetwork = 'linear'
    agent = DQN_Agent(environment_name, network=usenetwork)
    if args.train:
        agent.train()
        model_save = os.path.join('saved_models', usenetwork+'_'+environment_name)
        agent.model.save_model(model_save)
    elif os.path.exists(model_load):
        model_load = args.model_load
        agent.model = agent.model.load_model(model_load)

if __name__ == '__main__':
    main(sys.argv)