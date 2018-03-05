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
from collections import deque
from collections import namedtuple
import copy

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
    
class mlpQNetwork(baseQNetwork):
    def __init__(self, env, hidden1=16, hidden2=16):
        super(mlpQNetwork, self).__init__(env)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.linear1 = nn.Linear(self.input_size, self.hidden1)
        self.linear2 = nn.Linear(self.hidden1, self.hidden2)
        self.linear3 = nn.Linear(self.hidden2, self.output_size)
    
    def forward(self, input_state):
        hidden1 = self.linear1(input_state)
        hidden1 = nn.functional.relu(hidden1)
        hidden2 = self.linear2(hidden1)
        hidden2 = nn.functional.relu(hidden2)
        output = self.linear3(hidden2)
        return output
    
class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in 
        # the memory are replaced. A simple (if not the most efficient) was to implement the memory 
        # is as a list of transitions.
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = deque(maxlen=self.memory_size)

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, 
        # next state, terminal flag tuples. You will feed this to your model to train.
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)

class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env_name, network='linear', render = False, gamma=1, num_episodes = 5000, 
                 use_cuda=False, transfer_every=1): 
        
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.num_episodes = num_episodes
        self.num_iter = 0
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.transfer_every = transfer_every
        self.loss_function = nn.MSELoss()
        
        if network=='linear':
            self.model = linearQNetwork(self.env)
        elif network=='mlp':
            self.model = mlpQNetwork(self.env)
        
        #self.target_model = copy.deepcopy(self.model)
        if self.use_cuda and torch.cuda.is_available():
            self.model.cuda()
            #self.target_model.cuda()
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.transition = namedtuple('transition', ('state', 'action', 'reward', 
                                                    'next_state', 'is_terminal'))
    
    def loss_function_alt(self, pred, target):
        return torch.sum((pred - target)**2) / pred.data.nelement()

    def epsilon_greedy_policy(self, qvalues, eps_start = 0.9, eps_end = 0.05, eps_decay = 1e6):
        # Epsilon greedy probabilities to sample from.
        
        sample = random.random()
        eps_threshold = eps_start + (eps_end - eps_start) * (self.num_iter / eps_decay)
        self.num_iter += 1
        if sample > eps_threshold:
            action = qvalues.data.max(0)[1]
        else:
            action = torch.LongTensor([random.randrange(self.model.output_size)])
        return action[0]

    def greedy_policy(self, qvalues):
        # Greedy policy for test time.
        
        action = qvalues.data.max(0)[1]
        return action[0]
    
    def save_criteria(self, t_counter):
        # Greedy policy for test time.
        
        if self.env_name == 'CartPole-v0':
            return min(t_counter)>195
        elif self.env_name == 'MountainCar-v0':
            return False

    def train(self, exp_replay=False, model_save=None, verbose = False, eval_every = 10000):
        
        print ('*'*80)
        print ('Training.......')
        print ('*'*80)
        t_counter = deque(maxlen=10)
        self.update_counter = 1
        for episode in range(1,self.num_episodes+1):
            cur_state = self.env.reset()
            for t in count():
                state_var = Variable(torch.FloatTensor(cur_state))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()
                qvalues = self.model(state_var)
                action = self.epsilon_greedy_policy(qvalues)
                prediction = qvalues.max(0)[0]
                next_state, reward, done, _ = self.env.step(action)
                
                #target = Variable(torch.zeros(1))
                next_state_var = Variable(torch.FloatTensor(next_state), volatile=True)
                if self.use_cuda and torch.cuda.is_available():
                    next_state_var = next_state_var.cuda()
                nqvalues = self.model(next_state_var)
                nqvalues = nqvalues.max(0)[0]
                nqvalues.volatile = False
                
                target = reward + (1-done)* self.gamma* nqvalues
                #target[0] = reward + (1-done)* self.gamma* nqvalues
                
                loss = self.loss_function(prediction, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.update_counter += 1
                cur_state = next_state
                
                #if self.update_counter% self.transfer_every == 0:
                #    self.target_model.load_state_dict(self.model.state_dict())
                
                if self.update_counter% eval_every == 0:
                    avg_reward = self.test(20, True, self.update_counter/10000, False)
                    t_counter.append(avg_reward)
                    if self.save_criteria(t_counter):
                        print ('*'*80)
                        print ('Saving Best Model')
                        print ('*'*80)
                        self.model.save_model(model_save)
                if done:
                    if verbose and episode % 100 == 0:
                        print('Episode %07d : Steps = %03d, Loss = %.2f' %(episode,t,loss))
                    break
        print ('*'*80)
        print ('Training Complete')
        print ('*'*80)
        return

    def test(self, num_episodes = 100, eval = False, num_updates = 0,verbose = True):
        print ('*'*80)
        print ('Testing Performance.......')
        print ('*'*80)
        episode_reward_list = []
        for episode in range(1,num_episodes+1):
            cur_state = self.test_env.reset()
            episode_reward = 0
            for t in count():
                state_var = Variable(torch.FloatTensor(cur_state))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()
                qvalues = self.model(state_var)
                
                if eval:
                    action = self.epsilon_greedy_policy(qvalues, eps_start = 0.05, eps_end = 0.05)
                else:
                    action = self.greedy_policy(qvalues)

                next_state, reward, done, _ = self.test_env.step(action)
                cur_state = next_state
                episode_reward += reward
                if done:
                    if verbose:
                        print('Episode %07d : Steps = %03d, Reward = %.2f' %(episode,t,episode_reward))
                    episode_reward_list.append(episode_reward)
                    break
        episode_reward_list = np.array(episode_reward_list)
        print ('*'*80)
        print ('Num_Updates(*10000): %03d => Cumulative Reward: Mean= %f, Std= %f'  %(num_updates,
                episode_reward_list.mean(),episode_reward_list.std()))
        print ('*'*80)
        return episode_reward_list.mean()
        
    def burn_in_memory():
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        self.replay = Replay_Memory()
        state = self.env.reset()
        while len(self.replay.memory) < self.replay.burnin:
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.replay.append(self.transition(state, action, next_state, reward, done))
            state = next_state

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--network', dest='network', type=str, default='mlp')
    parser.add_argument('--epochs', dest='num_episodes', type=int, default=80000)
    parser.add_argument('--load', dest='model_load', type=str, default=None)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env

    # You want to create an instance of the DQN_Agent class here, and then train / test it
    usenetwork = args.network
    agent = DQN_Agent(environment_name, network=usenetwork, num_episodes = args.num_episodes)
    if args.train:
        model_save = os.path.join('saved_models',usenetwork+'_'+environment_name)
        agent.train(model_save = model_save, verbose = True)
    elif os.path.exists(args.model_load):
        model_load = args.model_load
        agent.model.load_model(model_load)
        agent.test()

if __name__ == '__main__':
    main(sys.argv)

