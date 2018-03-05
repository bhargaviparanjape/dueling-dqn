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
from gym import wrappers
import  matplotlib.pyplot as plt
from gym.wrappers.monitoring import stats_recorder, video_recorder
import copy
import shutil
from SumTree1 import SumTree


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

## use this if you dont want to combine two MLP outputs
class duelmlpQNetwork(baseQNetwork):
    def __init__(self, env, hidden1, hidden2, hidden3, output1, output2):
        super(duelmlpQNetwork, self).__init__(env)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output1 = output1
        self.output2 = output2

        self.linear1 = nn.Linear(self.input_size, self.hidden1)
        self.linear1_bn = nn.BatchNorm1d(self.hidden1)
        self.linear2 = nn.Linear(self.hidden1, self.hidden2)
        self.linear2_bn = nn.BatchNorm1d(self.hidden2)

        self.linear3_value = nn.Linear(self.hidden2, self.hidden3)
        self.linear3_value_bn = nn.BatchNorm1d(self.hidden3)

        self.linear3_advantage = nn.Linear(self.hidden2, self.hidden3)
        self.linear3_advantage_bn = nn.BatchNorm1d(self.hidden3)
        self.output_value = nn.Linear(self.hidden3, self.output1)
        self.output_advantage = nn.Linear(self.hidden1, self.output2)

    def forward(self, input_state):
        hidden1 = self.linear1(input_state)
        # hidden1 = self.linear1_bn(hidden1)
        hidden1 = nn.functional.relu(hidden1)
        hidden2 = self.linear2(hidden1)
        # hidden2 = self.linear2_bn(hidden2)
        hidden2 = nn.functional.relu(hidden2)
        
        ## value function
        ##hidden2 goes through linear 3, hidden3, and output to give scalar
        linear3_value = self.linear3_value(hidden2)
        # linear3_value = self.linear3_value_bn(linear3_value)
        linear3_value = nn.functional.relu(linear3_value)
        output1 = self.output_value(linear3_value)

        ##action function
        ##hidden2 goes through linear 3, hidden3 and output to give action space
        linear3_advantage = self.linear3_advantage(hidden2)
        # linear3_advantage = self.linear3_advantage_bn(linear3_advantage)
        linear3_advantage = nn.functional.relu(linear3_advantage)
        output2 = self.output_advantage(linear3_advantage)

        return output1, output2


class mlpQNetwork(baseQNetwork):
    def __init__(self, env, hidden1=16, hidden2=16):
        super(mlpQNetwork, self).__init__(env)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.linear1 = nn.Linear(self.input_size, self.hidden1)
        self.linear2 = nn.Linear(self.hidden1, self.hidden2)
        self.linear1_bn = nn.BatchNorm1d(self.hidden1)
        self.linear2_bn = nn.BatchNorm1d(self.hidden2)
        self.linear3 = nn.Linear(self.hidden2, self.output_size)
    
    def forward(self, input_state):
        hidden1 = self.linear1(input_state)
        # hidden1 = self.linear1_bn(hidden1)
        hidden1 = nn.functional.relu(hidden1)
        hidden2 = self.linear2(hidden1)
        # hidden2 = self.linear2_bn(hidden2)
        hidden2 = nn.functional.relu(hidden2)
        output = self.linear3(hidden2)
        return output


class Prioritized_Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=50000):
        self.size = memory_size
        self.burn_in = burn_in
        self.tree = SumTree(self.size)
        self.eta = 0.01
        self.alpha = 0.6
        self.memory = []

    def getPriority(self, error):
        return (error + self.eta) ** self.alpha

    def sample_batch(self, batch_size=32):
        batch = []
        segment = self.tree.total()/batch_size
        for i in range(batch_size):
            leftleaf = segment * i
            rightleaf = segment * (i + 1)

            s = random.uniform(leftleaf, rightleaf)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data + (0,)))
        return batch


    # will be called on the batch update..
    def update(self, idx, error):
        p = self.getPriority(error)
        self.tree.update(idx, p)

    def append(self, transition):
        p = self.getPriority(transition[-1])
        self.tree.add(p, transition[:-1])
        if len(self.memory) < self.size:
            self.memory.append(0)


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

    def __init__(self, env_name, network='linear', render = False, gamma=1, num_episodes = 15000, 
                 use_cuda=False, transfer_every=1, lrate=0.0001): 
        
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        if render:
            self.test_env = wrappers.Monitor(self.env, os.path.join("recording", env_name, "-", network), video_callable=lambda episode_id: True, force=True)
        self.num_episodes = num_episodes
        self.num_iter = 0
        self.gamma = gamma
        self.clip = 5
        self.use_cuda = use_cuda
        self.transfer_every = transfer_every
        self.loss_function = nn.MSELoss()
        self.performance_average_reward = list()
        if network=='linear':
            self.model = linearQNetwork(self.env)
            self.target_model = linearQNetwork(self.env)
        elif network=='mlp':
            self.model = mlpQNetwork(self.env)
            self.target_model = mlpQNetwork(self.env)
        
        if self.use_cuda and torch.cuda.is_available():
            self.model.cuda()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = lrate)
        self.transition = namedtuple('transition', ('state', 'action', 'next_state', 
                                                    'reward', 'is_terminal', 'priority'))

    def epsilon_greedy_policy(self, qvalues, eps_start = 0.9, eps_end = 0.05, eps_decay = 1e4):
        # Epsilon greedy probabilities to sample from.
        sample = random.random()
        eps_threshold = eps_start + (eps_end - eps_start) * (self.num_iter / eps_decay)
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
    
    def save_criteria(self, t_counter):
        # Greedy policy for test time.
        
        if self.env_name == 'CartPole-v0':
            return min(t_counter)>195
        elif self.env_name == 'MountainCar-v0':
            return False
    
    def update_model(self, next_state, reward, qvalues, done):
        
        prediction = qvalues.max(0)[0]
        
        next_state_var = Variable(torch.FloatTensor(next_state), volatile=True)
        if self.use_cuda and torch.cuda.is_available():
            next_state_var = next_state_var.cuda()
        
        nqvalues = self.model(next_state_var)
        nqvalues = nqvalues.max(0)[0]
        nqvalues.volatile = False
        
        target = reward + (1-done)* self.gamma* nqvalues
        if self.use_cuda and torch.cuda.is_available():
            target = target.cuda()
        
        loss = self.loss_function(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss
    
    def update_model_with_replay(self, batch_size):
        
        batch = self.replay.sample_batch(batch_size)
        indexes = [o[0] for o in batch]
        ## extra code for prioritized replay (I remove the sum tree index)
        batch = [o[1] for o in batch]
        batch = self.transition(*zip(*batch))
        
        state_batch = Variable(torch.FloatTensor(batch.state))
        action_batch = Variable(torch.LongTensor(batch.action))
        
        if self.use_cuda and torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
        prediction = self.model(state_batch).gather(1, action_batch.view(batch_size, 1))
        
        target = Variable(torch.zeros(batch_size))
        next_state_batch = Variable(torch.FloatTensor(batch.next_state),volatile=True)
        if self.use_cuda and torch.cuda.is_available():
            next_state_batch = next_state_batch.cuda()
        nqvalues = self.model(next_state_batch)
        nqvalues = nqvalues.max(1)[0]
        nqvalues.volatile = False

        for i in range(batch_size):
            target[i] = batch.reward[i] + (1 - batch.is_terminal[i])* self.gamma* nqvalues[i]
        if self.use_cuda and torch.cuda.is_available():
            target = target.cuda()

        errors = torch.abs(prediction.view(-1) - target).data
        for i in range(batch_size):
            idx = indexes[i]
            self.replay.update(idx, errors[i])

        loss = self.loss_function(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss
    

    def train(self, batch_size =32, exp_replay=False, model_save=None, verbose = False,
              eval_every = 10000):
        print ('*'*80)
        print ('Training.......')
        print ('*'*80)
        t_counter = deque(maxlen=10)
        self.update_counter = 1
        stop_checkpointing = False

        self.checkpoint_directory = os.path.join("checkpoints", model_save)
        if os.path.exists(self.checkpoint_directory):
            shutil.rmtree(self.checkpoint_directory)
        os.makedirs(self.checkpoint_directory)
        
        if exp_replay:
            self.burn_in_memory()
            
        for episode in range(1,self.num_episodes+1):
            cur_state = self.env.reset()
            for t in count():
                state_var = Variable(torch.FloatTensor(cur_state))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()
                qvalues = self.model(state_var)
                action = self.epsilon_greedy_policy(qvalues, eps_start =1)
                next_state, reward, done, _ = self.env.step(action)
                next_state_variable = Variable(torch.FloatTensor(next_state))
                nqvalues = self.model(next_state_variable)
                nqvalues = nqvalues.max(0)[0]
                target = reward + (1 - done) * self.gamma * nqvalues
                priority = math.fabs(target.data[0] - qvalues.max(0)[0])
                if exp_replay:
                    self.replay.append(self.transition(cur_state, action, next_state, 
                                                       reward, done, priority))
                    loss = self.update_model_with_replay(batch_size)
                else:
                    loss = self.update_model(next_state, reward ,qvalues, done)
                    
                self.update_counter += 1
                cur_state = next_state
                
                if self.update_counter% eval_every == 0:
                    if self.update_counter % (5*eval_every) == 0 and not stop_checkpointing:
                        self.model.save_model(os.path.join(self.checkpoint_directory, "checkpoint_" + str(self.update_counter/(5*eval_every))))
                    avg_reward = self.test(num_episodes = 20, eval = True, num_updates = self.update_counter/10000, verbose = False, render = False)
                    self.performance_average_reward.append(avg_reward)
                    t_counter.append(avg_reward)
                    if self.save_criteria(t_counter):
                        print ('*'*80)
                        print ('Saving Best Model')
                        print ('*'*80)
                        stop_checkpointing = True
                        self.model.save_model(model_save)
                        self.plot_average_reward(model_save, eval_every)
                if done:
                    if verbose and episode % 100 == 0:
                        print('Episode %07d : Steps = %03d, Loss = %.2f' %(episode,t+1,loss))
                    break
        print ('*'*80)
        print ('Training Complete')
        print ('*'*80)
        return

    def test(self, num_episodes = 100, eval = False, num_updates = 0,verbose = True, render=False):
        print ('*'*80)
        print ('Testing Performance.......')
        print ('*'*80)
        episode_reward_list = []
        for episode in range(1,num_episodes+1):
            cur_state = self.test_env.reset()
            episode_reward = 0
            for t in count():
                if render and eval == False:
                    self.env.render()
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
        
    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        self.replay = Prioritized_Replay_Memory()
        state = self.env.reset()
        while len(self.replay.memory) < self.replay.burn_in:
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            priority = 1e4
            self.replay.append(self.transition(state, action, next_state, reward, done, priority))
            if done:
                state = self.env.reset()
            else:
                state = next_state

    def plot_average_reward(self, model_save, eval_every):
        plt.plot(self.performance_average_reward)
        fout = open(os.path.join(self.checkpoint_directory, "performance.txt"), "w+")
        fout.write("\n".join(self.plot_average_reward))
        fout.close()
        plt.xlabel("Number of updates ({0})".format(eval_every))
        plt.ylabel("Reward")
        plt.title("Performance Curve for {0}".format(model_save))
        plt.savefig(os.path.join(self.checkpoint_directory, "performance.png"))


class Dueling_DQN_Agent(DQN_Agent):

    def __init__(self, env_name, network='linear', render=False, gamma=1, num_episodes=15000,
                 use_cuda=False, transfer_every=1, lrate=0.0001):
        # not calling super because wtf
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        if render:
            self.test_env = wrappers.Monitor(self.env, os.path.join("recording", env_name, "-", network),
                                             video_callable=lambda episode_id: True, force=True)
        self.num_episodes = num_episodes
        self.num_iter = 0
        self.clip = 10
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.transfer_every = transfer_every
        self.loss_function = nn.MSELoss()
        self.performance_average_reward = list()
        self.model = duelmlpQNetwork(self.env, 16, 16, 16, 1, self.env.action_space.n)

        if self.use_cuda and torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        self.transition = namedtuple('transition', ('state', 'action', 'next_state',
                                                    'reward', 'is_terminal', 'priority'))

    def update_model(self, next_state, reward, qvalues, done):

        prediction = qvalues.max(0)[0]

        next_state_var = Variable(torch.FloatTensor(next_state), volatile=True)
        if self.use_cuda and torch.cuda.is_available():
            next_state_var = next_state_var.cuda()

        vvalues_, avalues_ = self.model(next_state_var)
        nqvalues = vvalues_.repeat(self.env.action_space.n) + (avalues_ - torch.mean(avalues_).repeat(self.env.action_space.n))
        nqvalues = nqvalues.max(0)[0]
        nqvalues.volatile = False

        target = reward + (1 - done) * self.gamma * nqvalues
        if self.use_cuda and torch.cuda.is_available():
            target = target.cuda()

        loss = self.loss_function(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss

    ## not updated
    def update_model_with_replay(self, batch_size):

        batch = self.replay.sample_batch(batch_size)
        indexes = [o[0] for o in batch]
        ## extra code for prioritized replay (I remove the sum tree index)
        batch = [o[1] for o in batch]
        batch = self.transition(*zip(*batch))

        state_batch = Variable(torch.FloatTensor(batch.state))
        action_batch = Variable(torch.LongTensor(batch.action))

        if self.use_cuda and torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
        vvalues_,avalues_  = self.model(state_batch)
        prediction = vvalues_.repeat(1, self.env.action_space.n) + (avalues_ - torch.mean(avalues_).repeat(1, self.env.action_space.n))
        prediction = prediction.gather(1, action_batch.view(batch_size, 1))

        target = Variable(torch.zeros(batch_size))
        next_state_batch = Variable(torch.FloatTensor(batch.next_state), volatile=True)
        if self.use_cuda and torch.cuda.is_available():
            next_state_batch = next_state_batch.cuda()
        n_vvalues, n_avalues = self.model(next_state_batch)
        nqvalues = n_vvalues.repeat(1, self.env.action_space.n) + (n_avalues - torch.mean(n_avalues).repeat(1, self.env.action_space.n))
        nqvalues = nqvalues.max(1)[0]
        nqvalues.volatile = False

        for i in range(batch_size):
            target[i] = batch.reward[i] + (1 - batch.is_terminal[i]) * self.gamma * nqvalues[i]
        if self.use_cuda and torch.cuda.is_available():
            target = target.cuda()

        errors = torch.abs(prediction.view(-1) - target).data
        for i in range(batch_size):
            idx = indexes[i]
            self.replay.update(idx, errors[i])

        loss = self.loss_function(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss

    ## not updated
    def train(self, batch_size=32, exp_replay=False, model_save=None, verbose=False,
              eval_every=10000):
        print('*' * 80)
        print('Training.......')
        print('*' * 80)
        t_counter = deque(maxlen=10)
        self.update_counter = 1
        stop_checkpointing = False

        self.checkpoint_directory = os.path.join("checkpoints", model_save)
        if os.path.exists(self.checkpoint_directory):
            shutil.rmtree(self.checkpoint_directory)
        os.makedirs(self.checkpoint_directory)

        if exp_replay:
            self.burn_in_memory()

        for episode in range(1, self.num_episodes + 1):
            cur_state = self.env.reset()
            for t in count():
                state_var = Variable(torch.FloatTensor(cur_state))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()
                vvalues, avalues = self.model(state_var)
                qvalues = vvalues.repeat(self.env.action_space.n) + (avalues - torch.mean(avalues).repeat(self.env.action_space.n))
                action = self.epsilon_greedy_policy(qvalues, eps_start=1)
                next_state, reward, done, _ = self.env.step(action)
                next_state_variable = Variable(torch.FloatTensor(next_state))
                nvvalues, navalues = self.model(next_state_variable)
                nqvalues = nvvalues.repeat(self.env.action_space.n) + (navalues - torch.mean(navalues).repeat(self.env.action_space.n))
                nqvalues = nqvalues.max(0)[0]
                target = reward + (1 - done) * self.gamma * nqvalues
                priority = math.fabs(target.data[0] - qvalues.max(0)[0])
                if exp_replay:
                    self.replay.append(self.transition(cur_state, action, next_state,
                                                       reward, done, priority))
                    loss = self.update_model_with_replay(batch_size)
                else:
                    loss = self.update_model(next_state, reward, qvalues, done)

                self.update_counter += 1
                cur_state = next_state

                if self.update_counter % eval_every == 0:
                    if self.update_counter % (5 * eval_every) == 0 and not stop_checkpointing:
                        self.model.save_model(os.path.join(self.checkpoint_directory,
                                                           "checkpoint_" + str(self.update_counter / (5 * eval_every))))
                    avg_reward = self.test(num_episodes=20, eval=True, num_updates=self.update_counter / 10000,
                                           verbose=False, render=False)
                    self.performance_average_reward.append(avg_reward)
                    t_counter.append(avg_reward)
                    if self.save_criteria(t_counter):
                        print('*' * 80)
                        print('Saving Best Model')
                        print('*' * 80)
                        stop_checkpointing = True
                        self.model.save_model(model_save)
                        self.plot_average_reward(model_save, eval_every)
                if done:
                    if verbose and episode % 100 == 0:
                        print('Episode %07d : Steps = %03d, Loss = %.2f' % (episode, t + 1, loss))
                    break
        print('*' * 80)
        print('Training Complete')
        print('*' * 80)
        return

    def test(self, num_episodes=100, eval=False, num_updates=0, verbose=True, render=False):
        print('*' * 80)
        print('Testing Performance.......')
        print('*' * 80)
        episode_reward_list = []
        for episode in range(1, num_episodes + 1):
            cur_state = self.test_env.reset()
            episode_reward = 0
            for t in count():
                if render and eval == False:
                    self.env.render()
                state_var = Variable(torch.FloatTensor(cur_state))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()
                vvalues, avalues = self.model(state_var)
                qvalues = vvalues.repeat(self.env.action_space.n) + (avalues - torch.mean(avalues).repeat(self.env.action_space.n))
                if eval:
                    action = self.epsilon_greedy_policy(qvalues, eps_start=0.05, eps_end=0.05)
                else:
                    action = self.greedy_policy(qvalues)

                next_state, reward, done, _ = self.test_env.step(action)
                cur_state = next_state
                episode_reward += reward
                if done:
                    if verbose:
                        print('Episode %07d : Steps = %03d, Reward = %.2f' % (episode, t, episode_reward))
                    episode_reward_list.append(episode_reward)
                    break
        episode_reward_list = np.array(episode_reward_list)
        print('*' * 80)
        print('Num_Updates(*10000/batchsize): %03d => Cumulative Reward: Mean= %f, Std= %f' % (num_updates,
                                                                                     episode_reward_list.mean(),
                                                                                     episode_reward_list.std()))
        print('*' * 80)
        return episode_reward_list.mean()

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        self.replay = Prioritized_Replay_Memory()
        state = self.env.reset()
        while len(self.replay.memory) < self.replay.burn_in:
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            priority = 1e4
            self.replay.append(self.transition(state, action, next_state, reward, done, priority))
            if done:
                state = self.env.reset()
            else:
                state = next_state
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--replay', dest='replay', default=False, action="store_true")
    parser.add_argument('--cuda', dest='use_cuda', default=False, action="store_true")
    parser.add_argument('--network', dest='network', type=str, default='mlp')
    parser.add_argument('--epochs', dest='num_episodes', type=int, default=20000)
    parser.add_argument('--load', dest='model_load', type=str, default=None)
    parser.add_argument('--duel', dest='duel', default=False, action="store_true")
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    usenetwork = args.network
    replay = args.replay
    use_cuda = args.use_cuda

    # You want to create an instance of the DQN_Agent class here, and then train / test it
    if args.duel:
        agent = Dueling_DQN_Agent(environment_name, num_episodes=args.num_episodes,
                          use_cuda=use_cuda, lrate=0.001)
    else:
        agent = DQN_Agent(environment_name, network=usenetwork, num_episodes = args.num_episodes,
                      use_cuda =use_cuda, lrate=0.001)
    if args.train:
        model_save = os.path.join('saved_models',usenetwork+'_'+environment_name+'_'+str(replay))
        agent.train(model_save = model_save, exp_replay = replay, verbose = True)
    elif os.path.exists(args.model_load):
        agent.render = True
        model_load = args.model_load
        agent.model.load_model(model_load)
        agent.test(render=False)

if __name__ == '__main__':
    main(sys.argv)

