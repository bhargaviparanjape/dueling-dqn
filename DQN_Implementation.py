#!/usr/bin/env python
from __future__ import print_function

fixedseed = 0
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import os
import gym
import sys
import copy
import argparse
from torch.autograd import Variable
import math
from itertools import count
from collections import deque
from collections import namedtuple
import copy
from gym import wrappers
import  matplotlib.pyplot as plt
from parameters import *
from SumTree import SumTree
import cv2
import shutil

class Logger():
    def __init__(self, location, trial = False):
        self.location = location
        self.trial = trial
    
    def printboth(self, message):
        print(message)
        if not self.trial:
            print(message, file=open(self.location,'a'))

class baseQNetwork(nn.Module):
    def __init__(self, env):
        super(baseQNetwork, self).__init__()
        self.env = env
    
    def forward(self):
        pass

    def load_model(self, model_file, location='cpu'):
        if location=='cpu':
            self.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(model_file))

    def save_model(self, file_name):
        torch.save(self.state_dict(), file_name)
    
class linearQNetwork(baseQNetwork):
    def __init__(self, env):
        super(linearQNetwork, self).__init__(env)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.linear = nn.Linear(self.input_size, self.output_size)
    
    def forward(self, input_state):
        output = self.linear(input_state)
        return output
    
class mlpQNetwork(baseQNetwork):
    def __init__(self, env, hidden1=16, hidden2=16):
        super(mlpQNetwork, self).__init__(env)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.linear1 = nn.Linear(self.input_size, self.hidden1)
        self.linear2 = nn.Linear(self.hidden1, self.hidden2)
        self.linear3 = nn.Linear(self.hidden2, self.output_size)
    
    def forward(self, input_state):
        hidden1 = self.linear1(input_state)
        hidden1 = F.relu(hidden1)
        hidden2 = self.linear2(hidden1)
        hidden2 = F.relu(hidden2)
        output = self.linear3(hidden2)
        return output

class duelmlpQNetwork(baseQNetwork):
    def __init__(self, env, hidden1=16, hidden2=16, hidden3=16):
        super(duelmlpQNetwork, self).__init__(env)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.linear1 = nn.Linear(self.input_size, self.hidden1)
        self.linear2 = nn.Linear(self.hidden1, self.hidden2)
        self.linear3_val = nn.Linear(self.hidden2, self.hidden3)
        self.output_val = nn.Linear(self.hidden3, 1)
        self.linear3_adv = nn.Linear(self.hidden2, self.hidden3)
        self.output_adv = nn.Linear(self.hidden1, self.output_size)

    def forward(self, input_state):
        hidden1 = self.linear1(input_state)
        hidden1 = nn.functional.relu(hidden1)
        hidden2 = self.linear2(hidden1)
        hidden2 = nn.functional.relu(hidden2)
        
        linear3_val = self.linear3_val(hidden2)
        linear3_val = nn.functional.relu(linear3_val)
        output1 = self.output_val(linear3_val)
        
        linear3_adv = self.linear3_adv(hidden2)
        linear3_adv = nn.functional.relu(linear3_adv)
        output2 = self.output_adv(linear3_adv)
        
        return output1, output2

class convQNetwork(baseQNetwork):
    def __init__(self, env):
        super(convQNetwork, self).__init__(env)
        self.output_size = self.env.action_space.n
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.linear1 = nn.Linear(2592, 256)
        self.linear2 = nn.Linear(256, self.output_size)

    def forward(self, input_state):
        x1 = F.relu(self.bn1(self.conv1(input_state)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.linear1(x2.view(x2.shape[0],-1)))
        x4 = self.linear2(x3)
        return x4
        
class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = deque(maxlen=self.memory_size)

    def sample_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        self.memory.append(transition)


class Prioritized_Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=50000):
        self.size = memory_size
        self.burn_in = burn_in
        self.tree = SumTree(self.size)
        self.eta = 0.0001
        self.alpha = 1
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
            batch.append((idx, data + (p,)))
        return batch
    
    # Will be called on the batch update
    def update(self, idx, error):
        p = self.getPriority(error)
        self.tree.update(idx, p)

    def append(self, transition):
        p = self.getPriority(transition[-1])
        self.tree.add(p, transition[:-1])
        ## supporting data structure to work with what aditya wrote
        if len(self.memory) < self.size:
            self.memory.append(0)

class DQN_Agent():

    def __init__(self, env_name, network, agent, exp_replay, paramdict, render = False, 
                 use_cuda=False, seed = 0): 

        # Initialize Agent Parameters
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.test_env = gym.make(env_name)
        self.test_env.seed(seed)
        self.use_cuda = use_cuda
        if paramdict["loss_function"] is not None:
            self.loss_function = nn.SmoothL1Loss()
        else:
            self.loss_function = nn.MSELoss()
        self.exp_replay = exp_replay
        self.agent = agent
        self.network = network
        
        self.paramdict = paramdict
        self.num_episodes = self.paramdict['num_episodes']
        self.gamma = self.paramdict['gamma']
        self.target_update = self.paramdict['target_update']
        self.plot_values = []

        if self.exp_replay == "priority":
            self.transition = namedtuple('transition', ('state', 'action', 'next_state',
                                                    'reward', 'is_terminal', 'priority'))
        elif self.exp_replay == "exp":
            self.transition = namedtuple('transition', ('state', 'action', 'next_state',
                                                    'reward', 'is_terminal'))
        
        # Initilize Q-network
        if self.network=='linear':
            self.model = linearQNetwork(self.env)
        elif self.network=='mlp' and self.agent=='duelling':
            self.model = duelmlpQNetwork(self.env)
        elif self.network=='mlp':
            self.model = mlpQNetwork(self.env)
        elif self.network=='conv' and self.agent=='doubledqn':
            self.model = convQNetwork(self.env)
        else:
            raise NotImplementedError()

        # Initialize Target Network
        if self.target_update is not None:
            self.target_model = copy.deepcopy(self.model)

        if env_name == "MountainCar-v0":
            high = self.env.observation_space.high
            low = self.env.observation_space.low
            self.mean = (high + low) / 2
            self.spread = abs(high - low) / 2

    def normalize(self, s):
        return s #(s - self.mean) / self.spread
            
    def get_epsilon(self, update_counter):
        eps_strat = self.paramdict['eps_strat']
        eps_start = self.paramdict['eps_start']
        eps_end = self.paramdict['eps_end']
        eps_decay = float(self.paramdict['eps_decay'])
        
        if eps_strat=='linear_decay':
            eps_threshold = eps_start + (eps_end - eps_start) * min((update_counter/eps_decay),1)
        elif eps_strat=='exp_decay':
            eps_decay= eps_decay/100
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.*update_counter/eps_decay)
        elif eps_strat=='log_decay':
            # eps_threshold = eps_end + (eps_start - eps_end)
            # max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
            raise NotImplementedError()
        return eps_threshold
            
    def epsilon_greedy_policy(self, qvalues, update_counter=None, eps_fixed=None):
        sample = random.random()
        if eps_fixed is not None:
            eps_threshold = eps_fixed
        else: 
            eps_threshold = self.get_epsilon(update_counter)
        if sample > eps_threshold:
            action = qvalues.data.max(1)[1]
        else:
            action = torch.LongTensor([random.randrange(self.model.output_size)])
        return action[0]

    def greedy_policy(self, qvalues):
        action = qvalues.data.max(1)[1]
        return action[0]
    
    def early_stop(self, t_counter):
        if self.env_name == 'CartPole-v0':
            return min(t_counter) > 195
        elif self.env_name == 'MountainCar-v0':
            return min(t_counter) > -110
        elif self.env_name == 'SpaceInvaders-v0':
            return min(t_counter) > 400

    def set_logger(self, model_location, logfile, trial=False):
        self.logger = Logger(os.path.join(model_location,logfile), trial=trial)
    
    def update_model(self, next_state, reward, qvalues, done):
        
        prediction = qvalues.max(1)[0]
        next_state_var = Variable(torch.FloatTensor(next_state).unsqueeze(0), 
                                  volatile=True)
        if self.use_cuda and torch.cuda.is_available():
            next_state_var = next_state_var.cuda()
        
        if self.target_update is None:
            if self.agent=='duelling':
                vvalues, avalues = self.model(next_state_var)
                nqvalues = vvalues.repeat(1,self.env.action_space.n) + \
                          (avalues - torch.mean(avalues).repeat(1,self.env.action_space.n))
            else:
                nqvalues = self.model(next_state_var)
        else:
            if self.agent=='duelling':
                vvalues, avalues = self.target_model(next_state_var)
                nqvalues = vvalues.repeat(1,self.env.action_space.n) + \
                          (avalues - torch.mean(avalues).repeat(1,self.env.action_space.n))
            else:
                nqvalues = self.target_model(next_state_var)

        nqvalues = nqvalues.max(1)[0]
        nqvalues.volatile = False

        ## extra condition to differentiate termination at the bottom vs termination at the top
        if self.env_name == "MountainCar-v0" and done == 1 and next_state[0] <= 0.5:
            done = 0

        target = reward + (1-done)* self.gamma* nqvalues
        if self.use_cuda and torch.cuda.is_available():
            target = target.cuda()
        
        loss = self.loss_function(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def update_model_with_replay(self, batch_size):
        
        batch = self.replay.sample_batch(batch_size)

        ## Extra code for replay memory with priority
        if self.exp_replay == "priority":
            indexes = [o[0] for o in batch]
            batch = [o[1] for o in batch]

        batch = self.transition(*zip(*batch))
        
        state_batch = Variable(torch.FloatTensor(batch.state))
        action_batch = Variable(torch.LongTensor(batch.action))
        
        if self.use_cuda and torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
        
        if self.agent=='duelling':
            vvalues_,avalues_  = self.model(state_batch)
            prediction = vvalues_.repeat(1, self.env.action_space.n) + \
                        (avalues_ - torch.mean(avalues_).repeat(1, self.env.action_space.n))
            prediction = prediction.gather(1, action_batch.view(batch_size, 1))
        else:
            prediction = self.model(state_batch).gather(1, action_batch.view(batch_size, 1))
        
        target = Variable(torch.zeros(batch_size))
        next_state_batch = Variable(torch.FloatTensor(batch.next_state),volatile=True)
        
        if self.use_cuda and torch.cuda.is_available():
            next_state_batch = next_state_batch.cuda()
        
        if self.target_update is None:
            if self.agent=='duelling':
                n_vvalues, n_avalues = self.model(next_state_batch)
                nqvalues = n_vvalues.repeat(1, self.env.action_space.n) + \
                (n_avalues - torch.mean(n_avalues).repeat(1, self.env.action_space.n))
            else:
                nqvalues = self.model(next_state_batch)
        else:
            if self.agent=='duelling':
                n_vvalues, n_avalues = self.target_model(next_state_batch)
                nqvalues = n_vvalues.repeat(1, self.env.action_space.n) + \
                (n_avalues - torch.mean(n_avalues).repeat(1, self.env.action_space.n))                
            else:
                nqvalues = self.target_model(next_state_batch)
        
        nqvalues = nqvalues.max(1)[0]
        nqvalues.volatile = False

        for i in range(batch_size):
            done = batch.is_terminal[i]
            if self.env_name == "MountainCar-v0" and done == 1 and batch.next_state[i][0] <= 0.5:
                done = 0
            target[i] = batch.reward[i] + (1 - done)* self.gamma* nqvalues[i]
        if self.use_cuda and torch.cuda.is_available():
            target = target.cuda()


        ## If using priority queue , update priorities
        if self.exp_replay == "priority":
            errors = torch.abs(prediction.view(-1) - target).data
            for i in range(batch_size):
                idx = indexes[i]
                self.replay.update(idx, errors[i])

        loss = self.loss_function(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def get_frames(self, cur_frame, previous_frames=None):
        num_frames = self.paramdict['num_frames'] 

        cur_frame= cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.resize(cur_frame,(84, 84))
        if previous_frames is None:
            final_frame = np.array([cur_frame]*num_frames).astype('float64')
        else:
            final_frame = np.zeros(previous_frames.shape)
            final_frame[:num_frames-1] = previous_frames[1:]
            final_frame[num_frames-1] = cur_frame
        return final_frame

    def train(self, model_save, verbose = False, trial=False):

        ## Initialize training parameters
        log_every = self.paramdict['log_every']
        eval_every = self.paramdict['eval_every']
        cp_every = self.paramdict['cp_every']
        stop_after = self.paramdict['stop_after']
        batch_size = self.paramdict['batch_size']
        repeat_action = self.paramdict['repeat_action']
        model_update_frequency = self.paramdict['update_frequency']
        choose_optimizer =  self.paramdict['optimizer']
        
        if repeat_action is None:
            repeat_action = 1
        
        if model_update_frequency is None:
            model_update_frequency = 1

        # Place model on Cuda
        if self.use_cuda and torch.cuda.is_available():
            self.model.cuda()
            if self.target_update is not None:
                self.target_model.cuda()

        #Initialize Optimizer
        if choose_optimizer =='rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.paramdict['lrate'])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.paramdict['lrate'])

        self.logger.printboth('*'*80)
        self.logger.printboth('Training.......')
        self.logger.printboth('*'*80)
        t_counter = deque(maxlen=stop_after)
        update_counter = 1
        model_update_counter = 0
        max_average_reward = float('-Inf')

        #Initialize burn-in memory
        if self.exp_replay=='exp' or self.exp_replay=='priority':
            self.logger.printboth("Initializing {0} Replay Memory".format(self.exp_replay))
            self.burn_in_memory()
            
        for episode in range(1,self.num_episodes+1):
            cur_state = self.env.reset()
            if self.env_name == "MountainCar-v0":
                cur_state = self.normalize(cur_state)
            if self.network == 'conv':
                cur_state = self.get_frames(cur_state)
            
            for t in count():
                
                if not trial and (update_counter-1) % cp_every ==0:
                    cpdir = os.path.join(model_save ,'checkpoint')
                    if not os.path.exists(cpdir):
                        os.makedirs(cpdir)
                    self.model.save_model(os.path.join(cpdir,str(update_counter-1)))

                #Sample random state, sample action from e-greedy policy and take step in enviroment
                state_var = Variable(torch.FloatTensor(cur_state).unsqueeze(0))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()
                if (update_counter-1) % repeat_action == 0:
                    if self.agent=='duelling':
                        vvalues, avalues = self.model(state_var)
                        qvalues = vvalues.repeat(1,self.env.action_space.n) + \
                                  (avalues - torch.mean(avalues).repeat(1,self.env.action_space.n))
                    else:
                        qvalues = self.model(state_var)

                    action = self.epsilon_greedy_policy(qvalues, model_update_counter+1)
                    epsilon = self.get_epsilon(model_update_counter+1)
                next_state, reward, done, _ = self.env.step(action)
                if self.env_name == "MountainCar-v0":
                    next_state = self.normalize(next_state)
                if self.network == 'conv':
                    next_state = self.get_frames(cur_frame=next_state, previous_frames=cur_state)


                # Calculate loss
                if self.exp_replay =='exp' or self.exp_replay=='priority':
                    ## Code for Prioritized replay
                    if self.exp_replay == "priority":
                        ## Another model computation is required if using priority based replay memory: to compute priority value
                        next_state_var = Variable(torch.FloatTensor(next_state))
                        if self.agent == 'duelling':
                            if self.target_update is None:
                                nvvalues, navalues = self.model(next_state_var)
                            else:
                                nvvalues, navalues = self.target_model(next_state_var)
                            nqvalues = nvvalues.repeat(1, self.env.action_space.n) + \
                                      (navalues - torch.mean(navalues).repeat(1, self.env.action_space.n))
                        else:
                            if self.target_update is None:
                                nqvalues = self.model(next_state_var)
                            else:
                                nqvalues = self.target_model(next_state_var)
                        nqvalues = nqvalues.max(0)[0]
                        target = reward + (1 - done) * self.gamma * nqvalues
                        priority = math.fabs(target.data[0] - qvalues.view(-1).max(0)[0])
                        self.replay.append(self.transition(cur_state, action, next_state,
                                                           reward, done, priority))

                    else:
                        self.replay.append(self.transition(cur_state, action, next_state,
                                                       reward, done))
                    if update_counter % model_update_frequency == 0:
                        loss = self.update_model_with_replay(batch_size)
                        model_update_counter += 1
                elif update_counter % model_update_frequency == 0:
                    loss = self.update_model(next_state, reward ,qvalues, done)
                    model_update_counter += 1

                ## Resets for delayed updates
                update_counter += 1
                cur_state = next_state

                # If using target network. update parameters of target parameter
                if self.target_update is not None and model_update_counter % self.target_update == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                # Evaluate on 20 episodes, Bookkeeping
                if (model_update_counter * model_update_frequency + \
                    ((update_counter-1) % model_update_frequency)) % (eval_every*model_update_frequency) == 0:
                    
                    assert (model_update_counter*model_update_frequency == (update_counter-1))
                    
                    avg_reward = self.test(num_test_episodes = 20, evaluate = True, verbose= False, 
                                           num_updates = model_update_counter/eval_every)
                    t_counter.append(avg_reward)
                    self.plot_values.append(avg_reward)
                    if not trial and avg_reward >= max_average_reward:
                        self.logger.printboth('*'*80)
                        self.logger.printboth('Saving Best Model')
                        self.logger.printboth('*'*80)
                        max_average_reward = avg_reward
                        self.model.save_model(os.path.join(model_save,'best_model'))
                    if self.early_stop(t_counter):
                        self.logger.printboth('*'*80)
                        self.logger.printboth('EarlyStopping.... Training Complete, Plotting and Recording')
                        self.logger.printboth('*'*80)
                        if not trial:
                            self.plot_average_reward(model_save, eval_every)
                        return
                if done:
                    if verbose and episode % log_every == 0:
                        self.logger.printboth('Episode %07d : Steps = %03d, Loss = %.2f, epsilon = %.2f' %(episode,t+1,loss, epsilon))
                    break

        self.logger.printboth('*'*80)
        self.logger.printboth('Training Complete')
        self.logger.printboth('*'*80)
        if not trial:
            self.plot_average_reward(model_save, eval_every)
        return

    def test(self, model_load=None, render = False, num_test_episodes = 100, evaluate = False, verbose = True, 
             num_updates = 0):
        self.logger.printboth('*'*80)
        self.logger.printboth('Testing Performance.......')
        self.logger.printboth('*'*80)
        
        if not evaluate:
            if self.use_cuda and torch.cuda.is_available():
                self.model.cuda()
                self.model.load_model(model_load)
            else:
                self.model.load_model(model_load, location='cpu')

        if render:
            model_load_loc, model_load_file = os.path.split(model_load)
            self.test_env = wrappers.Monitor(self.test_env, os.path.join(model_load_loc, model_load_file+'_recording'), 
                                             video_callable=lambda episode_id: True, force=True)
            
        episode_reward_list = []
        for episode in range(1,num_test_episodes+1):
            cur_state = self.test_env.reset()
            if self.env_name == "MountainCar-v0":
                cur_state = self.normalize(cur_state)
            if self.network == 'conv':
                cur_state = self.get_frames(cur_state)
            episode_reward = 0
            for t in count():
                if render and evaluate == False:
                    self.test_env.render()
                state_var = Variable(torch.FloatTensor(cur_state).unsqueeze(0))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()

                if self.agent=='duelling':
                    vvalues, avalues = self.model(state_var)
                    qvalues = vvalues.repeat(1,self.env.action_space.n) + \
                                (avalues - torch.mean(avalues).repeat(1, self.env.action_space.n))
                else:
                    qvalues = self.model(state_var)

                if evaluate:
                    action = self.epsilon_greedy_policy(qvalues, eps_fixed =0.05)
                else:
                    action = self.greedy_policy(qvalues)
                next_state, reward, done, _ = self.test_env.step(action)
                if self.env_name == "MountainCar-v0":
                    next_state = self.normalize(next_state)
                if self.network == 'conv':
                    next_state = self.get_frames(cur_frame=next_state, previous_frames=cur_state)
                cur_state = next_state
                episode_reward += reward
                if done:
                    if verbose:
                        self.logger.printboth('Episode %07d : Steps = %03d, Reward = %.2f' %(episode,t,episode_reward))
                    episode_reward_list.append(episode_reward)
                    break
        episode_reward_list = np.array(episode_reward_list)
        self.logger.printboth('*'*80)
        if evaluate:
            self.logger.printboth ('Num_Updates(*%d): %03d => Cumulative Reward: Mean= %f, Std= %f'  %(
                    self.paramdict['eval_every'],num_updates,episode_reward_list.mean(),
                    episode_reward_list.std()))
        else:
            self.logger.printboth('Cumulative Reward: Mean= %f, Std= %f'  %(episode_reward_list.mean(),
                                                             episode_reward_list.std()))
        self.logger.printboth('*'*80)
        return episode_reward_list.mean()
        
    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        memory_size = self.paramdict['memory_size']
        burn_in = self.paramdict['burn_in']

        if self.exp_replay == "priority":
            self.replay = Prioritized_Replay_Memory(memory_size=memory_size, burn_in=memory_size)
        else:
            self.replay = Replay_Memory(memory_size = memory_size, burn_in = burn_in)
        state = self.env.reset()
        if self.network == 'conv':
            state = self.get_frames(state)
        state_var = Variable(torch.FloatTensor(state).unsqueeze(0))
        if self.use_cuda and torch.cuda.is_available():
            state_var = state_var.cuda()
            
        if self.agent == 'duelling':
            vvalues, avalues = self.model(state_var)
            qvalues = vvalues.repeat(1, self.env.action_space.n) + \
                      (avalues - torch.mean(avalues).repeat(1, self.env.action_space.n))
        else:
            qvalues = self.model(state_var)
        while len(self.replay.memory) < self.replay.burn_in:
            ## Action still follows policy with very large exploration
            action = self.epsilon_greedy_policy(qvalues, eps_fixed=0.9)
            next_state, reward, done, _ = self.env.step(action)
            if self.env_name == "MountainCar-v0":
                state = self.normalize(state)
                next_state = self.normalize(next_state)
            if self.network == 'conv':
                next_state = self.get_frames(cur_frame=next_state, previous_frames=state)

            ## Code to assign initial priorities to burn-in for prioritized replay
            if self.exp_replay == "priority":
                next_state_var = Variable(torch.FloatTensor(next_state))
                if self.use_cuda and torch.cuda.is_available():
                    next_state_var = next_state_var.cuda()
                if self.agent == 'duelling':
                    ## Target model not required since they are identical
                    nvvalues, navalues = self.model(next_state_var)
                    nqvalues = nvvalues.repeat(1, self.env.action_space.n) + \
                               (navalues - torch.mean(navalues).repeat(1, self.env.action_space.n))
                else:
                    nqvalues = self.model(next_state_var)
                nqvalues = nqvalues.max(0)[0]
                target = reward + (1 - done) * self.gamma * nqvalues
                priority = math.fabs(target.data[0] - qvalues.view(-1).max(0)[0])
                # Priority = 1e4
                self.replay.append(self.transition(state, action, next_state, reward, done, priority))
            else:
                self.replay.append(self.transition(state, action, next_state, reward, done))

            if done:
                state = self.env.reset()
                if self.network == 'conv':
                    state = self.get_frames(state)
            else:
                state = next_state
        self.logger.printboth('*'*80)
        self.logger.printboth('Memory Burn In Complete')
        self.logger.printboth('*'*80)

    def plot_average_reward(self, model_save, eval_every):
        plt.plot(self.plot_values)
        plt.xlabel("Number of updates (*{0})".format(eval_every))
        plt.ylabel("Cumulative Average Reward")
        plt.savefig(os.path.join(model_save, "reward_plot.png"))

            
def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--network', dest='network', type=str, default='mlp')
    parser.add_argument('--replay', dest='replay', type=str, default='exp')
    parser.add_argument('--agent', dest='agent', type=str, default='dqn')
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--load', dest='model_load', type=str, default=None)
    parser.add_argument('--trial', dest='trial', type=int, default=0)
    parser.add_argument('--cuda', dest='use_cuda', type=int, default=0)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--run', dest='run', type=str, default='0')
    parser.add_argument('--seed',dest='seed', type=int, default=0)
    return parser.parse_args()

def main(args):
    args = parse_arguments()

    ## Primary Arguments for Code
    env_name = args.env
    network = args.network
    replay = args.replay
    agent = args.agent

    use_cuda = args.use_cuda
    is_trial = args.trial
    model_load = args.model_load
    run = args.run
    render = args.render==1
    seed = args.seed
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    ## Check for Saved Models
    model_location = os.path.join('saved_models','_'.join([env_name,network,str(replay),agent,run,str(seed)]))
    if args.train and not is_trial:
        if os.path.exists(model_location):
            print('A model with same specifications and seed exist.... Aborting.....')
            return
        else:
            os.makedirs(model_location)

    ## Create Agent
    paramdict = get_parameters(env_name, network, replay, agent, run)
    dqn_agent = DQN_Agent(env_name = env_name, network=network, exp_replay = replay, agent= agent, 
                          paramdict = paramdict, use_cuda = use_cuda, seed=seed)

    if args.train:
        dqn_agent.set_logger(model_location=model_location, logfile= 'trainlogs.txt', trial=is_trial)
        dqn_agent.train(model_save = model_location,verbose = True, trial=is_trial)
    elif os.path.exists(model_load):
        model_load_loc, model_load_file = os.path.split(model_load)
        if render:
            dqn_agent.set_logger(model_location=model_load_loc, logfile= model_load_file + '_testlogs.txt',
                                 trial = True)
            dqn_agent.test(model_load, num_test_episodes=1, render=True)
        else:
            dqn_agent.set_logger(model_location=model_load_loc, logfile= model_load_file + '_testlogs.txt')
            dqn_agent.test(model_load, num_test_episodes = 100)

if __name__ == '__main__':
    main(sys.argv)


