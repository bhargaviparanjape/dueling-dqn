#!/usr/bin/env python
from __future__ import print_function
import os
import gym
import sys
import copy
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
from itertools import count
import torch.optim as optim
import torch
from collections import deque
from collections import namedtuple
import copy
from parameters import *
import cv2

class baseQNetwork(nn.Module):
    def __init__(self, env):
        super(baseQNetwork, self).__init__()
        self.env = env
    
    def forward(self):
        pass

    def load_model(self, model_file):
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

class DQN_Agent():

    def __init__(self, env_name, network, agent, exp_replay, paramdict, render = False, 
                 use_cuda=False): 
        
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.use_cuda = use_cuda
        self.loss_function = nn.MSELoss()
        self.exp_replay = exp_replay
        self.agent = agent
        self.network = network
        
        self.paramdict = paramdict
        self.num_episodes = self.paramdict['num_episodes']
        self.gamma = self.paramdict['gamma']
        self.target_update = self.paramdict['target_update']
        
        if self.network=='linear':
            self.model = linearQNetwork(self.env)
        elif self.network=='mlp':
            self.model = mlpQNetwork(self.env)
        elif self.network=='conv' and self.agent=='doubledqn':
            self.model = convQNetwork(self.env)
        elif self.network=='mlp' and self.agent=='duellingdqn':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        if self.target_update is not None:
            self.target_model = copy.deepcopy(self.model)
            
        self.transition = namedtuple('transition', ('state', 'action', 'next_state', 
                                                    'reward', 'is_terminal'))
        
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
            raise NotImplementedError()
            
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
            return min(t_counter) == 200
        elif self.env_name == 'MountainCar-v0':
            return min(t_counter) > -120
        elif self.env_name == 'SpaceInvaders-v0':
            return min(t_counter) > 1000
    
    def update_model(self, next_state, reward, qvalues, done):
        
        prediction = qvalues.max(1)[0]
        next_state_var = Variable(torch.FloatTensor(next_state).unsqueeze(0), 
                                  volatile=True)
        if self.use_cuda and torch.cuda.is_available():
            next_state_var = next_state_var.cuda()
        
        if self.target_update is None:
            nqvalues = self.model(next_state_var)
        else:
            nqvalues = self.target_model(next_state_var)
        nqvalues = nqvalues.max(1)[0]
        nqvalues.volatile = False
        
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
        
        if self.target_update is None:
            nqvalues = self.model(next_state_batch)
        else:
            nqvalues = self.target_model(next_state_batch)
        
        nqvalues = nqvalues.max(1)[0]
        nqvalues.volatile = False

        for i in range(batch_size):
            target[i] = batch.reward[i] + (1 - batch.is_terminal[i])* self.gamma* nqvalues[i]
        if self.use_cuda and torch.cuda.is_available():
            target = target.cuda()

        loss = self.loss_function(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss/batch_size
    
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

    def train(self, verbose = False, trial=False):
        
        log_every = self.paramdict['log_every']
        eval_every = self.paramdict['eval_every']
        stop_after = self.paramdict['stop_after']
        batch_size = self.paramdict['batch_size']
        model_save = os.path.join('saved_models', self.paramdict['model_save'])
        repeat_action = self.paramdict['repeat_action']
        model_update_frequency = self.paramdict['update_frequency']
        choose_optimizer =  self.paramdict['optimizer']
        
        if repeat_action is None:
            repeat_action = 1
        
        if model_update_frequency is None:
            model_update_frequency = 1
        
        if self.use_cuda and torch.cuda.is_available():
            self.model.cuda()
            if self.target_update is not None:
                self.target_model.cuda()
        
        if choose_optimizer =='rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.paramdict['lrate'])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.paramdict['lrate'])
            
        print ('*'*80)
        print ('Training.......')
        print ('*'*80)
        t_counter = deque(maxlen=stop_after)
        update_counter = 1
        model_update_counter = 1
        max_average_reward = float('-Inf')
        if self.exp_replay=='exp':
            self.burn_in_memory()
            
        for episode in range(1,self.num_episodes+1):
            cur_state = self.env.reset()
            if self.network == 'conv':
                cur_state = self.get_frames(cur_state)
            
            for t in count():
                state_var = Variable(torch.FloatTensor(cur_state).unsqueeze(0))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()
                if (update_counter-1) % repeat_action == 0:
                    qvalues = self.model(state_var)
                    action = self.epsilon_greedy_policy(qvalues, update_counter)
                next_state, reward, done, _ = self.env.step(action)
                if self.network == 'conv':
                    next_state = self.get_frames(cur_frame=next_state, previous_frames=cur_state)
                
                if self.exp_replay=='exp':
                    self.replay.append(self.transition(cur_state, action, next_state, 
                                                       reward, done))
                    if update_counter % model_update_frequency == 0:
                        loss = self.update_model_with_replay(batch_size)
                        model_update_counter += 1
                elif update_counter % model_update_frequency == 0:
                    loss = self.update_model(next_state, reward ,qvalues, done)
                    model_update_counter += 1
                    
                update_counter += 1
                cur_state = next_state
                
                if self.target_update is not None and model_update_counter % self.target_update == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                
                if model_update_counter% eval_every == 0:
                    avg_reward = self.test(num_test_episodes = 20, evaluate = True, verbose= False, 
                                           num_updates = model_update_counter/eval_every)
                    t_counter.append(avg_reward)
                    if not trial and avg_reward >= max_average_reward:
                        print ('*'*80)
                        print ('Saving Best Model')
                        print ('*'*80)
                        max_average_reward = avg_reward
                        self.model.save_model(model_save)
                    if self.early_stop(t_counter):
                        print ('*'*80)
                        print ('EarlyStopping.... Training Complete')
                        print ('*'*80)
                        return
                if done:
                    if verbose and episode % log_every == 0:
                        print('Episode %07d : Steps = %03d, Loss = %.2f' %(episode,t+1,loss))
                    break
        print ('*'*80)
        print ('Training Complete')
        print ('*'*80)
        return

    def test(self, model_load=None, num_test_episodes = 100, evaluate = False, verbose = True, 
             num_updates = 0):
        print ('*'*80)
        print ('Testing Performance.......')
        print ('*'*80)
        
        if not evaluate:
            if self.use_cuda and torch.cuda.is_available():
                self.model.cuda()
            self.model.load_model(model_load)
            
        episode_reward_list = []
        for episode in range(1,num_test_episodes+1):
            cur_state = self.test_env.reset()
            if self.network == 'conv':
                cur_state = self.get_frames(cur_state)
            episode_reward = 0
            for t in count():
                state_var = Variable(torch.FloatTensor(cur_state).unsqueeze(0))
                if self.use_cuda and torch.cuda.is_available():
                    state_var = state_var.cuda()
                qvalues = self.model(state_var)
                if evaluate:
                    action = self.epsilon_greedy_policy(qvalues, eps_fixed =0.05)
                else:
                    action = self.greedy_policy(qvalues)
                next_state, reward, done, _ = self.test_env.step(action)
                if self.network == 'conv':
                    next_state = self.get_frames(cur_frame=next_state, previous_frames=cur_state)
                cur_state = next_state
                episode_reward += reward
                if done:
                    if verbose:
                        print('Episode %07d : Steps = %03d, Reward = %.2f' %(episode,t,episode_reward))
                    episode_reward_list.append(episode_reward)
                    break
        episode_reward_list = np.array(episode_reward_list)
        print ('*'*80)
        if evaluate:
            print ('Num_Updates(*%d): %03d => Cumulative Reward: Mean= %f, Std= %f'  %(
                    self.paramdict['eval_every'],num_updates,episode_reward_list.mean(),
                    episode_reward_list.std()))
        else:
            print ('Cumulative Reward: Mean= %f, Std= %f'  %(episode_reward_list.mean(),
                                                             episode_reward_list.std()))
        print ('*'*80)
        return episode_reward_list.mean()
        
    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        memory_size = self.paramdict['memory_size']
        burn_in = self.paramdict['burn_in']
        self.replay = Replay_Memory(memory_size = memory_size, burn_in = burn_in)
        state = self.env.reset()
        if self.network == 'conv':
            state = self.get_frames(state)
        while len(self.replay.memory) < self.replay.burn_in:
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            if self.network == 'conv':
                next_state = self.get_frames(cur_frame=next_state, previous_frames=state)

            self.replay.append(self.transition(state, action, next_state, reward, done))
            if done:
                state = self.env.reset()
                if self.network == 'conv':
                    state = self.get_frames(state)
            else:
                state = next_state
        print('*'*80)
        print('Memory Burn In Complete')
        print('*'*80)

            
def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--network', dest='network', type=str, default='mlp')
    parser.add_argument('--replay', dest='replay', type=int, default=0)
    parser.add_argument('--agent', dest='agent', type=str, default='dqn')
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--load', dest='model_load', type=str, default=None)
    parser.add_argument('--trial', dest='trial', type=int, default=0)
    parser.add_argument('--cuda', dest='use_cuda', type=int, default=0)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--run', dest='run', type=str, default='0')
    return parser.parse_args()

def main(args):
    args = parse_arguments()
    env_name = args.env
    network = args.network
    replay = 'exp' if args.replay else 'noexp'
    agent = args.agent
    use_cuda = args.use_cuda
    is_trial = args.trial
    model_load = args.model_load
    run = args.run
    
    model_location = os.path.join('saved_models','_'.join([env_name,network,str(replay),agent,run]))
    if args.train and not is_trial and os.path.exists(model_location):
        print('A model with same specifications exist')
        print(os.listdir('saved_models'))
        print('Aborting')
        return
    
    paramdict = get_parameters(env_name, network, replay, agent, run)
    dqn_agent = DQN_Agent(env_name = env_name, network=network, exp_replay = replay, 
                          agent= agent, paramdict = paramdict, use_cuda = use_cuda)

    if args.train:
        dqn_agent.train(verbose = True, trial=is_trial)
    elif os.path.exists(model_load):
        dqn_agent.test(model_load)

if __name__ == '__main__':
    main(sys.argv)


