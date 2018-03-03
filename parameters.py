from collections import defaultdict

def get_parameters(env, network, replay, agent):
    paramdict = defaultdict(lambda: None)

    if env=='MountainCar-v0' and network=='linear' and replay==False and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['save_after'] = 10
        paramdict['num_episodes'] = 5000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent])

    elif env=='MountainCar-v0' and network=='mlp' and replay==False and agent=='dqn':
    	paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['hidden1'] = 16
        paramdict['hidden2'] = 16
        paramdict['save_after'] = 10
        paramdict['num_episodes'] = 5000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent])

    elif env=='MountainCar-v0' and network=='linear' and replay==True and agent=='dqn':
    	paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
       	paramdict['batch_size'] = 32
        paramdict['save_after'] = 10
        paramdict['num_episodes'] = 5000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent])

    elif env=='MountainCar-v0' and network=='mlp' and replay==True and agent=='dqn':
    	paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['hidden1'] = 16
        paramdict['hidden2'] = 16
        paramdict['batch_size'] = 32
        paramdict['save_after'] = 10
        paramdict['num_episodes'] = 5000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent])

    elif env=='MountainCar-v0' and network=='mlp' and replay==False and agent=='duelling':
    	raise NotImplementedError()

    elif env=='MountainCar-v0' and network=='conv' and replay==False and agent=='doubledqn':
    	raise NotImplementedError()

    if env=='CartPole-v0' and network=='linear' and replay==False and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 100
        paramdict['save_after'] = 10
        paramdict['num_episodes'] = 15000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent])

    elif env=='CartPole-v0' and network=='mlp' and replay==False and agent=='dqn':
    	paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 100
        paramdict['hidden1'] = 16
        paramdict['hidden2'] = 16
        paramdict['save_after'] = 10
        paramdict['num_episodes'] = 20000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent])

    elif env=='CartPole-v0' and network=='linear' and replay==True and agent=='dqn':
    	paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 100
       	paramdict['batch_size'] = 32
        paramdict['save_after'] = 10
        paramdict['num_episodes'] = 15000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent])

    elif env=='CartPole-v0' and network=='mlp' and replay==True and agent=='dqn':
    	paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 100
        paramdict['hidden1'] = 16
        paramdict['hidden2'] = 16
        paramdict['batch_size'] = 32
        paramdict['save_after'] = 10
        paramdict['num_episodes'] = 12500
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent])

    elif env=='CartPole-v0' and network=='mlp' and replay==False and agent=='duelling':
        raise NotImplementedError()

    elif env=='CartPole-v0' and network=='conv' and replay==False and agent=='doubledqn':
        raise NotImplementedError()

    return paramdict