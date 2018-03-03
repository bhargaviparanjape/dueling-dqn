from collections import defaultdict

def get_parameters(env, network, replay, agent, run='0'):
    paramdict = defaultdict(lambda: None)

    if env=='MountainCar-v0' and network=='linear' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['stop_after'] = 10
        paramdict['num_episodes'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='MountainCar-v0' and network=='mlp' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['stop_after'] = 10
        paramdict['num_episodes'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='MountainCar-v0' and network=='linear' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 10
        paramdict['num_episodes'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='MountainCar-v0' and network=='mlp' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 10
        paramdict['num_episodes'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='MountainCar-v0' and network=='mlp' and replay=='exp' and agent=='duelling':
        raise NotImplementedError()

    if env=='CartPole-v0' and network=='linear' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 100
        paramdict['stop_after'] = 10
        paramdict['num_episodes'] = 10000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='CartPole-v0' and network=='mlp' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 5000
        paramdict['log_every'] = 10
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 10000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='CartPole-v0' and network=='linear' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 100
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 10
        paramdict['num_episodes'] = 10000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='CartPole-v0' and network=='mlp' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 5000
        paramdict['log_every'] = 10
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 2000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='CartPole-v0' and network=='mlp' and replay=='exp' and agent=='duelling':
        raise NotImplementedError()

    elif env=='SpaceInvader-v0' and network=='conv' and replay=='exp' and agent=='doubledqn':
        raise NotImplementedError()
    
    else:
        raise NotImplementedError()

    return paramdict