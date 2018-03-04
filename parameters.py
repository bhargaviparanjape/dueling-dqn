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
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 10000
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
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 10000
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
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='MountainCar-v0' and network=='mlp' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 2000
        paramdict['log_every'] = 5
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='MountainCar-v0' and network=='mlp' and replay=='exp' and agent=='duelling':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 2000
        paramdict['log_every'] = 5
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='CartPole-v0' and network=='linear' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 2000
        paramdict['log_every'] = 10
        paramdict['stop_after'] = 5
        paramdict['burn_in'] = 10000
        paramdict['num_episodes'] = 20000
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
        paramdict['burn_in'] = 10000
        paramdict['num_episodes'] = 5000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='CartPole-v0' and network=='linear' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 2000
        paramdict['log_every'] = 10
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 10000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000
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
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='CartPole-v0' and network=='mlp' and replay=='exp' and agent=='duelling':
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
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])

    elif env=='SpaceInvaders-v0' and network=='conv' and replay=='exp' and agent=='doubledqn' and run=='0':
        paramdict['optimizer'] = 'rmsprop'
        paramdict['lrate'] = 0.00025
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.1
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 2
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['target_update'] = 10000
        paramdict['repeat_action'] = 4
        paramdict['update_frequency'] = 4
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 50000
        paramdict['memory_size'] = 1000000
        paramdict['num_frames'] = 4
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])
        
    elif env=='SpaceInvaders-v0' and network=='conv' and replay=='exp' and agent=='doubledqn' and run=='1':
        paramdict['optimizer'] = 'adam'
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.1
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 2
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['target_update'] = 10000
        paramdict['repeat_action'] = 4
        paramdict['update_frequency'] = 4
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 50000
        paramdict['memory_size'] = 1000000
        paramdict['num_frames'] = 4
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])
    
    elif env=='SpaceInvaders-v0' and network=='conv' and replay=='exp' and agent=='doubledqn' and run=='2':
        paramdict['optimizer'] = 'rmsprop'
        paramdict['lrate'] = 0.00025
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.1
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 2
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['repeat_action'] = 4
        paramdict['update_frequency'] = 4
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 50000
        paramdict['memory_size'] = 1000000
        paramdict['num_frames'] = 4
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])
        
    elif env=='SpaceInvaders-v0' and network=='conv' and replay=='exp' and agent=='doubledqn' and run=='3':
        paramdict['optimizer'] = 'rmsprop'
        paramdict['lrate'] = 0.00025
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 1
        paramdict['eps_end'] = 0.1
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 2
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['target_update'] = 10000
        paramdict['repeat_action'] = 1
        paramdict['update_frequency'] = 1
        paramdict['num_episodes'] = 50000
        paramdict['burn_in'] = 50000
        paramdict['memory_size'] = 1000000
        paramdict['num_frames'] = 4
        paramdict['model_save'] = '_'.join([env,network,str(replay),agent,run])
    
    else:
        raise NotImplementedError()

    return paramdict