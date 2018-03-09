from collections import defaultdict

def get_parameters(env, network, replay, agent, run='0'):
    paramdict = defaultdict(lambda: None)

    ##############################  MOUNTAIN CAR  ##########################################



    ## A model ran with these settings and produced results at seed 91, do not change !!
    if env=='MountainCar-v0' and network=='linear' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['cp_every'] = 10000
        paramdict['target_update'] = 10000
        paramdict['loss_function'] = 1
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 10000
        paramdict['burn_in'] = 10000

    ## if target update is kept fixed, then we get a slow learner, so i removed it. seed 91 saved
    elif env=='MountainCar-v0' and network=='linear' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['cp_every'] = 100000
        paramdict['batch_size'] = 32
        paramdict['target_update'] = 10000
        # paramdict['loss_function'] = 1
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 12000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000

    ## didnt work well : come back to it later
    elif env=='MountainCar-v0' and network=='linear' and replay=='priority' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['cp_every'] = 10000
        paramdict['batch_size'] = 32
        paramdict['target_update'] = 10000
        # paramdict['loss_function'] = 1
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 10000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000

    ## target fixing made it slower
    elif env=='MountainCar-v0' and network=='mlp' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 10000
        paramdict['log_every'] = 10
        paramdict['cp_every'] = 100000
        # paramdict['target_update'] = 10000
        paramdict['loss_function'] = 1
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 10000
        paramdict['burn_in'] = 10000

    ## 999 not good, trying 91
    elif env=='MountainCar-v0' and network=='mlp' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 20000
        paramdict['log_every'] = 10
        paramdict['cp_every'] = 100000
        paramdict['batch_size'] = 32
        paramdict['target_update'] = 10000
        # paramdict['loss_function'] = 1
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 20000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000

    elif env=='MountainCar-v0' and network=='mlp' and replay=='exp' and agent=='duelling':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 20000
        paramdict['log_every'] = 5
        paramdict['cp_every'] = 100000
        paramdict['batch_size'] = 32
        paramdict['target_update'] = 10000
        # paramdict['loss_function'] = 1
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 20000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000

    ## for 999 without standardizing the reward functions, trains but not that great
    elif env=='MountainCar-v0' and network=='mlp' and replay=='priority' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 20000
        paramdict['log_every'] = 10
        paramdict['cp_every'] = 100000
        paramdict['target_update'] = 1
        paramdict['loss_function'] = 1
        paramdict['batch_size'] = 32
        paramdict['target_update'] = 10000
        # paramdict['loss_function'] = 1
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 15000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000

    ### 999 works magically
    elif env=='MountainCar-v0' and network=='mlp' and replay=='priority' and agent=='duelling':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 1
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.9
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 20000
        paramdict['log_every'] = 5
        paramdict['cp_every'] = 100000
        paramdict['batch_size'] = 32
        paramdict['target_update'] = 10000
        # paramdict['loss_function'] = 1
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 15000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000

######################################  CART POLE ########################################################

    elif env=='CartPole-v0' and network=='linear' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 1000
        paramdict['log_every'] = 5
        paramdict['cp_every'] = 2000
        paramdict['stop_after'] = 5
        paramdict['burn_in'] = 10000
        paramdict['num_episodes'] = 500

    elif env=='CartPole-v0' and network=='mlp' and replay=='noexp' and agent=='dqn':
        paramdict['lrate'] = 0.001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 5000
        paramdict['log_every'] = 20
        paramdict['cp_every'] = 20000
        paramdict['stop_after'] = 5
        paramdict['burn_in'] = 10000
        paramdict['num_episodes'] = 5000

    elif env=='CartPole-v0' and network=='linear' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.0001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 100
        paramdict['log_every'] = 1
        paramdict['cp_every'] = 1000
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 10000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000

    elif env=='CartPole-v0' and network=='mlp' and replay=='exp' and agent=='dqn':
        paramdict['lrate'] = 0.001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 5000
        paramdict['log_every'] = 10
        paramdict['cp_every'] = 10000
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 2000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000

    elif env=='CartPole-v0' and network=='mlp' and replay=='exp' and agent=='duelling':
        paramdict['lrate'] = 0.001
        paramdict['gamma'] = 0.99
        paramdict['eps_strat'] = 'linear_decay'
        paramdict['eps_start'] = 0.5
        paramdict['eps_end'] = 0.05
        paramdict['eps_decay'] = 1e6
        paramdict['eval_every'] = 5000
        paramdict['log_every'] = 10
        paramdict['cp_every'] = 10000
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['num_episodes'] = 2000
        paramdict['burn_in'] = 10000
        paramdict['memory_size'] = 50000
    
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
        paramdict['cp_every'] = 100000
        paramdict['batch_size'] = 32
        paramdict['stop_after'] = 5
        paramdict['target_update'] = 10000
        paramdict['repeat_action'] = 4
        paramdict['update_frequency'] = 4
        paramdict['num_episodes'] = 10000
        paramdict['burn_in'] = 50000
        paramdict['memory_size'] = 1000000
        paramdict['num_frames'] = 4
        
    else:
        raise NotImplementedError()

    return paramdict