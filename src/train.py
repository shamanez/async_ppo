#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization


Written by Patrick Coady (pat-coady.github.io)

Modified by Tin-Yin Lai (wu6u3) into asynchronous version


PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
import numpy as np
from gym import wrappers
from thread_policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import time
from rmsprop_applier import RMSPropApplier

import threading
from multiprocessing.managers import BaseManager     
import multiprocessing as mp 
import tensorflow as tf

class MPManager(BaseManager): 
    pass


MPManager.register('Policy', Policy) 
MPManager.register('NNValueFunction', NNValueFunction)


N_WORKERS = 3
RMSP_ALPHA = 0.99
RMSP_EPSILON = 0.1
GRAD_NORM_CLIP = 40.0 
DEVICE = device = "/cpu:0"

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(sess, env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(sess, obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(sess, env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for _ in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(sess, env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(sess, trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(sess, observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode, time):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_Time': time
                })



def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, policy_logvar):
    '''
    '''
    ##################
    #  shared policy #
    ##################

    tic = time.clock()

    manarger = MPManager() 
    manarger.start()

    shared_env, shared_obs_dim, shared_act_dim = init_gym(env_name)
    shared_obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    shared_logger = Logger(logname=env_name, now=now+"-Master")
    shared_aigym_path = os.path.join('./vedio', env_name, now+"-Master")
    #env = wrappers.Monitor(env, aigym_path, force=True)
    shared_scaler = Scaler(shared_obs_dim)


    shared_val_func = NNValueFunction(shared_obs_dim, hid1_mult, -1, None)
    shared_policy = Policy(shared_obs_dim, shared_act_dim, kl_targ, hid1_mult, policy_logvar, -1, None)





    learning_rate_input = tf.placeholder("float")
    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device) 


        
    # lacal policy declair 
    env_a = [None]*N_WORKERS
    obs_dim_a = [None]*N_WORKERS
    act_dim_a = [None]*N_WORKERS
    logger_a  = [None]*N_WORKERS
    aigym_path_a = [None]*N_WORKERS
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    val_func_a  = [None]*N_WORKERS
    policy_a = [None]*N_WORKERS
    scaler_a = [None]*N_WORKERS
    for i in range(N_WORKERS):
        env_a[i], obs_dim_a[i], act_dim_a[i] = init_gym(env_name)
        obs_dim_a[i] += 1  # add 1 to obs dimension for time step feature (see run_episode())
        logger_a[i] = Logger(logname=env_name, now=now+"-"+str(i))
        aigym_path_a[i] = os.path.join('./vedio', env_name, now+"-"+str(i))
        #env_a[i] = wrappers.Monitor(env, aigym_path, force=True)
        scaler_a[i] = Scaler(obs_dim_a[i])

        val_func_a[i] = NNValueFunction(obs_dim_a[i], hid1_mult, i, shared_val_func)
        val_func_a[i].apply_gradients = grad_applier.apply_gradients(
            shared_val_func.get_vars(),
            val_func_a[i].gradients )


        policy_a[i] = Policy(obs_dim_a[i], act_dim_a[i], kl_targ, hid1_mult, policy_logvar, i, shared_policy)
        policy_a[i].apply_gradients = grad_applier.apply_gradients(
            shared_policy.get_vars(),                         
            policy_a[i].gradients ) 


    # init tensorflow
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))
    init = tf.global_variables_initializer()
    
    ## start sess
    sess.run(init)

    


    ## init shared scalar policy
    run_policy(sess, shared_env, shared_policy, shared_scaler, shared_logger, episodes=5)

    
    def single_work(thread_idx):
        """ training loop

        Args:
            env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
            num_episodes: maximum number of episodes to run
            gamma: reward discount factor (float)
            lam: lambda from Generalized Advantage Estimate
            kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
            batch_size: number of episodes per policy training batch
            hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
            policy_logvar: natural log of initial policy variance
        """
        env = env_a[thread_idx] 
        policy = policy_a[thread_idx]
        obs_dim = obs_dim_a[thread_idx]
        act_dim = act_dim_a[thread_idx]
        logger = logger_a[thread_idx]
        aigym_path = aigym_path_a[thread_idx]
        scaler = scaler_a[thread_idx]
        val_func = val_func_a[thread_idx]
    
    
        print("=== start thread "+str(policy.get_thread_idx())+" "+ policy.get_scope()+" ===")
        print(shared_policy.get_vars())
        print(policy.get_vars())


        # run a few episodes of untrained policy to initialize scaler:
        #run_policy(sess, env, policy, scaler, logger, episodes=5)


        #policy.sync(shared_policy)
        #val_func.sync(shared_val_func)
        episode = 0

        while episode < num_episodes:

            ## copy global var into local
            sess.run( policy.sync)
            sess.run(val_func.sync)              

            ## compute new model on local policy
            trajectories = run_policy(sess, env, policy, scaler, logger, episodes=batch_size)
            episode += len(trajectories)
            add_value(sess, trajectories, val_func)  # add estimated values to episodes
            add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            add_gae(trajectories, gamma, lam)  # calculate advantage
            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
            # add various stats to training log:
            log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode, time.clock()-tic)



            policy.update(sess, observes, actions, advantages, logger)  # update policy
            val_func.fit(sess, observes, disc_sum_rew, logger)  # update value function




           #cur_learning_rate = self._anneal_learning_rate(global_t)
            feed_dict = { 
                          policy.old_log_vars_ph: policy.old_log_vars_np,
                          policy.old_means_ph: policy.old_means_np,
                          policy.obs_ph: observes,
                          policy.act_ph: actions,
                          policy.advantages_ph: advantages,
                          policy.beta_ph: policy.beta,
                          policy.lr_ph: policy.lr,
                          policy.eta_ph: policy.eta,
                          learning_rate_input: policy.lr
                         }

            sess.run( policy.apply_gradients, feed_dict)
            
            shared_policy.update(sess, observes, actions, advantages, shared_logger) 

            feed_dict = { 
                          val_func.obs_ph: observes,
                          val_func.val_ph: disc_sum_rew,
                          learning_rate_input: val_func.lr
                         }


            sess.run( val_func.apply_gradients, feed_dict)

            shared_val_func.fit(sess, observes, disc_sum_rew, shared_logger)

            shared_logger.log({'_Time':time.clock()-tic})

            logger.write(display=True)  # write logger results to file and stdout
            


        logger.close()
    ## end def single work 




    train_threads = []
    for i in range(N_WORKERS):
        train_threads.append(threading.Thread(target=single_work, args=(i,)))
                                                       
    





    [t.start() for t in train_threads]  
    [t.join() for t in train_threads]


    saver = tf.train.Saver()
    for i in range(N_WORKERS):
        logger_a[i].close()
    
    #path = os.path.join('log-files', env_name, now+'-Master', 'checkpoint')
    #saver.save(sess, path )

    sess.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)

    args = parser.parse_args()
    main(**vars(args))
