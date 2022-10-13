from datetime import datetime as dt
import gym
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3
import numpy as np
import sys
from uuid import uuid4
import logging
import json
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Starting")


def get_seed():
    np.random.seed()
    return np.random.randint(0, 2**32)


def get_model(env, seed, model_type):
    if model_type == "a2c":
        model = A2C(
            policy = 'MlpPolicy',
            env = env,    
            tensorboard_log=log_dir,    
            verbose=0,
            seed=seed,
            device='cuda'
        )
    elif model_type == "ddpg":
        model = DDPG(
            policy = 'MlpPolicy',
            env = env,    
            tensorboard_log=log_dir,    
            verbose=0,
            seed=seed,
            device='cuda'
        )
    elif model_type == "dqn":
        model = DQN(
            policy = 'MlpPolicy',
            env = env,    
            tensorboard_log=log_dir,    
            learning_rate=1e-3,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.1,
            verbose=0,
            seed=seed,
            device='cuda'
        )
    elif model_type == "her":
        model = HER(
            policy = 'MlpPolicy',
            env = env,    
            tensorboard_log=log_dir,    
            verbose=0,
            seed=seed,
            device='cuda'
        )
    elif model_type == "ppo":
        model = PPO(
            policy = 'MlpPolicy',
            env = env,    
            tensorboard_log=log_dir,    
            verbose=0,
            seed=seed,
            device='cuda'
        )
    elif model_type == "sac":
        model = SAC(
            policy = 'MlpPolicy',
            env = env,    
            tensorboard_log=log_dir,    
            verbose=0,
            seed=seed,
            device='cuda'
        )
    elif model_type == "td3":
        model = TD3(
            policy = 'MlpPolicy',
            env = env,    
            tensorboard_log=log_dir,    
            verbose=0,
            seed=seed,
            device='cuda'
        )
    return model


if __name__ == '__main__':
    data_dir = "./data/"
    # read the script params
    with open(data_dir + 'config.json', 'r') as f:
        config = json.load(f)
    env_name = config['env_name']
    logger.info("env_name: %s", env_name)
    model_type = config['model_type']
    logger.info("model_type: %s", model_type)
    # create the environment
    env = gym.make(env_name)
    stable_baselines3.common.utils.get_device()
    seed = get_seed()
    # create the log directory if not exists\
    log_dir = data_dir + str(seed) + "/"
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info("Seed: %s", seed)
    # create the model
    model = get_model(env, seed, model_type)
    # train the model
    logger.info("Training "+str(dt.now()))
    model.learn(total_timesteps=5000000, tb_log_name=log_dir)
    logger.info("Complete "+str(dt.now()))
    # evaluate the model
    eval_env = gym.make(env_name)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    logger.info("mean_reward: %s", mean_reward)
    logger.info("std_reward: %s", std_reward)
    # save the model
    model.save(
        data_dir + 
        env_name + "_" + 
        model_type + "_" + 
        str(seed) + "_" + 
        str(mean_reward) + "_" + 
        str(std_reward)
        )
