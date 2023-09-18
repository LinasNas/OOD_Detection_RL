"""
This file loads in an environment from envs.py OR mujoco_envs and an optimal policy for that environments from data/policies
and rolls out a large number of trajectories.

These trajectories are then stores in a file (HDF5 or memmap).

The idea is then to load these pre-generated data files in for analysis of the different OOD detectors.

The file is implemented in a way that is compatible with the Haider 2023 codebase, so that we can use their detectors on the rollouts generated here.
"""

import argparse
import os
import random
import time
from distutils.util import strtobool


import hashlib
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from tqdm import tqdm
import pickle
from collections import namedtuple
from utils.data import save_object, load_object
#this is crucial for custom environments to be loaded
#import envs

from utils.env_utils import make_env

def generate_hash():
    timestamp = str(time.time())  # Use current timestamp as the identifier
    hash_object = hashlib.md5(timestamp.encode())  # 
    return hash_object.hexdigest()  # Get the hexadecimal representation of the hash

def parse_float_list(string):
    if string == "" or string == "None":
        return None
    try:
        floats = [float(x) for x in string.split(',')]
        return floats
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float value encountered")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    
    # Evnironment
    parser.add_argument("--env-id", type=str, default="CartPole-v0",
        help="the id of the environment")
    parser.add_argument('--env-noise-corr', type=parse_float_list, required=False, default=(0.0,),help="which noise to train on, e.g. type: << --env-noise-corr ")
    parser.add_argument('--env-noise-std', type=float, required=False, default=0.0, help="the magnitude of noise")
    parser.add_argument("--num-envs", type=int, default=1,
        help="number of environments created in parallel")
    parser.add_argument("--capture-video", type=bool, default=False,
        help="Whether to create a video recording of the agent's trajectory")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    
    parser.add_argument("--haider-env", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use one of Haider's envs or not")
    parser.add_argument("--discrete", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether the environment is discrete or not")
    
    #policy/agent
    parser.add_argument("--policy-path", type=str, default="../assets/policies/CartPole-v0/PPO_policy/5000000_timesteps/model.pth", help="the path to the policy to be loaded for rollout generation")
    parser.add_argument("--policy-name",default="PPO",choices=["PPO"],type=str, help="name/class of the policy that interacts with the env")
    parser.add_argument("--num-episodes", type=int, default=50, help="number of episodes to generate rollouts for")
    #currently not used
    parser.add_argument("--max-steps-per-episode", type=int, default=200,
        help="the number of steps to run in each environment per episode")

    args = parser.parse_args()
    return args


def make_env_cleanrl(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if args.haider_env == True:
            env = make_env(seed=None, env_id=env_id, anomaly_delay=None, mod=None)
        else:
            env = gym.make(env_id, seed=None)

        # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video:
        #     if idx == 0:
        #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    # def __init__(self, envs):
    #     super().__init__()
    #     self.critic = nn.Sequential(
    #         layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
    #         nn.Tanh(),
    #         layer_init(nn.Linear(64, 64)),
    #         nn.Tanh(),
    #         layer_init(nn.Linear(64, 1), std=1.0),
    #     )
    #     self.actor_mean = nn.Sequential(
    #         layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
    #         nn.Tanh(),
    #         layer_init(nn.Linear(64, 64)),
    #         nn.Tanh(),
    #         layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
    #     )
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


#DEFINE ROLLOUT PROCEDURE
rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])
def policy_rollout(env, policy, max_steps_per_episode):
    #create empty lists for states, actions, rewards, dones
    states, actions, rewards, dones = ([],[],[],[],)

    #reset the environment
    state = torch.Tensor(env.reset(seed=args.seed)[0]).to(device)
    #store the state variable
    states.append(state.cpu().numpy())

    # if hasattr(policy, "reset"):
    #     policy.reset()

    #run until done
    done = False
    while not done:
            #pick action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(state)
            
            #take step
            next_state, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            
            #calculate done
            done = [1 if te or tr else 0 for te, tr in zip(terminated, truncated)]

            #store
            states.append(next_state)
            actions.append(action.cpu().numpy())
            rewards.append(reward)
            dones.append(done[0])
            
            next_state = torch.Tensor(next_state).to(device)
            done = torch.Tensor(done).to(device)
            state = next_state

    #reshape arrays to conform to Haider
    state_arr = np.array(states)
    state_arr = state_arr.reshape(state_arr.shape[0], -1)
    action_arr = np.array(actions).reshape(len(actions), -1)
    reward_arr = np.array(rewards).reshape(-1)
    done_arr = np.array(dones).reshape(-1)

    #return np.array(states), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)
    return state_arr, action_arr, reward_arr, done_arr

def n_policy_rollouts(env, 
                      policy, 
                      n_episodes, 
                      max_steps_per_episode,
                      verbose=False):
    episodes = []
    trange = tqdm(range(n_episodes), position=0, desc="episode", ncols=80) if verbose else range(n_episodes)
    for n in trange:
        states, actions, rewards, dones = policy_rollout(env, policy, max_steps_per_episode)
        episode = rollout(states, actions, rewards, dones)
        episodes.append(episode)
    return episodes


if __name__ == "__main__":
    args = parse_args()
    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # print(args.haider_env)
    # print(type(args.haider_env))
    # print(args.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("")
    print("DEVICE: ", device)
    print("")

    options = {"mod_corr_noise": args.env_noise_corr,
               "mod_noise_std": args.env_noise_std}

    run_name = f"{args.env_id}__{args.seed}__{int(time.time())}"

    # if args.discrete == True:
    #     envs = gym.vector.SyncVectorEnv(
    #         [make_env_cleanrl(args.env_id, args.seed + i, i, args.capture_video, run_name, options) for i in range(args.num_envs)]
    #         )
        
    #     assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # else:
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env_cleanrl(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)

    if args.policy_path is not None:
        agent.load_state_dict(torch.load(args.policy_path))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # #===GENERATE ROLLOUTS===
    # #initialize the agent
    # agent = Agent(envs).to(device)

    # #load the policy
    # # agent.load_state_dict(torch.load(full_model_path))
    # if os.path.exists(args.policy_path):
    #     print("")
    #     print("=== LOAD POLICY ===")
    #     print("loading policy from:", args.policy_path)
    #     agent.load_state_dict(torch.load(args.policy_path, map_location=torch.device(device)))
    #     agent.eval()
    #     print("=== POLICY LOADED ===")
              
    # else:
    #     raise ValueError("the specified policy path does not exist! Please specify a proper policy path with --policy-path 'path_to_policy.pth'")

    print("")
    print("=== GENERATE ROLLOUTS===")
    print("Number of episodes: ", args.num_episodes)
    
    #create data directory
    data_path = os.path.join(
        "data",
        "rollouts",
        args.env_id,
        args.policy_name + "_policy",
        f"{args.num_episodes}_ep",
        generate_hash(),
        "ep_data.pkl")

    #generate rollouts
    print(f"generating rollout data")
    ep_data = n_policy_rollouts(env=envs, 
                                policy=agent, 
                                n_episodes=args.num_episodes,
                                max_steps_per_episode=args.max_steps_per_episode)
    print(f"generated rollout data")
    
    # Save rollouts
    parent_dir = os.path.dirname(os.getcwd())
    full_data_path = os.path.join(parent_dir, data_path)
    os.makedirs(os.path.dirname(full_data_path), exist_ok=True)
    save_object(ep_data, full_data_path)
    print("=== ROLLOUTS SAVED ===\n")
    envs.close()