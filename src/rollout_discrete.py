"""
This file loads in an environment from envs.py and an optimal policy for that environments from data/policies
and rolls out a large number of trajectories.

These trajectories are then stores in a file (HDF5 or memmap).

The idea is then to load these pre-generated data files in for analysis of the different OOD detectors.

The file is implemented in a way that is compatible with the Haider 2023 codebase, so that we can use their detectors on the rollouts generated here.
"""

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
from tqdm import tqdm
import pickle
from collections import namedtuple
from utils.data import save_object, load_object

#this is crucial for custom environments to be loaded
import envs

def generate_hash():
    timestamp = str(time.time())  # Use current timestamp as the identifier
    hash_object = hashlib.md5(timestamp.encode())
    short_hash = hash_object.hexdigest()[:10]  # Get the hexadecimal representation of the hash
    return short_hash  # Get the hexadecimal representation of the hash

def parse_float_list(string):
    if string == "" or string == "None":
        return None
    try:
        floats = [float(x) for x in string.split(',')]
        return floats
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float value encountered")

def parse_float_tuple(string):
    if string == "" or string == "None":
        return None
    try:
        # Remove parentheses and split the values using commas
        cleaned_string = string.strip('()')
        
        # Check if the cleaned string is empty
        if cleaned_string:
            floats = [float(x) for x in cleaned_string.split(',') if x.strip() != '']
            
            # Handle single-element tuple
            if len(floats) == 1:
                return floats[0],  # Add a trailing comma to create a single-element tuple
            else:
                return tuple(floats)
        else:
            raise ValueError("Empty string")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float value encountered")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # Evnironment
    parser.add_argument("--env-id", type=str, default="CartPole-v0",
        help="the id of the environment")
    
    parser.add_argument('--env-noise-corr', type=parse_float_tuple, required=False, default=(0.0,0.0),help="correlations for noise in the environment: (0.0) means no correlation of noise, (1.0, 0.0) means one-step correlation only, (0.0, 1.0) means two step correlation only, etc.")
    parser.add_argument('--noise-strength', type=float, required=False, default=1.0, help="the strength of the noise applied to tthe env")
    parser.add_argument("--injection-time", type=int, default=0,
        help="injection time of the anomaly (0 = from start of episode)")

    parser.add_argument("--num-envs", type=int, default=1,
        help="number of environments created in parallel")
    parser.add_argument("--capture-video", type=bool, default=False,
        help="Whether to create a video recording of the agent's trajectory")
    
    #policy/agent
    parser.add_argument("--policy-path", type=str, default="../assets/policies/CartPole-v0/PPO_policy/5000000_timesteps/model.pth", help="the path to the policy to be loaded for rollout generation")
    parser.add_argument("--policy-name",default="PPO",choices=["PPO"],type=str, help="name/class of the policy that interacts with the env")
    parser.add_argument("--num-episodes", type=int, default=50, help="number of episodes to generate rollouts for")
    #currently not used
    parser.add_argument("--max-steps-per-episode", type=int, default=200,
        help="the number of steps to run in each environment per episode")
    
    args = parser.parse_args()
    return args


def make_env(env_id, seed, idx, capture_video, run_name, options=None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        #LINAS
        env.unwrapped.ep_length = env.spec.max_episode_steps
        #env.unwrapped.ep_length = 10000
        if options is not None:
            for k, v in options.items():
                setattr(env.unwrapped, k, v)
        return env
    return thunk

#DEFINE AGENT CLASS
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


#DEFINE ROLLOUT PROCEDURE
rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])
def policy_rollout(env, policy, max_steps_per_episode):
    #create empty lists for states, actions, rewards, dones
    states, actions, rewards, dones = ([],[],[],[],)

    #reset the environment
    state = torch.Tensor(env.reset(seed=args.seed)[0]).to(device)
    #store the state variable
    states.append(state.cpu().numpy())

    if hasattr(policy, "reset"):
        policy.reset()

    #run until done
    done = False
    while not done:

    #LINAS: 
    # print("NOTE: generating episodes of fixed length! ", args.max_steps_per_episode)
    # for _ in range(args.max_steps_per_episode):
            #pick action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(state)
            
            #take step
            next_state, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        

            #calculate done
            done = [1 if te or tr else 0 for te, tr in zip(terminated, truncated)]
            #done = done[0]

            #DO NOT SET ACTION TO ALWAYS 0
            # if args.env_id == "TimeSeriesEnv-v0":
            #     pass
            #     #action = torch.zeros(1, dtype=torch.int64)
            #     #action = torch.tensor([0])
            #     # print("action", action)
            #     # print("type action", type(action.cpu().numpy()[0]))
            #     # print("type state", type(next_state))
            # else:
            #     pass

            #store
            states.append(next_state)
            actions.append(action.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("")
    print("DEVICE: ", device)
    print("")

    options = {
        "test_mod_corr_noise": args.env_noise_corr,
        "test_noise_strength": args.noise_strength,
        "injection_time": args.injection_time}
    
    run_name = f"{args.env_id}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, options) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    #===GENERATE ROLLOUTS===
    #initialize the agent
    agent = Agent(envs).to(device)

    #load the policy
    # agent.load_state_dict(torch.load(full_model_path))
    if os.path.exists(args.policy_path):
        print("")
        print("=== LOAD POLICY ===")
        print("loading policy from:", args.policy_path)
        agent.load_state_dict(torch.load(args.policy_path))
        agent.eval()
        print("=== POLICY LOADED ===")
              
    else:
        raise ValueError("the specified policy path does not exist! Please specify a proper policy path with --policy-path 'path_to_policy.pth'")

    print("")
    print("=== GENERATE ROLLOUTS===")
    print("Number of episodes: ", args.num_episodes)
    
    #create data directory
    if args.env_id in ["IMANOCartpoleEnv-v0", "IMANSCartpoleEnv-v0", "TimeSeriesEnv-v0"]:
        data_path = os.path.join(
            "..",
            "data",
            "rollouts",
            "rollout_discrete",
            args.env_id,
            "env_noise_corr_" + "_".join([str(elem).replace(".", "p") for elem in args.env_noise_corr]) + "_noise_strength_" + str(args.noise_strength).replace('.', 'p'),
            args.policy_name + "_policy",
            f"{args.num_episodes}_ep",
            generate_hash(),
            "ep_data.pkl")
    else:
        data_path = os.path.join(
            "..",
            "data",
            "rollouts",
            "rollout_discrete",
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
    #parent_dir = os.path.dirname(os.getcwd())
    #full_data_path = os.path.join(parent_dir, data_path)
    #os.makedirs(os.path.dirname(full_data_path), exist_ok=True)
    #save_object(ep_data, full_data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    save_object(ep_data, data_path) 
    print("=== ROLLOUTS SAVED ===\n")
    
    envs.close()