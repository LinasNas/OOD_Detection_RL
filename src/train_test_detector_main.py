"""
This file loads in a pregenerated dataset, and trains an OOD detector
"""

import os
import torch
import argparse
import numpy as np
import time 
import hashlib
import dask.dataframe as dd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.feature_extraction import settings
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import pandas as pd
import concurrent.futures
import argparse
from collections import namedtuple
from typing import List, Optional, Union
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import pickle
from stable_baselines3.common.base_class import BaseAlgorithm
from mujoco_envs.utils.env_utils import make_env_haider
from detectors.haider.base_detector import Base_Detector
from detectors.haider.classical.knn_detector import KNN_Detector
from detectors.haider.classical.guassian_detector import GAUSSIAN_Detector
from detectors.haider.classical.gmm_detector import GMM_Detector
from detectors.haider.classical.isolation_forest import ISOFOREST_Detector
from detectors.haider.classical.random_detector import RANDOM_Detector
from detectors.haider.pedm.pedm_detector import PEDM_Detector
from detectors.haider.lstm.lstm_detector import LSTM_Detector
from detectors.haider.riqn.riqn_detector import RIQN_Detector
#KEY IMPORT for custom envs
import envs

#===IMPORTS FOR R TO PYTHON====#
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
ocd = importr("ocd")

#===BEGIN: R WRAPPERS===
def float_vector(arr):
    if isinstance(arr, (int, float, np.int64, np.float32, np.float64)):
        arr = np.array([arr])
    else:
        pass
    return robjects.vectors.FloatVector(arr)

def arr_to_vec(arr):
    robjects.numpy2ri.activate()
    return numpy2ri.py2rpy(arr)

def dict_to_listvec(dic):
    robjects.numpy2ri.activate()
    return robjects.ListVector(dic)

def rnorm(n, mean = 0):
    return np.random.normal(loc = mean, size = int(n))

def c(*args):
    return np.concatenate(args)
#====END R WRAPPERS====

from utils.data import (
    load_object,
    n_policy_rollouts,
    save_object,
    split_train_test_episodes,
)
from utils.stats import eval_metrics

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
    
def parse_str_or_int(string):
    if string == "random":
        return "random"
    else:
        return int(string)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="seed of the experiment")

    parser.add_argument("--policy-path", type=str, default="", help="the path to the policy to be loaded for rollout generation of both train and test environments")
    parser.add_argument("--train-data-path", type=str, default="", help="Path of rollout data (to be used to train the detector)")

    #environments
    parser.add_argument("--train-env-id", type=str, default="CartPole-v0",
        help="name of the train environment")
    parser.add_argument("--test-env-id", type=str, default="CartPole-v0",
        help="name of the test environment")
    parser.add_argument("--max-steps-per-episode", type=int, default=200,
        help="the number of steps to run in each environment per episode")
    
    parser.add_argument('--train-env-noise-corr', type=parse_float_tuple, required=False, default=(0.0,0.0),help="set the correlations for noise in the train environment: (0.0) means no correlation of noise, (1.0, 0.0) means one-step correlation only, (0.0, 1.0) means two step correlation only, etc.")
    parser.add_argument('--train-noise-strength', type=float, required=False, default=1.0, help="the strength of the noise applied to the train env")
    parser.add_argument('--test-env-noise-corr', type=parse_float_tuple, required=False, default=(0.0,0.0),help="set the correlations for noise in the test environment: (0.0) means no correlation of noise, (1.0, 0.0) means one-step correlation only, (0.0, 1.0) means two step correlation only, etc.")
    parser.add_argument('--test-noise-strength', type=float, required=False, default=1.0, help="the strength of the noise applied to the env")
    parser.add_argument("--train-injection-time", type=int, default=0,
        help="injection time of the anomaly (0 = from start of episode)")
    # parser.add_argument("--test-injection-time", type = parse_str_or_int, default="random",help="injection time of the anomaly (0 = from start of episode)")
     
    #detector-specific
    parser.add_argument("--detector-name", required=True, type=str, default = "PEDM_Detector", help="class/type of the detector to use")
    parser.add_argument("--detector-path", required=False, default = "", type=str, help="path to save the detector")

    parser.add_argument("--num-train-episodes", default=2000, type=int, help="number of training episodes to train the dynamics model")
    parser.add_argument("--num-test-episodes", default=200, type=int, help="number of episodes to test the detector")

    parser.add_argument("--num-envs", type=int, default=1,
        help="number of environments created in parallel")
    parser.add_argument("--capture-video", type=bool, default=False,
        help="Whether to create a video recording of the agent's trajectory")

    #TF specific
    parser.add_argument("--TF-train-data-feature-path", type=str, default="", help="Path to feature extractions of train data")
    parser.add_argument("--TF-imputer-path", type=str, default="", help="Path to feature extractions of train data")

    parser.add_argument(
        "--mods",
        default="['act_factor_severe']",
        type=str,
        help="in case of Haider envs, which mods to evaluate on, e.g. type: << --mods \"['act_factor_severe']\" >>. if not provided, will run all mods ",
    )
    parser.add_argument("--haider-env", type=lambda x: bool(strtobool(x)), default=False, help="run on anomalous mujoco envs from Haider et al 2023")

    args = parser.parse_args()
    return args

#===BEGIN: IMPORTS FROM CLEANRL====#
#source: https://github.com/vwxyzjn/cleanrl
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def make_env(env_id, run_name = generate_hash(), capture_video = False, seed = None,  options = None):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    # env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.unwrapped.ep_length = env.spec.max_episode_steps
    if options is not None:
        for k, v in options.items():
            setattr(env.unwrapped, k, v)
    return env
#===END: IMPORTS FROM CLEANRL====#

#===BEGIN: IMPORTS FROM HAIDER ET AL 2023====#
#source: https://github.com/FraunhoferIKS/pedm-ood/blob/main/oodd_runner.py
#DEFINE ROLLOUT PROCEDURE
rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])
def policy_rollout(env, policy):

    #create empty lists for states, actions, rewards, dones
    states, actions, rewards, dones = ([],[],[],[],)

    #reset the environment
    state = torch.Tensor(env.reset()[0]).to(device)
    states.append(state.cpu().numpy())
    
    done = False
    while not done:
            #pick action
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(state)
            
            #take step
            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            
            #calculate done
            done = terminated or truncated 

            #store
            states.append(next_state)
            actions.append(action.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            
            state = torch.Tensor(next_state).to(device)

    #reshape arrays to conform to Haider
    state_arr = np.array(states)
    state_arr = state_arr.reshape(state_arr.shape[0], -1)
    action_arr = np.array(actions).reshape(len(actions), -1)
    reward_arr = np.array(rewards).reshape(-1)
    done_arr = np.array(dones).reshape(-1)

    # return np.array(states), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)
    return state_arr, action_arr, reward_arr, done_arr

def policy_rollout_haider(env, policy, max_steps=2e3):
    states, actions, rewards, dones = (
        [],
        [],
        [],
        [],
    )
    state = env.reset()
    states.append(state)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    while not done:
        action, _ = policy.predict(state, deterministic=True)
        n_state, reward, done, _info = env.step(action)
        states.append(n_state)
        actions.append(action)
        rewards.append(reward)
        state = n_state.copy()
        if len(states) > max_steps:
            print("aborting long episode")
            done = True
        dones.append(done)

    return np.array(states), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)

def train_detector(
    args,
    #env: gym.Env,
    env,
    num_train_episodes: int,
    detector_class: type,
    data_path: str = "",
    data: list = [],
    detector_kwargs: Optional[dict] = {},
    detector_fit_kwargs: Optional[dict] = {},
    haider_env: bool = False,
):
    """
    main function to train an env dynamics model with data from some policy in some env

    Args:
        env: env to collect experience in
        data_path: path to save to or load experience buffer from (if applicable)
        n_train_episodes: how many episodes to collect/train THE DETECTOR
        detector_name: type/class of the detector
        detector_kwargs: kwargs to pass for the detector constructor
        detector_fit_kwargs="kwargs for the training loop of the detector"

    Returns:
        detector: the trained ood detector
    """

    if haider_env == True:
        ep_data = data

    else:
        #load rollout data
        if os.path.exists(data_path):
            print("loading rollout data")
            ep_data = load_object(data_path)
        
        else:
            raise ValueError("the specified data rollout path does not exist! Please specify a proper policy path with --train-data-path 'path_to_data.pkl'")

    #key: apply data shape normalization
    if (args.detector_name == "LSTM_Detector") or (args.detector_name == "RIQN_Detector"):
        ep_data = normalize_data_shape(ep_data)
    else:
        pass

    train_ep_data, val_ep_data = split_train_test_episodes(episodes=ep_data)

    # initialize the detector
    detector = detector_class(env=env, 
                              normalize_data=True,
                              num_train_episodes=args.num_train_episodes)
    
    print("")
    print("Detector: ", args.detector_name)
    print("Training environment: ", args.train_env_id)
    print("")
    
    #train the detector
    detector.fit(train_ep_data=train_ep_data, val_ep_data=val_ep_data, **detector_fit_kwargs)
    print("")

    return detector
#===END: IMPORTS FROM HAIDER ET AL 2023====#


def train_test_detector_CPD(args, 
                           detector,
                           injection_time,
                           observations_test,
                           actions_test):
    #LOAD train data
    if os.path.exists(args.train_data_path):
        print("loading rollout data")
        ep_data = load_object(args.train_data_path)   
    else:
        raise ValueError("the specified data rollout path does not exist! Please specify a proper policy path with --train-data-path 'path_to_data.pkl'")

    train_ep_data, val_ep_data = split_train_test_episodes(episodes=ep_data)
    #train_ep_data = ep_data

    #==BEGIN TRAINING==#
    #needed because we have to reset the detector after each episode, by the implementation of it
    print("")
    print("===BEGIN TRAINING CPD DETECTOR===")
    detector = ocd.setStatus(detector,'estimating')

    states_train = [ep.states for ep in train_ep_data]
    actions_train = [ep.actions for ep in train_ep_data]

    #for each episode:
    for episode_idx in range(0, len(train_ep_data)):
    
        #loop through the episode
        for i in range(0, len(actions_train[episode_idx])):
            
            #get the state
            state_train = states_train[episode_idx][i]
            action_train = actions_train[episode_idx][i]

            #IF USE ACTIONS:
            # x_new = np.concatenate((state_train, action_train), axis = 0)
            # x_new = float_vector(x_new)
            #ELSE:
            if state_train.shape[0] == 1:
                state_train = np.concatenate((state_train, state_train), axis = 0)
            else:
                pass

            x_new = float_vector(state_train)
            #add to the detector

            detector = ocd.getData(detector, x_new)

    print("Detector: ", detector)
    print("===CPD DETECTOR TRAINED===")
    print("")
    
    
    #TEST THE DETECTOR
    anom_scores = []
    anom_score = 0
    #iterate over each step in the episode
    print("===BEGIN TESTING CPD DETECTOR===")
    detector = ocd.setStatus(detector, 'monitoring')
    for i in range(0, len(actions_test)):
        
        state = observations_test[i]
        action = actions_test[i]

        x_new = float_vector(state)

        detector = ocd.getData(detector, x_new)
        status = ocd.status(detector)

        if isinstance(status[0], float) == True:
            anom_score = 1
        else:
            anom_score = 0

        anom_scores.append(anom_score)
    
    print("Detector:", detector)
    print("===CPD DETECTOR TESTED===")
    print("")
    
    #reset the detector for the next episode
    detector = ocd.reset(detector)

    return anom_scores, detector


#NOTE: This is a temporary workaround to make the data shape compatible with the detector.
#Main issue: In LSTM and RIQN, Haider has implicitly hardcoded the fact that in episode of train data, the data is of the shape (max_len, 4)
#max_len is 201 in Cartpole-v0. If that's not reached even in a single episode, the code breaks, and unfortunately, I wasn't able to find a beter way to fix this, besides changing the architecture of the world dynamics models
#in train_detector, see: ep_data = normalize_data_shape(ep_data)
#rollout is used as the default data structure when loading the training data
rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])
def normalize_data_shape(ep_data, target_size = 201):
    print("")
    print("NOTE: Normalizing data shape to be compatible with the detector.")
    print("")
    irregular_episodes = []
    for ep in ep_data:
        if ep.states.shape[0] != target_size:
            irregular_episodes.append(ep.states.shape[0])

    if irregular_episodes == []:
        return ep_data

    else:
        max_size = min(irregular_episodes) - 1
        print("")
        print("WARNING: Irregular length of episodes detected. Truncating to length: ", max_size)
        print("")

        reduced_ep_data = [
            rollout(ep.states[:max_size], 
                    ep.actions[:max_size], 
                    ep.rewards[:max_size], 
                    ep.dones[:max_size])
            for ep in ep_data]
        return reduced_ep_data

def load_policy_haider(env, device):
    from stable_baselines3 import TD3
    # path = path or os.path.join("data", "checkpoints", env.spec.id, "TD3", "best_model.zip")
    path = "../assets/policies/Haider/MJCartpole-v0/TD3/best_model.zip"
    policy = TD3.load(path, env=env, device=device)
    return policy

#====== TF DETECTORS ======#
def extract_features_from_multidim_batch(batch, settings):
    num_dimensions = batch.shape[1]
    all_features = []
    
    def extract_features_for_dim(dim):
        df = pd.DataFrame(batch[:, dim], columns=['value'])
        df['id'] = 0
        df['time'] = range(len(batch))

        X = extract_features(df, column_id="id", column_sort="time", column_value="value", 
                             impute_function=np.nanmean, default_fc_parameters=settings, 
                             disable_progressbar=True, n_jobs=1)
        return X.values[0]  # extract the row as a 1D array
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        all_features = list(executor.map(extract_features_for_dim, range(num_dimensions)))

    return np.hstack(all_features)

def preprocess_data(list_data, batch_size, step_size=1):
    processed_data = []

    for episode in list_data:
        batched_array = [episode[i:i+batch_size] for i in range(0, len(episode) - batch_size + 1, step_size)]
        processed_data.append(batched_array)

    return processed_data

def train_detector_TF(
    args,
    batch_size, 
    n_dimensions):
    """
    """

    #IF EXTRACTED FEATURES ARE AVAILABLE, LOAD THEM
    if args.TF_train_data_feature_path != "" and args.TF_imputer_path != "":
        #load the extracted features      
        if os.path.exists(args.TF_train_data_feature_path):
            print("Loading feature extractions data from: ", args.TF_train_data_feature_path)

            with open(args.TF_train_data_feature_path, 'rb') as file:
                features_imputed = pickle.load(file)    

        else:
            raise ValueError("the specified extrated train data feature path does not exist!")
        
        #load the impyter
        if os.path.exists(args.TF_imputer_path):
            print("Loading imputer from: ", args.TF_imputer_path)

            with open(args.TF_imputer_path, 'rb') as file:
                imputer = pickle.load(file) 

        else:
            raise ValueError("the specified imputer path does not exist!")
        
    #ELSE: IF NO EXTRACTED FEATURES ARE AVAILABLE, EXTRACT THEM
    else:
        #load rollout data
        if os.path.exists(args.train_data_path):
            print("loading rollout data")
            ep_data = load_object(args.train_data_path)
        
        else:
            raise ValueError("the specified data rollout path does not exist! Please specify a proper policy path with --train-data-path 'path_to_data.pkl'")
        
        train_ep_data, val_ep_data = split_train_test_episodes(episodes=ep_data)
        
        # initialize the detector
        print("")
        print("Extracting features from train data...")
        num_eps = len(train_ep_data)
        states_train = [ep.states for ep in train_ep_data]
        action_train = [ep.actions for ep in train_ep_data]

        processed_train_data = preprocess_data(list_data = states_train, 
                                               batch_size = batch_size)

        settings_efficient = settings.EfficientFCParameters()

        features = []
        train_ep_ctr = 0
        batch_ctr = 0

        for episode in processed_train_data:
            batch_ctr = 0
            print("Episode: ", train_ep_ctr)
            
            for batch in episode:
                X = extract_features_from_multidim_batch(batch=batch, settings=settings_efficient)
                features.append(X)
                if batch_ctr % 50 == 0:
                    print("Batch: ", batch_ctr)
                batch_ctr +=1 
            train_ep_ctr += 1
            
        features = np.vstack(features)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(features)
        # Transform the training data
        features_imputed = imputer.transform(features)

        print("TRAIN DATA PROCESSING FINISHED")

        #save the extracted features & their values
        if not os.path.exists(os.path.dirname(args.detector_path)):
            args.curr_path = os.path.join(
                "data",
                "extracted_features",
                args.train_env_id,
                args.detector_name,
                "env_noise_corr_" + "_".join([str(elem).replace(".", "p") for elem in args.test_env_noise_corr]) + "_noise_strength_" + str(args.test_noise_strength).replace('.', 'p'),
                "ep_" + str(num_eps))
        
        parent_dir = os.path.dirname(os.getcwd())
        os.makedirs(os.path.join(parent_dir, args.curr_path), exist_ok=True)  
        #os.makedirs(os.path.dirname(parent_dir), exist_ok=True)  
        
        train_data_path = os.path.join(parent_dir, args.curr_path, "train_data_features.pkl")
        imputer_path = os.path.join(parent_dir, args.curr_path, "imputer.pkl")

        with open(train_data_path, 'wb') as f:
            pickle.dump(features_imputed, f)

        with open(imputer_path, 'wb') as f:
            pickle.dump(imputer, f)
        print("")
        print("=== Train data features saved in: ", train_data_path, " ===\n")
        print("=== Imputer saved in: ", imputer_path, " ===\n")

    #TRAIN THE DETECTOR
    #SINGLE MODEL
    ISOFOREST_MODELS = []
    # detector = IsolationForest(random_state=2023)
    # detector = detector.fit(features_imputed)

    if n_dimensions == 1:
        model = IsolationForest(random_state=2023)
        model.fit(features_imputed)
        ISOFOREST_MODELS.append(model)
        num_features_per_dim = num_features_per_dim = features_imputed.shape[1] // n_dimensions #not assuming 4 dimensions
        print("num_features_per_dim", num_features_per_dim)

    else:
        # ONE MODEL PER DIMENSION
        num_features_per_dim = num_features_per_dim = features_imputed.shape[1] // n_dimensions #not assuming 4 dimensions
        for dim in range(n_dimensions):
            start_idx = dim * num_features_per_dim
            end_idx = (dim + 1) * num_features_per_dim
            features_imputed_dim = features_imputed[:, start_idx:end_idx]
            model = IsolationForest(random_state=2023)
            model.fit(features_imputed_dim)
            ISOFOREST_MODELS.append(model)

    detector = ISOFOREST_MODELS

    print("DETECTOR FITTED")

    return detector, imputer, num_features_per_dim


def test_detector_TF(observations, 
                     actions, 
                     detector, 
                     imputer, 
                     batch_size,
                     num_features_per_dim,
                     n_dimensions):
    
    #generate a list of test data
    test_data = preprocess_data(list_data = [observations], 
                                batch_size = batch_size)
    
    settings_efficient = settings.EfficientFCParameters()

    all_features_test = []

    for episode in test_data:
        batch_ctr = 0
            
        features_test = []
        
        for batch in episode:
            X = extract_features_from_multidim_batch(batch=batch, settings=settings_efficient)
            features_test.append(X)
            
            if batch_ctr % 5 == 0:
                print("Batch: ", batch_ctr)
            
            batch_ctr += 1
            
        features_test = np.vstack(features_test)
        
        # Impute missing values
        features_imputed_test = imputer.transform(features_test)
        all_features_test.append(features_imputed_test)

    print("TEST DATA PROCESSING FINISHED")

    for episode_idx, episode in enumerate(all_features_test):
        anom_scores = []

        #ONE MODEL FOR EACH DIM
        #-1 SINCE WE DON'T WANT TO INCLUDE THE LAST STEP
        for i in range(episode.shape[0] - 1):
            feats = episode[i,:]
            anomaly_scores_dim = []
            for dim in range(n_dimensions):
                start_idx = dim * num_features_per_dim
                end_idx = (dim + 1) * num_features_per_dim
                feats_dim = feats[start_idx:end_idx].reshape(1, -1)
                anomaly_scores_dim.append(-1 * detector[dim].decision_function(feats_dim)[0])
            anomaly_score = np.mean(anomaly_scores_dim)
            
            #OLD:
            # for _ in range(batch_size):
            #     anom_scores.append(anomaly_score)
            
            print("anomaly_score", anomaly_score)
            #NEW
            # If this is the first window of the episode
            if i == 0:
                # Append the initial score for the first 10 steps
                for _ in range(batch_size):
                    anom_scores.append(anomaly_score)
            else:
                anom_scores.append(anomaly_score)

            #anom_scores.append(anomaly_score)

    return anom_scores
#=== END: TF DETECTORS ====#

#===MAIN LOOP====#
if __name__ == "__main__":
    args = parse_args()
    #run_name = f"{args.test_env_id}__{args.seed}"
    print("args.seed", args.seed)

    #set seeds
    if args.seed != None:
        #torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        #np
        np.random.seed(args.seed)

        #R
        # robjects.r('set.seed(' + str(args.seed) + ')')

    if args.haider_env == True:
        print("TESTING ON ENVS FROM HAIDER ET AL 2023")
        train_test_haider_detector(args)
        exit()

    else:
        print("TESTING ON NOT FROM HAIDER ET AL 2023")
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    #Different procedure for CPD detectors
    #No need to define detector_class for CPD
    CPD_Detectors = ["ocd", "Chan", "Mei", "XS"]
    TF_Detectors = ["TF_ISOFOREST_Detector"]

    if args.detector_name not in CPD_Detectors and args.detector_name not in TF_Detectors:
        detector_class = globals()[args.detector_name]
        print("Testing on one of the detectors from Haider et al (2023)")

    elif (args.detector_name in CPD_Detectors):
        print("Testing on one of the CPD detectors")

    elif (args.detector_name in TF_Detectors):
        print("Testing on one of the TF detectors")
    else:
        print("Detector not found!")
    
    print("")
    print("DEVICE: ", device)
    print("")

    #initialize the train environment
    #train_injection_time = 0 by default, so the whole episode runs on the test params.
    # therefore, test params here should be the params of the train env of the detector
    options_train = {
        "test_mod_corr_noise": args.train_env_noise_corr,
        "test_noise_strength": args.train_noise_strength,
        "injection_time": args.train_injection_time}
    
    train_env = make_env(env_id = args.train_env_id, options = options_train, seed = args.seed)

    #TRAIN DETECTOR
    #If it's a CPD Detector, we have to train in with each test episode, so we skip training for now
    #else: train it now
    if args.detector_name in CPD_Detectors:
        print("")
        print("DETECTOR TYPE: CPD")

        #ELSE:
        p = train_env.observation_space.shape[0]
        print("P: ", p)
        #ocd cannot deal with 1-dim obs, set to 2, and duplicate the obs
        if p == 1:
            p = 2
        else:
            pass
        print("P: ", p)

        #adjust patience by the approximate length of the episode
        #patience = "average run length under the null" (Chen, 2020)
        if args.test_noise_strength <= 1.5:
            patience = args.max_steps_per_episode

        elif 1.5 < args.test_noise_strength <= 5.0:
            patience = 150

        else:
            patience = 100


        thresh = "MC"
        MC_reps = 100
        detector = ocd.ChangepointDetector(dim=p, 
                                           method=args.detector_name, 
                                           beta=1, 
                                           patience = patience,
                                           MC_reps = MC_reps,
                                           thresh=thresh)
    
    elif args.detector_name in TF_Detectors:
        #IMP: SPECIFY BATCH SIZE HERE
        n_dimensions = train_env.observation_space.shape[0]
        print("NUM DIMS: ", n_dimensions)

        batch_size = 10

        #prepare data for training
        detector, imputer, num_features_per_dim = train_detector_TF(args = args,
                                                                    batch_size=batch_size, n_dimensions = n_dimensions)
        print("=== TF DETECTOR TRAINED! ===\n")

    else:
        print("")
        print("DETECTOR TYPE: Haider et al 2023")
        print(" ")
        print("=== BEGIN TRAINING DETECTOR ===")
        detector = train_detector(
            args,
            env = train_env,
            num_train_episodes = args.num_train_episodes,
            data_path = args.train_data_path,
            detector_class = detector_class)
        print("=== DETECTOR TRAINED! ===\n")

        # Save the detector
        if not os.path.exists(os.path.dirname(args.detector_path)):
            args.detector_path = os.path.join(
                "data",
                "detectors",
                args.train_env_id,
                args.detector_name,
                f"{args.num_train_episodes}_ep",
                generate_hash(),
                "model.pth")
            
        parent_dir = os.path.dirname(os.getcwd())
        full_data_path = os.path.join(parent_dir, args.detector_path)
        os.makedirs(os.path.dirname(full_data_path), exist_ok=True)
        detector.save(full_data_path)
        print("")
        print("=== DETECTOR SAVED! ===\n")

    #TEST DETECTOR
    print("=== BEGIN TESTING DETECTOR ===")
    results_dict = {}
    y_scores = []
    y_true = []
    ep_rewards_modified = []
    
    #LINAS: add for testing
    list_obs = []
    list_injection_times = []

    #initialize the policy
    #train_env is used here to make sure the dimensions of train & test envs are the same
    agent = Agent(train_env).to(device)

    #load the same policy on both agents
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

    for episode in range(1, args.num_test_episodes + 1):

        #initialize the environment
        #injection time = time at which we inject the anomaly/noise into the env
        #the stronger the noise, the shorter the episode: adjust injection time
        if args.test_noise_strength <= 1.5:
            injection_time = np.random.randint(5, 200-5)
        elif 1.5 < args.test_noise_strength <= 5.0:
            injection_time = np.random.randint(5, 150-5)
        else:
            injection_time = np.random.randint(5, 100-5)

        options_test = {
            "train_mod_corr_noise": args.train_env_noise_corr,
            "train_noise_strength": args.train_noise_strength,
            "test_mod_corr_noise": args.test_env_noise_corr,
            "test_noise_strength": args.test_noise_strength,
            "injection_time": injection_time}
        
        test_env = make_env(env_id = args.test_env_id, options = options_test, seed = args.seed)

        #get the rollouts
        obs, acts, rewards, dones = policy_rollout(env=test_env, policy=agent)
        ep_rewards_modified.append(np.sum(rewards))

        #Generate anomaly scores
        if args.detector_name in CPD_Detectors:
            anom_scores, detector = train_test_detector_CPD(args = args, 
                                    detector = detector, 
                                    injection_time = injection_time,
                                    observations_test = obs,
                                    actions_test = acts)
            #reset the detector for the next episode
            #detector = ocd.reset(detector)

        elif args.detector_name in TF_Detectors:
            anom_scores = test_detector_TF(detector = detector, 
                                           observations = obs, 
                                           actions = acts,
                                           imputer = imputer,
                                           batch_size = batch_size, 
                                           num_features_per_dim = num_features_per_dim, n_dimensions = n_dimensions)

        else:
            anom_scores = detector.predict_scores(obs, acts)

        #get the real anomaly occurence vector
        anom_occurrence = [0 if i < injection_time else 1 for i in range(len(anom_scores))]

        #LINAS: add for testing
        if injection_time >= 5 and (len(anom_scores) - injection_time >= 5):
            # list_obs.append(obs)
            # list_injection_times.append(injection_time)
            pass

        print("Episode: ", episode)
        print("Test injection time: ", injection_time)
        print("len(anom_scores): ", len(anom_scores))
        print("len(anom_occurrence): ", len(anom_occurrence))
        print("")

        #only add the scores if the injection time is less than the length of the episode (otherwise all scores are 0)
        if injection_time < len(anom_scores):
            #Follow Haider et al
            y_scores.extend(anom_scores)
            y_true.extend(anom_occurrence)

        else:
            print("NOTE: injection time is greater than the length of the anom_scores list, so we skip the previous episode")
            pass

    #AFTER THE WHOLE LOOP:
    auroc, ap, fpr95, tpr95 = eval_metrics(y_scores, y_true)
    results_dict = {
        "reward": round(np.mean(ep_rewards_modified), 2),
        "auroc": round(auroc, 2),
        "fpr95": round(fpr95, 2),
        "tpr95": round(tpr95, 2),
        "ap": round(ap, 2)}

    print("=== TESTING COMPLETE ===")
    print("Detector:", args.detector_name)
    print("train_env", args.train_env_id)
    print("test_env", args.test_env_id)
    print("num train episodes:", args.num_train_episodes)
    print("train_env_noise_corr", args.train_env_noise_corr)
    print("train_noise_strength", args.train_noise_strength)
    print("test_env_noise_corr", args.test_env_noise_corr)
    print("test_noise_strength", args.test_noise_strength)
    print("")
    print("RESULTS:")
    print(results_dict)