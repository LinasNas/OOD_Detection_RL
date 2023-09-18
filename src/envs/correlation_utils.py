import numpy as np
from envs.cartpole import IMANSCartpoleEnv, IMANOCartpoleEnv
import statsmodels.api as sm
import matplotlib.pyplot as plt

def get_autocorrelations(env_name, mod_corr_noise, mod_noise_std, noise_strength, max_lag, num_episodes):
    """
    input:
    - env_name: string. "IMANSCartpoleEnv" or "IMANOCartpoleEnv"
    - mod_corr_noise: list of integers. [0.0] for env with uncorrelated noise, [1.0] for one step corr, 
    [0.0, 1.0] for only 2-step corr, [1.0, 1.0] for 1 and 2 step corr, etc
    - mod_noise_std: int. Note: values for IMANSCartpoleEnv should generally be lower IMANSCartpoleEnv
    than for IMANOCartpoleEnv, e.g. 1.0 works for IMANOCartpoleEnv, but will break IMANSCartpoleEnv
    default: 0.1 for IMANOCartpoleEnv, 0.01 for IMANSCartpoleEnv
    - max_lag: maximum autoc-correlation lag to consider. 
    
    output: all_autocorr, a list where each element is a np.array with autocorrelations for each dim
    to run an env 
    """
    
    #initialize env
    if env_name == "IMANOCartpoleEnv":
        env = IMANOCartpoleEnv(mod_corr_noise=mod_corr_noise,
                               mod_noise_std = mod_noise_std,
                               noise_strength=noise_strength)
        
    elif env_name == "IMANOCartpoleEnv":
        env = IMANSCartpoleEnv(mod_corr_noise=mod_corr_noise,
                               mod_noise_std = mod_noise_std,
                               noise_strength=noise_strength)

    else:
        print("ERROR: env_name can only be  'IMANSCartpoleEnv' or 'IMANOCartpoleEnv'")
        return
        
    # Counter for the number of steps in this episode
    step_count = 0
    #find dimensions of the observation space
    obs_space = env.observation_space
    obs_dimension = obs_space.shape
    obs_dimension = obs_dimension[0]
    
    # Initialize a list, which stores autocorrelations along each observation dimension
    episode_autocorr = []
    all_autocorr = []
    for dim in range(obs_dimension):
        all_autocorr.append(np.zeros((num_episodes, max_lag)))

    #run the loop
    for episode in range(num_episodes):
        # Reset the environment to start a new episode
        observation = env.reset()
        #added this, because otherwise it carries an empty dict
        observation = observation[0]

        # Initialize lists to store trajectories for each dimension of observation for this episode
        list_observations = []
        for dim in range(obs_dimension):
            list_observations.append([])

        actions = []
        rewards = []
        step_count = 0
    
        # Run the episode until it is terminated
        terminated = False
        while not terminated:
            
            # Sample an action (in this example, we are taking random actions)
            action = env.action_space.sample()
            
            # Perform the action and receive the new observation, reward, termination, and info
            new_observation, reward, terminated, truncated, info = env.step(action)

            #store observations from each dim in separate lists
            for dim in range(obs_dimension):
                list_observations[dim].append(observation[dim])
        
            actions.append(action)
            rewards.append(reward)
        
            # Increment the step count
            step_count += 1
            
            # Update the observation for the next step
            observation = new_observation

        #once an episode terminates: find the autocorrelations in each dimension
        #store them
        for dim in range(obs_dimension):
            trajectories = list_observations[dim]
            episode_autocorr = sm.tsa.acf(trajectories, nlags = max_lag - 1)
            all_autocorr[dim][episode, ] = episode_autocorr
    
        episode += 1
    
    # Close the environment
    env.close()

    # Print summary 
    print("ENVIRONMENT: ", env)
    print(" ")
    print("PARAMETERS:")
    print("mod_corr_noise:", mod_corr_noise)
    print("mod_noise_std:", mod_noise_std)
    print(" ")
    print("Number of dimensions in the state / observation: ", obs_dimension)
    print("Number of autocorrelation lags: ", max_lag)
    print(" ")
    print("AUTOCORRELATIONS in each dimension:")
    for dim in range(obs_dimension):
        final_autocorr = np.mean(all_autocorr[dim], axis=0)
        print("Dim. of observation: ", dim)
        print("Autocorrelations: ", final_autocorr)
        print(" ")
        
    return all_autocorr

def get_excess_autocorrelations(env_name, mod_corr_noise, mod_noise_std, noise_strength,max_lag, num_episodes):
    
    list_autocorr = get_autocorrelations(env_name = env_name, 
                                         mod_corr_noise = mod_corr_noise, 
                                         mod_noise_std = mod_noise_std, 
                                         max_lag = max_lag, 
                                         num_episodes = num_episodes,
                                         noise_strength = noise_strength)
    
    list_autocorr_BASELINE = get_autocorrelations(env_name = env_name, 
                                                  mod_corr_noise = [0.0], 
                                                  mod_noise_std = mod_noise_std, 
                                                  max_lag = max_lag, 
                                                  num_episodes = num_episodes, 
                                                  noise_strength=noise_strength)

    obs_dimension = len(list_autocorr)
    
    print("===EXCESS CORRELATION===")   
    for dim in range(obs_dimension):
        final_autocorr_BASELINE = np.mean(list_autocorr_BASELINE[dim], axis=0)
        final_autocorr = np.mean(list_autocorr[dim], axis=0)
        excess_autocorr = final_autocorr - final_autocorr_BASELINE
        
        print("Dim. of observation: ", dim)
        print("Excess autocorrelation: ", excess_autocorr)
        print(" ")
        
    return [list_autocorr, list_autocorr_BASELINE]

    
#===PLOTS===
def plot_autocorrelations(list_autocorr, max_lag):
    lags = np.arange(max_lag)
    
    obs_dimension = len(list_autocorr)
    
    for dim in range(obs_dimension):
        final_autocorr = np.mean(list_autocorr[dim], axis=0)
        
        # Plot the autocorrelation scores for lags
        plt.stem(lags, final_autocorr)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title(f"Dimension of state/observation: {dim} \n Autocorrelation for lags 0-{max_lag-1}")
        plt.grid(True)
        plt.show()

    return  


def plot_excess_autocorrelations(list_autocorr, list_autocorr_BASELINE, max_lag):
    lags = np.arange(max_lag)
    obs_dimension = len(list_autocorr)
    
    for dim in range(obs_dimension):
        final_autocorr = np.mean(list_autocorr[dim], axis=0)
        final_autocorr_BASELINE = np.mean(list_autocorr_BASELINE[dim], axis=0)
        excess_autocorr = final_autocorr - final_autocorr_BASELINE
        
        # Plot the autocorrelation scores for lags
        plt.stem(lags, excess_autocorr)
        plt.xlabel("Lag")
        plt.ylabel("Excess Autocorrelation")
        plt.title(f"Dimension of state/observation: {dim} \n Excess Autocorrelation for lags 0-{max_lag-1}")
        plt.grid(True)
        plt.show()

    return  