"""
ADD INJECTION TIME PARAMETER

Create modified envs and test them

This file provides the environments with modified noise.
It also tests these environments using time series analysis in the main() function.

The purposes of this file is to be imported anywhere where the environments are needed.
"""

import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import logger, spaces
import math
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from typing import Optional
import pickle
import matplotlib.pyplot as plt


#seed
np.random.seed(2023)

#OLD
# def normalized_moving_average_process(coeffs, T):
#     """
#     https://en.wikipedia.org/wiki/Moving-average_model
#     Do not confuse with moving average!
#     :param coeffs: number of coeffs determines correlation cut-off
#     :param T: length of timeseries
#     :return:
#     """
#     ar1 = np.array([1])  # we must include the zero-lag coefficient of 1
#     ma1 = np.array([1]+list(coeffs))
#     MA_object1 = ArmaProcess(ar1, ma1)
#     norm = (ma1**2).sum()**0.5  # ensure variance is the same as for
#     return MA_object1.generate_sample(nsample=T) / norm


#NEW:OLD
# def normalized_moving_average_process(coeffs, T):
# #     """
# #     https://en.wikipedia.org/wiki/Moving-average_model
# #     Do not confuse with moving average!
# #     :param coeffs: number of coeffs determines correlation cut-off
# #     :param T: length of timeseries
# #     :return:
# #     """
# #     #OLD
# #     # ar1 = np.array([1])  # we must include the zero-lag coefficient of 1
# #     # ma1 = np.array([1]+list(coeffs))
# #     # MA_object1 = ArmaProcess(ar1, ma1)
# #     # norm = (ma1**2).sum()**0.5  # ensure variance is the same as for
# #     # normalized_noise = MA_object1.generate_sample(nsample=T) / norm

#     #NEW: BEFORE PROPER NORMALIZATION
#     coeffs = [-coeff for coeff in coeffs]
#     # # AR(1) coefficient for 1-step autocorrelation
#     ar1 = np.array([1] + list(coeffs)) 
#     # # # Define the MA coefficients
#     ma1 = np.array([1])
#     MA_object1 = ArmaProcess(ar1, ma1)
#     # #NEW NORM
#     norm = (ar1**2).sum()**0.5  # ensure variance is the same as for
#     normalized_noise = MA_object1.generate_sample(nsample=T) / norm
    
#     return normalized_noise


#NEW NEW NEW
def normalized_moving_average_process(coeffs, T):
    """
    :param coeffs: number of coeffs determines correlation cut-off
    :param T: length of timeseries
    :return:
    """

    coeffs = [-coeff for coeff in coeffs]
    #AR(1) coefficient for 1-step autocorrelation
    ar1 = np.array([1] + list(coeffs)) 
    #Define the MA coefficients
    ma1 = np.array([1])
    MA_object1 = ArmaProcess(ar1, ma1)
    
    #normalize
    noise = MA_object1.generate_sample(nsample=T)
    normalized_noise = noise / np.std(noise)
    return normalized_noise

class IMANSCartpoleEnv(CartPoleEnv):
    """

    This environment noises the state transition dynamics with a noise vector
    that has been drawn from a Independent Moving Average Process at the beginning of the episode.

    NOTE: This environment is an MDP.
    """
    
    def __init__(self, 
                #  mod_noise_std=(0.6, 0.7, 0.1, 0.65), #old
                #mod_noise_std=(1.0, 1.0, 1.0, 1.0),
                 #mod_noise_std=(0.25, 0.45, 0.06, 0.5) * 0.025
                 mod_noise_std = tuple(0.025 * element for element in (0.25, 0.45, 0.06, 0.5)),
                 train_mod_corr_noise=(0.0,0.0), 
                 train_noise_strength=1.0, 
                 test_mod_corr_noise=(0.0,0.0),
                 test_noise_strength=1.0,
                 ep_length=200, 
                 render_mode: Optional[str] = None,
                 step_counter = 0,
                 injection_time = 0):

        super().__init__(render_mode=render_mode)
        self.mod_noise_std = mod_noise_std

        self.train_mod_corr_noise = train_mod_corr_noise
        self.train_noise_strength = train_noise_strength
        self.test_mod_corr_noise = test_mod_corr_noise
        self.test_noise_strength = test_noise_strength

        self.ep_length = ep_length

        self.step_counter = step_counter
        self.injection_time = injection_time
        
        return

    def get_injection_time(self):
        return self.injection_time
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        retvals = super().reset(seed=seed, options=options)

        # generate the noise variates for the whole episode in advance
        T = self.ep_length + 1
        obs_dim = self.observation_space.shape[0]
        
        self.train_state_noise_vec = np.stack([normalized_moving_average_process(self.train_mod_corr_noise, T) for _ in range(obs_dim)])
        self.test_state_noise_vec = np.stack([normalized_moving_average_process(self.test_mod_corr_noise, T) for _ in range(obs_dim)])
        
        self.train_noise_step_ctr = 0
        self.test_noise_step_ctr = 0
        self.step_counter = 0
        
        # print("retvals: ", retvals)

        if self.injection_time == 0:
            state_noise = self.test_state_noise_vec[:, self.test_noise_step_ctr] * self.mod_noise_std * self.test_noise_strength
            # state_noise = self.test_state_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength / self.mod_noise_std
            mod_retvals_0 = retvals[0] + state_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.test_noise_step_ctr += 1

        else:
            state_noise = self.train_state_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength * self.mod_noise_std
            # state_noise = self.train_state_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength / self.mod_noise_std
            
            mod_retvals_0 = retvals[0] + state_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.train_noise_step_ctr += 1

        #return retvals
        return mod_retvals

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        #print("before noise: state: ", self.state)

        #apply noise to state
        #if the environment is not yet in injection time, apply train noise
        if self.step_counter < self.injection_time:
            #Before reaching injection time: inject train noise
            #first: slice a 4 dimensional vector of noise
            state_noise = self.train_state_noise_vec[:, self.train_noise_step_ctr] * self.mod_noise_std * self.train_noise_strength
            # state_noise = self.train_state_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength / self.mod_noise_std

            #then apply noise to the state
            self.state = tuple(val + noise for val, noise in zip(self.state, state_noise))

            #track both counters
            self.train_noise_step_ctr += 1
            self.step_counter += 1
            
        else:
            #first: slice a 4 dimensional vector of TEST noise
            state_noise = self.test_state_noise_vec[:, self.test_noise_step_ctr] * self.mod_noise_std * self.test_noise_strength
            # state_noise = self.test_state_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength / self.mod_noise_std
    
            #apply the test noise 
            self.state = tuple(val + noise for val, noise in zip(self.state, state_noise))

            #only need to track the test noise counter
            self.test_noise_step_ctr += 1
            self.step_counter += 1
        
        # print("noise: ", state_noise)
        # print("after noise: state: ", self.state)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0

        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

            self.train_noise_step_ctr = 0
            self.test_noise_step_ctr = 0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


#original
class IMANOCartpoleEnv(CartPoleEnv):
    """
    This environment noises the observations with a noise vector
    that has been drawn from ARMA Process at the beginning of the episode.

    NOTE: This environment is a POMDP.
    """

    def __init__(self, 
                 #mod_noise_std=(1.0, 1.0, 1.0, 1.0), #for testing
                 #mod_noise_std=(0.25, 0.35, 0.1, 0.2), #old
                 #mod_noise_std=(0.25, 0.4, 0.05, 0.3), #new1
                #  mod_noise_std = tuple(0.25 * element for element in (0.5, 0.5, 0.25, 0.4)),
                 mod_noise_std = tuple(0.25 * element for element in (0.75, 0.5, 0.25, 0.4)),
                 train_mod_corr_noise=(0.0,0.0), 
                 train_noise_strength=1.0, 
                 test_mod_corr_noise=(0.0,0.0),
                 test_noise_strength=1.0,
                 ep_length=200, 
                 render_mode: Optional[str] = None,
                 step_counter = 0,
                 injection_time = 0):
        
        super().__init__(render_mode=render_mode)
        self.mod_noise_std = mod_noise_std

        self.train_mod_corr_noise = train_mod_corr_noise
        self.train_noise_strength = train_noise_strength
        self.test_mod_corr_noise = test_mod_corr_noise
        self.test_noise_strength = test_noise_strength

        self.ep_length = ep_length

        self.step_counter = step_counter
        self.injection_time = injection_time

        return

    def get_injection_time(self):
        return self.injection_time
        
    def get_episode_length(self):
        return self.ep_length
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        retvals = super().reset(seed=seed, options=options)

        # generate the noise variates for the whole episode in advance
        T = self.ep_length + 1
        obs_dim = self.observation_space.shape[0]

        self.train_obs_noise_vec = np.stack([normalized_moving_average_process(self.train_mod_corr_noise, T) for _ in range(obs_dim)])
        self.test_obs_noise_vec = np.stack([normalized_moving_average_process(self.test_mod_corr_noise, T) for _ in range(obs_dim)])
        
        self.train_noise_step_ctr = 0
        self.test_noise_step_ctr = 0
        self.step_counter = 0

        #if injection_time is 0, inject test noise throughout the episode
        if self.injection_time == 0:
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength * self.mod_noise_std 
            # obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr]  * self.test_noise_strength / self.mod_noise_std

            mod_retvals_0 = retvals[0] + obs_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.test_noise_step_ctr += 1
            #print("USING TEST NOISE")

        else:
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.mod_noise_std * self.train_noise_strength
            # obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength / self.mod_noise_std  

            mod_retvals_0 = retvals[0] + obs_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.train_noise_step_ctr += 1
            #print("USING TRAIN NOISE")
        
        # print("obs_noise: ", obs_noise)
        # print("mod_retvals: ", mod_retvals)
        # print("test_noise_strength: ", self.test_noise_strength)
        # print("train_noise_strength: ", self.train_noise_strength)
        #return retvals
        # print("obs_noise/retvals[0]", obs_noise/retvals[0])

        return mod_retvals
    

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        # ADD NOISE STRENGTH PARAMETER: 
        #slice a 4 dimensional vector of noise
        if self.step_counter < self.injection_time:
            #Before reaching injection time: inject train noise
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.mod_noise_std * self.train_noise_strength
            # obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength / self.mod_noise_std 

            obs = tuple(val + noise for val, noise in zip(self.state, obs_noise))
            
            self.train_noise_step_ctr += 1
            self.step_counter += 1
            #print("USING TRAIN NOISE")

        else:
            # When reached injection time: inject test noise
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.mod_noise_std * self.test_noise_strength
            # obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength / self.mod_noise_std

            obs = tuple(val + noise for val, noise in zip(self.state, obs_noise))

            self.test_noise_step_ctr += 1
            #print("USING TEST NOISE")

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0

        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

            self.train_noise_step_ctr = 0
            self.test_noise_step_ctr = 0
            
        if self.render_mode == "human":
            self.render()

        #print("obs_noise/self.state", obs_noise/self.state)

        return np.array(obs, dtype=np.float32), reward, terminated, False, {}

class TimeSeriesEnv(CartPoleEnv):
    """
    Time series environment that only returns noise. Noise is drawn from a Independent Moving Average Process at the beginning of the episode.
    The environment is built on CartPole-v0, so it could be used in the same train/test loop as other cartpole envs. 
    """

    def __init__(self, 
                 train_mod_corr_noise=(0.0,0.0), 
                 train_noise_strength=1.0, 
                 test_mod_corr_noise=(1.0,0.0),
                 test_noise_strength=1.0,
                 ep_length=200, 
                 render_mode: Optional[str] = None,
                 step_counter = 0,
                 injection_time = 0):
        
        super().__init__(render_mode=render_mode)

        self.train_mod_corr_noise = train_mod_corr_noise
        self.train_noise_strength = train_noise_strength
        self.test_mod_corr_noise = test_mod_corr_noise
        self.test_noise_strength = test_noise_strength

        self.ep_length = ep_length

        self.step_counter = step_counter
        self.injection_time = injection_time

        #reset observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )

        return
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        retvals = super().reset(seed=seed, options=options)

        # generate the noise variates for the whole episode in advance
        T = self.ep_length + 1
        obs_dim = self.observation_space.shape[0]
        self.train_obs_noise_vec = np.stack([normalized_moving_average_process(self.train_mod_corr_noise, T) for _ in range(obs_dim)])
        self.test_obs_noise_vec = np.stack([normalized_moving_average_process(self.test_mod_corr_noise, T) for _ in range(obs_dim)])
        
        self.train_noise_step_ctr = 0
        self.test_noise_step_ctr = 0
        self.step_counter = 0
        

        #if injection_time is 0, inject test noise throughout the episode
        if self.injection_time == 0:
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength

            mod_retvals_0 = obs_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.test_noise_step_ctr += 1

        else:
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength

            mod_retvals_0 = obs_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.train_noise_step_ctr += 1
        
        return mod_retvals
    

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        #step simply returns noise: correlated or not
        if self.step_counter < self.injection_time:
            #Before reaching injection time: inject train noise
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength
            self.state = obs_noise

            self.train_noise_step_ctr += 1
            self.step_counter += 1

        else:
            # When reached injection time: inject test noise
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength
            self.state = obs_noise

            self.test_noise_step_ctr += 1
            self.step_counter += 1

        terminated = False
        if not terminated:
            reward = 1.0
        
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


class TestModCartpoleEnv(CartPoleEnv):
    """
    NOTE: DO NOT USE THIS ENV: Needs reviewing, may have a bug
    INSTEAD: Use TimeSeriesEnv-v0
    Independent Moving Average Noise States
    This is a one-dimensional version of the IMANS implementation: 
    The only output it produces is noise, which can be either correlated or not. 
    Noise is drawn from a Independent Moving Average Process at the beginning of the episode.

    """

    def __init__(self, 
                 mod_noise_std=(0.6, 0.7, 0.1, 0.65),
                 train_mod_corr_noise=(0.0,0.0), 
                 train_noise_strength=1.0, 
                 test_mod_corr_noise=(0.0,0.0),
                 test_noise_strength=1.0,
                 ep_length=200, 
                 render_mode: Optional[str] = None,
                 step_counter = 0,
                 injection_time = 0):

        super().__init__(render_mode=render_mode)
        self.mod_noise_std = mod_noise_std

        self.train_mod_corr_noise = train_mod_corr_noise
        self.train_noise_strength = train_noise_strength
        self.test_mod_corr_noise = test_mod_corr_noise
        self.test_noise_strength = test_noise_strength

        self.ep_length = ep_length

        self.step_counter = step_counter
        self.injection_time = injection_time

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),  # Set the length of the one-dimensional observation
            dtype=np.float32
        )
        return

    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None):
        
        super().reset()

        T = self.ep_length + 1
        obs_dim = 1
        
        self.train_state_noise_vec = np.stack([normalized_moving_average_process(self.train_mod_corr_noise, T) for _ in range(obs_dim)])

        self.test_state_noise_vec = np.stack([normalized_moving_average_process(self.test_mod_corr_noise, T) for _ in range(obs_dim)])

        self.step_counter = 0
        self.train_noise_step_ctr = 0
        self.test_noise_step_ctr = 0

        #If injection time is not specified, then the same level of noise is injected right from the start of the episode, throughout its duration
        if self.injection_time == 0:
            state_noise = self.test_state_noise_vec[:, self.test_noise_step_ctr]
            self.test_noise_step_ctr += 1
        
        else:
            state_noise = self.train_state_noise_vec[:, self.train_noise_step_ctr]
            self.train_noise_step_ctr += 1

        #temporary
        # with open("/Users/linasnasvytis/Desktop/Msc_thesis/illusory-attacks/data/tests/new_env/states.pkl", "wb") as file:
        #     pickle.dump(self.test_state_noise_vec, file)
        state_noise = np.array(state_noise, dtype=np.float32)
        return (state_noise, {})


    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        # print("INJECTION TIME: ", self.injection_time)
        # print("STEP COUNTER: ", self.step_counter)

        if self.step_counter < self.injection_time:
            # print("TRAIN NOISE STEP CTR: ", self.train_noise_step_ctr)
            state_noise = self.train_state_noise_vec[:, self.train_noise_step_ctr]
            self.state = tuple(state_noise)

            #track both counters
            self.train_noise_step_ctr += 1
            self.step_counter += 1
            # print("STATE: ", self.state)
            
        else:
            # print("TEST NOISE STEP CTR: ", self.test_noise_step_ctr)
            state_noise = self.test_state_noise_vec[:, self.test_noise_step_ctr]
            self.state = tuple(state_noise)

            self.test_noise_step_ctr += 1
            self.step_counter += 1
            # print("STATE: ", self.state)

        terminated = bool(self.step_counter > self.ep_length)
        reward = 1.0

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}



if __name__ == "__main__":
    """
    HOMEWORK:
    - Write a simple rollout loop for both environments using a uniform stochastic policy. 
    Use time series analysis to verify that both environments exhibit the 
    stochastic correlation length advertised in their observation transitions.
    This may be harder to the state dynamics noise example, but should work really well for the obs noise POMDP example. HINT: look at autocorrelation at different time lengths.
    - Interface these environments with the cleanrl_ppo training code.
    
    See https://www.gymlibrary.dev/ for a simple rollout loop
    
    mod_corr_noise is just a list of arbitrary length,
    mod_corr_noise=[0.0, 1.0] would imply only 2-step correlation.
    mod_corr_noise=[0.0] would imply no correlation
    and mod_corr_noise=[1.0] is only 1-step correlation
    
    adjust self.mod_noise_std < 1.0 to make sure the environment doesn't get too noisy!
    """

    # IMANO: noising agent observations, not the underlying states
    # one step correlation
    from cartpole import IMANOCartpoleEnv

    env_IMANO = IMANOCartpoleEnv(mod_corr_noise=[1.0],
                                 mod_noise_std=0.1)

    # Reset the environment to start a new episode
    observation = env_IMANO.reset()

    # Initialize the episode length counter
    episode_length = 0

    # Run the episode until it is terminated
    done = False

    # while not done:
    for i in range(5):
        # reset state: otherwise breaks
        # state = env.reset()

        # Sample an action (in this example, we are taking random actions)
        action = env_IMANO.action_space.sample()
        # action = 1

        # Perform the action and receive the new observation, reward, termination, and info
        new_observation, reward, terminated, truncated, info = env_IMANO.step(action)

        print("run: ", i)
        print("action", action)
        print("action shape", action.size)
        print("new obs", new_observation)
        print("new obs", new_observation.shape)
        print("===")

        # if terminated or truncated:
        #     observation, info = env.reset()

    # Close the environment
    env_IMANO.close()

    pass