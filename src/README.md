# DEBUGGING: Running on Haider envs

Haider et al have uploaded the trained policy they use to generate rollouts, on which their detector is trained. The policy is found in ```../assets/policies/Haider/MJCartpole-v0/TD3/best_model.zip```

Issue: at the moment, loading the policy from ```stable_baselines3``` causes an issue related to observation spaces. 

To verify the issue
1. Make sure you have the requirements installed (or do so in the root directory, from ```requirements.txt```)

2. Run the following script:

```
python3 test_haider_envs.py
```

The issue you should get is:
```
ValueError: Observation spaces do not match: Box([-inf -inf -inf -inf], 
[inf inf inf inf], (4,), float64) != Box(-inf, inf, (4,), float64)
```

(Also, even though ```gymnasium==0.29.0``` is in ```requirements.txt```, you may need to reinstall it - Im not sure why)

3. For some help: the observation spaces were defined by me, to solve some conflicts when using gymnasium with Haider setup. 

The file where cartpole environment (with my changes) is defined is: ```mujoco_envs/base_envs/cartpole.py```
Originally, the file used was: ```mujoco_envs/base_envs/old_cartpole.py```

You can find the full original implementation of Haider envs in: https://github.com/FraunhoferIKS/pedm-ood/tree/main/mujoco_envs

4. 