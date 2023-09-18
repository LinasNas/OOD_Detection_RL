# OOD_Detection_RL

This repository hosts the source code used for the experiments of the Msc Dissertation project, titled "Out-of-Distribution Detection in Reinforcement Learning: A Proposal for a Novel Experimental Framework and a New Detection Algorithm". 

To run the experiments, please follow these steps:

1. Install the requirements from `requirements.txt` file.
2. To reproduce the results of the experiment, please run scripts following the basic framework provided below:

```
python train_test_detector_main.py \
    --detector-name DETECTOR_NAME \
    --train-env-id TRAIN_ENV_ID \
    --test-env-id TEST_ENV_ID \
    --train-data-path TRAIN_DATA_PATH \
    --policy-path POLICY_PATH \
    --train-noise-strength TRAIN_NOISE_STRENGTH \
    --train-env-noise-corr TRAIN_ENV_NOISE_CORR \
    --test-noise-strength TEST_NOISE_STRENGTH \
    --test-env-noise-corr TEST_ENV_NOISE_CORR \
    --num-train-episodes NUM_TRAIN_EPISODES \
    --num-test-episodes NUM_TEST_EPISODES \
    --TF-sliding TF_SLIDING_FLAG \
    --seed SEED
```


### Parameters Description:
- `--detector-name DETECTOR_NAME`: Name of the detector algorithm to be used. Examples include: "TF_ISOFOREST", "ocd", "Chan".
- `--train-env-id TRAIN_ENV_ID`: ID of the environment used during training. Examples include "TimeSeriesEnv-v0" (ARTS), "IMANSCartpoleEnv-v0" (ARNS), "IMANOCartpoleEnv-v0" (ARNO)
- `--test-env-id TEST_ENV_ID`: ID of the environment used during testing. Examples include "TimeSeriesEnv-v0" (ARTS), "IMANSCartpoleEnv-v0" (ARNS), "IMANOCartpoleEnv-v0" (ARNO)
- `--train-data-path TRAIN_DATA_PATH`: Path to the training data file. 
- `--policy-path POLICY_PATH`: Path to the policy model file.
- `--train-noise-strength TRAIN_NOISE_STRENGTH`: Strength of noise during training (as a float).
- `--train-env-noise-corr TRAIN_ENV_NOISE_CORR`: Noise correlation during training. This is a tuple (e.g., `(0.0,0.0)`).
- `--test-noise-strength TEST_NOISE_STRENGTH`: Strength of noise during testing (as a float).
- `--test-env-noise-corr TEST_ENV_NOISE_CORR`: Noise correlation during testing. This is a tuple (e.g., `(0.95,)`).
- `--num-train-episodes NUM_TRAIN_EPISODES`: Number of episodes to be used during training.
- `--num-test-episodes NUM_TEST_EPISODES`: Number of episodes to be used during testing.
- `--seed SEED`: Seed for the random number generator to ensure reproducibility. Use 2023 to reproduce the exact results from the project. 










