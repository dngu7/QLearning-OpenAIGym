# QLearning-OpenAIGym
Reinforcement Learning on CartPole Environment

# Algorithm Performance over 50 runs
The algorithm achieves >190 average reward on the 100th episode and zero degradation of performance after the 200th episode. Stable performance is achieved in 92.6 episodes which is defined as 10 consecutive episodes with >180 reward.

- Total Runs                     50
- Total Episodes per Run         1000
- Total Time                     85.4mins
- Avg Time per Run               1.71mins
- +190 Reward on 100th Episode   48/50 (96%)
- +190 Reward on 200th Episode   50/50 (100%)
- Stable Performance             50/50 (100%)
- Avg Stable Episode             92.6

# Techniques
- Single Layer Dense Neural Network Layer (Tanh, Adam Optimizer with 100 Hidden Units)
- Memory Storage and Training
- Prioritization of recent memory during training
- Dynamic Training Batch Sizes
- Dynamic Episilon
- Measurement of stability with consecutive wins

# Setup Requirements
Ensure Python3.6 and following packages are installed.
>tensorflow >=1.12.0
>numpy >= 1.16.1
>gym

# Run
python Neural_QTrain.py
