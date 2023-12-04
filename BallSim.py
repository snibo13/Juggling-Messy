import gymnasium as gym
import gym_examples

import numpy as np
import time
import matplotlib.pyplot as plt

# Create a new environment
env = gym.make("gym_examples/BallEnv-v0")

sample_count = 10000
num_steps = 60

states = np.zeros([sample_count, num_steps, 13])
np.random.seed(0)
ball_positions = []
for i in range(sample_count):
    # Reset the environment
    state, _ = env.reset()
    # Run the environment for 1000 steps
    for step in range(num_steps):
        # print(_)
        # Render the environment

        # Take a random action
        # action = np.random.uniform(-1, 1, env.action_space.shape)
        # action = [-1, 0, 0]
        action = np.zeros(env.action_space.shape)

        # # Step the environment
        state, reward, done, truncated, info = env.step(action)

        # # Print the state, reward, done and info
        states[i, step, :] = state

        # # If the episode is done, reset the environment
        if done or truncated:
            state = env.reset()
            break

        # assert not done

    # Close the environment
env.close()

print(states.shape)

np.save("mass_33.npy", states)
