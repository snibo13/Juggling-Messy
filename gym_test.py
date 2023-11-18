import gymnasium as gym
import gym_examples

import numpy as np

# Create a new environment
env = gym.make("gym_examples/JugglingEnv-v0", render_mode="human")

# Reset the environment
state = env.reset()
# Run the environment for 1000 steps
for _ in range(1000):
    # print(_)
    # Render the environment
    env.render()

    # Take a random action
    action = env.action_space.sample()

    # # Step the environment
    state, reward, done, truncated, info = env.step(action)

    # # Print the state, reward, done and info
    # print(state, reward, done, info)

    # # If the episode is done, reset the environment
    if done or truncated:
        state = env.reset()

    # assert not done

# Close the environment
env.close()
