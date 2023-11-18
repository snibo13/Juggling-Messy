from __future__ import annotations

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
import gym_examples


class Policy_Network(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(input_dims, 16), nn.Tanh(), nn.Linear(16, 32), nn.Tanh()
        )

        self.policy_mean_net = nn.Sequential(nn.Linear(32, output_dims))

        self.policy_sigma_net = nn.Sequential(nn.Linear(32, output_dims))

    def forward(self, x):
        shared = self.shared_net(x.float())

        policy_mean = self.policy_mean_net(shared)
        policy_sigma = torch.log(1 + torch.exp(self.policy_sigma_net(shared)))

        return policy_mean, policy_sigma


class REINFORCE:
    def __init__(self, intput_dims, output_dims):
        self.lr = 1e-3
        self.gamma = 0.9
        self.eps = 1e-6

        self.probs = []
        self.rewards = []

        self.net = Policy_Network(intput_dims, output_dims)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def select_action(self, state):
        print(state)
        state = torch.tensor(np.array([state]))
        policy_mu, policy_sigma = self.net(state)
        print(policy_mu, policy_sigma)
        p = Normal(policy_mu[0] + self.eps, policy_sigma[0] + self.eps)
        action = p.sample()
        prob = p.log_prob(action)

        self.probs.append(prob)
        return action

    def update(self):
        running_g = 0
        gs = []

        for R in self.rewards:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []


training = True

env_name = "Pendulum-v1"
env_name = "InvertedPendulum-v4"
# env_name = "gym_examples/JugglingEnv-v0"


if training:
    env = gym.make(env_name)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    obseration_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]

    total_num_episodes = int(4e3)

    reward_over_seeds = []
    for seed in [1]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        reinforce = REINFORCE(obseration_dims, action_dims)
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            state, info = wrapped_env.reset(seed=seed)
            done = False

            while not done:
                action = reinforce.select_action(state)
                state, reward, done, truncated, _ = wrapped_env.step(action)
                reinforce.rewards.append(reward)
                done = done or truncated
            reward_over_episodes.append(wrapped_env.return_queue[-1])
            reinforce.update()

            avg_reward = int(np.mean(reward_over_episodes))
            if episode % 100 == 0:
                print(
                    "Seed: {} Episode: {} Reward: {}".format(seed, episode, avg_reward)
                )
        reward_over_seeds.append(reward_over_episodes)

    print("Saving weights")
    torch.save(reinforce.net.state_dict(), "{}=reinforce.pth".format(env_name))
    print("Weights saved")
    rewards_to_plot = [
        [reward[0] for reward in rewards] for rewards in reward_over_seeds
    ]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for InvertedPendulum-v4"
    )
    plt.show()

env = gym.make(env_name, render_mode="human")
obseration_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]
trained_model = Policy_Network(obseration_dims, action_dims)
trained_model.load_state_dict(torch.load("reinforce.pth"))
trained_model.eval()
reinforce = REINFORCE(obseration_dims, action_dims)
for i in range(10):
    state, _ = env.reset()
    done = False
    while not done:
        env.render()
        action = reinforce.select_action(state)
        state, reward, done, truncated, _ = env.step(action.detach().numpy())
        done = done or truncated
    print("Episode {} finished".format(i))
    print("Reward: {}".format(reward))
env.close()
