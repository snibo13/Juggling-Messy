from __future__ import annotations

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

import gymnasium as gym
import gym_examples


class Policy_Network(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(input_dims, 8), nn.Tanh(), nn.Linear(8, 16), nn.Tanh()
        )

        self.policy_mean_net = nn.Sequential(nn.Linear(16, output_dims))

        # self.policy_sigma_net = nn.Sequential(nn.Linear(16, output_dims))
        self.policy_sigma = torch.tensor([1, 1]).to(device)

    def forward(self, x):
        shared = self.shared_net(x.float())
        policy_mean = self.policy_mean_net(shared)
        # policy_sigma = torch.log(1 + torch.exp(self.policy_sigma_net(shared)))

        return policy_mean, self.policy_sigma


class REINFORCE:
    def __init__(self, intput_dims, output_dims):
        self.lr = 1e-3
        self.gamma = 0.9
        self.eps = 1e-6

        self.probs = []
        self.probs_0 = []
        self.probs_1 = []
        self.rewards = []

        self.net = Policy_Network(intput_dims, output_dims)
        self.optimizer = torch.optim.Adam(
            list(self.net.parameters()) + [self.net.policy_sigma], lr=self.lr
        )

    def select_action(self, state):
        state = torch.tensor(np.array([state])).to(device)
        policy_mu, policy_sigma = self.net(state)

        policy_sigma = torch.exp(policy_sigma)
        p = MultivariateNormal(policy_mu, torch.diag(policy_sigma))

        action = p.sample()

        self.probs.append(p.log_prob(action))

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

env_name = "gym_examples/JugglingEnv-v0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if training:
    print("Training...")
    env = gym.make(env_name)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    obseration_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]

    total_num_episodes = int(2e3)

    reward_over_seeds = []
    for seed in [1, 2, 4, 8, 16]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        reinforce_j1 = REINFORCE(17, 2)
        reinforce_j1.net.to(device)
        # reinforce_j2 = REINFORCE(4, 1)
        # reinforce_j2.net.to(device)
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            # print("Episode: {}".format(episode))
            state, info = wrapped_env.reset(seed=seed)
            done = False

            while not done:
                action_j1 = reinforce_j1.select_action(state.flatten())
                # action_j2 = reinforce_j2.select_action(state.flatten()).cpu()
                state, reward, done, truncated, _ = wrapped_env.step(
                    np.array([action_j1.cpu()]).flatten()
                )
                reinforce_j1.rewards.append(reward)
                # reinforce_j2.rewards.append(reward)
                done = done or truncated
            reward_over_episodes.append(wrapped_env.return_queue[-1])
            reinforce_j1.update()
            # reinforce_j2.update()

            avg_reward = int(np.mean(reward_over_episodes))
            if episode % 100 == 0:
                print(
                    "Seed: {} Episode: {} Reward: {}".format(seed, episode, avg_reward)
                )
                # print("Weights: {}".format(reinforce_j1.net.state_dict()))
        reward_over_seeds.append(reward_over_episodes)

        print("Saving weights")
        filename = "{}-reinforce-{}.pth".format("Juggling", seed)
        print(filename)
        torch.save(reinforce_j1.net.state_dict(), filename)
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
reinforce = REINFORCE(17, 2)
reinforce.net.load_state_dict(torch.load("Juggling-reinforce-8.pth"))
reinforce.net.to(device)

for i in range(10):
    state, _ = env.reset()
    done = False
    while not done:
        env.render()
        action = reinforce.select_action(state)
        state, reward, done, truncated, _ = env.step(np.array([action.cpu()]).flatten())
        done = done or truncated
    print("Episode {} finished".format(i))
    print("Reward: {}".format(reward))
env.close()
