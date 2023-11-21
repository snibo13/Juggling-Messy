import torch
import random
import numpy as np
import torch.nn as nn

from collections import deque

import gymnasium as gym
import gym_examples

import sys

import matplotlib.pyplot as plt
from tqdm import tqdm


class Actor(nn.Module):
    def __init__(self, num_obs, hidden_size, num_actions, lr=1e-4):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(num_obs, hidden_size[0])
        self.norm1 = nn.BatchNorm1d(hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], num_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        # x = self.norm1(x)
        x = torch.relu(self.linear2(x))
        # x = nn.BatchNorm1d(x.shape[1])(x)
        x = torch.tanh(self.linear3(x))  # Output joint torques in range -1, 1
        return x


class Critic(nn.Module):
    def __init__(self, num_obs, hidden_size, num_actions, lr=1e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_obs, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0] + num_actions, hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-2)

    def forward(self, state, action):
        # TODO: Try adding batch normalization
        x = torch.relu(self.linear1(state))
        # x = nn.BatchNorm1d(x.shape[1])(x)
        x = torch.relu(self.linear2(torch.cat([x, action], dim=1)))
        # x = nn.BatchNorm1d(x.shape[1])(x)
        x = self.linear3(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DDPG:
    def __init__(
        self,
        n_obs,
        n_act,
        hidden_dim,
        buffer_size,
        batch_size,
        actor_lr=1e-4,
        critic_lr=1e-3,
        tau=1e-2,
        gamma=0.99,
        device="cpu",
    ):
        self.actor = Actor(n_obs, hidden_dim, n_act, actor_lr).to(device)
        self.actor_target = Actor(n_obs, hidden_dim, n_act, actor_lr).to(device)
        self.critic = Critic(n_obs, hidden_dim, n_act, critic_lr).to(device)
        self.critic_target = Critic(n_obs, hidden_dim, n_act, critic_lr).to(device)

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.device = device
        self.update(1)

    def act(self, state, noise=0.0):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().detach().numpy()
        return np.clip(action + noise, -1, 1)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            np.array(actions), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            np.array(rewards), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            np.array(dones), dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Update critic
        self.critic.optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_next = torch.add(rewards, self.gamma * q_next * (1 - dones))

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, q_next)

        critic_loss.backward()
        self.critic.optimizer.step()

        # Update actor
        self.actor.optimizer.zero_grad()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update()

    def update(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def train(episodes=5e4, max_steps=5e2, continuous=False, show=False):
    if show:
        env = gym.make("gym_examples/BimanualEnv-v0", render_mode="human")
    else:
        env = gym.make("gym_examples/BimanualEnv-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    state, _ = env.reset()
    n_actions = env.action_space.shape[0]
    n_obs = len(state)

    agent = DDPG(
        n_obs,
        n_actions,
        hidden_dims,
        buffer_size=int(5e5),
        batch_size=1024,
        tau=1e-2,
        gamma=0.998,
        device=device,
    )

    if continuous:
        agent.actor.load_state_dict(
            torch.load("bimanual-actor.pth", strict=False, map_location=device)
        )
        agent.critic.load_state_dict(
            torch.load("bimanual-critic.pth", strict=False, map_location=device)
        )
        agent.actor.optimizer.load_state_dict(
            torch.load(
                "bimanual-actor-optimizer.pth",
                strict=False,
                map_location=device,
            )
        )
        agent.critic.optimizer.load_state_dict(
            torch.load(
                "bimanual-critic-optimizer.pth", strict=False, map_location=device
            )
        )
        agent.actor_target.load_state_dict(
            torch.load("bimanual-actor_target.pth", strict=False, map_location=device)
        )
        agent.critic_target.load_state_dict(
            torch.load("bimanual-critic_target.pth", strict=False, map_location=device)
        )

    rewards = []
    rs = []
    catches = []
    catch_count = 0
    throw_count = 0
    stopping_count = 0
    for episode in tqdm(range(1, int(episodes + 1))):
        state, _ = env.reset()
        for step in range(int(max_steps)):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            if info["caught"]:
                catch_count += 1
            if info["thrown"]:
                throw_count += 1
            done = done or truncated
            rs.append(reward)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            agent.update()
            state = next_state
            if done:
                break
        if episode % 100 == 0:
            avg_reward = np.mean(rs)
            print(
                "Episode: {} Reward: {} Catches: {} Throws: {}".format(
                    episode, avg_reward, catch_count, throw_count
                )
            )
            if catch_count == 100:
                stopping_count += 1
            catches.append(catch_count)
            catch_count = 0
            throw_count = 0
            rs = []
            rewards.append(avg_reward)
            print("Checkpoint")
            torch.save(agent.actor.state_dict(), "bimanual-actor.pth")
            torch.save(agent.actor_target.state_dict(), "bimanual-actor_target.pth")
            torch.save(agent.critic.state_dict(), "bimanual-critic.pth")
            torch.save(agent.critic_target.state_dict(), "bimanual-critic_target.pth")
            torch.save(
                agent.actor.optimizer.state_dict(), "bimanual-actor-optimizer.pth"
            )
            torch.save(
                agent.critic.optimizer.state_dict(), "bimanual-critic-optimizer.pth"
            )

    plt.plot(catches)
    plt.show()
    plt.plot(np.cumsum(rewards) / np.arange(1, len(rewards) + 1))
    plt.show()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def evaluate():
    env = gym.make("gym_examples/BimanualEnv-v0", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(n_obs, hidden_dims, n_actions).to(device)
    critic = Critic(n_obs, hidden_dims, n_actions).to(device)

    actor.load_state_dict(
        torch.load("bimanual-actor.pth", map_location=device), strict=True
    )
    critic.load_state_dict(
        torch.load("bimanual-critic.pth", map_location=device), strict=True
    )

    state, _ = env.reset()

    for episode in range(10):
        state, _ = env.reset()
        for step in range(1000):
            action = actor(torch.tensor(state, dtype=torch.float, device=device))
            action = np.clip(action.cpu().detach().numpy(), -1, 1)
            next_state, reward, done, truncated, info = env.step(action)
            env.render()
            state = next_state
            if done:
                break

    exit()


if __name__ == "__main__":
    # Read environment parameters
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    hidden_dims = [512, 512]
    print(sys.argv)
    num_episodes = 1e4
    max_steps = 800
    if sys.argv[1] == "train":
        if len(sys.argv) == 4:
            train(
                episodes=num_episodes,
                max_steps=max_steps,
                continuous=sys.argv[2] == "c",
                show=sys.argv[3] == "show",
            )
        else:
            train()
    else:
        evaluate()
