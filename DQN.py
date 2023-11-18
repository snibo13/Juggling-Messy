import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Memory stuff
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    # Verified
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # Verified
    def __init__(self, n_obs, n_actions):
        super().__init__()

        self.layer1 = nn.Linear(n_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def select_action(state):
    # Verified
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * (
        math.exp(-1.0 * steps_done / EPS_DECAY)
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


def plot_durations(show_result=False):
    # Verified
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())


episode_durations = []


def optimize_model():
    # Verified
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        dtype=torch.bool,
        device=device,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    # Loss on transistions
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# Hyperparameters

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


TRAINING = False

if TRAINING:
    env = gym.make("CartPole-v1")
else:
    env = gym.make("CartPole-v1", render_mode="human")

n_actions = env.action_space.n
state, info = env.reset()
n_obs = len(state)

policy_net = DQN(n_obs, n_actions).to(device)
target_net = DQN(n_obs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())


if TRAINING:
    print("Training mode")
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, done, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = done or truncated

            if done:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, device=device, dtype=torch.float32
                ).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict.keys():
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break
    print("Complete")
    torch.save(policy_net.state_dict(), "cartpole.pth")
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
else:
    print("Loading weights")
    policy_net.load_state_dict(torch.load("cartpole.pth"))

    for _ in range(10):
        state, info = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        for t in count():
            action = policy_net(state).max(1)[1].view(1, 1)
            observation, reward, done, truncated, info = env.step(action.item())
            state = torch.tensor(
                observation, device=device, dtype=torch.float32
            ).unsqueeze(0)
            env.render()
            if done:
                break
    env.close()
    print("Done")
