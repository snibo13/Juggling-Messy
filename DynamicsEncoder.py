# Encoder Decoder for Dynamics Model
# Takes in a series of states and actions

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

representation_size = 5
hidden_sizes = [64, 64, 32, 16]


class Encoder(nn.Module):
    def __init__(self, in_size, representation_size, hidden_sizes=[64, 64, 32, 16]):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(in_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], representation_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_size, representation_size, hidden_sizes=[64, 64, 32, 16]):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(representation_size, hidden_sizes[2])
        self.fc2 = nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.fc4 = nn.Linear(hidden_sizes[0], in_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class VAE(nn.Module):
    def __init__(self, in_size, representation_size, hidden_sizes=[64, 64, 32, 16]):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_size, representation_size, hidden_sizes)
        self.decoder = Decoder(in_size, representation_size, hidden_sizes)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def backward(self, loss):
        loss.backward()


class TrajectoryPredictor(nn.Module):
    # Predicts next position given representation and current position and velocity
    def __init__(self, n_obs, representation_size, hidden_sizes=[64, 64]):
        super(TrajectoryPredictor, self).__init__()

        self.fc1 = nn.Linear(representation_size + n_obs, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], n_obs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def backward(self, loss):
        loss.backward()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datas = [
    "mass_1.npy",
    "mass_2.npy",
    "mass_5.npy",
    "mass_10.npy",
    "mass_100.npy",
]
np.random.shuffle(datas)

a = np.load("mass_1.npy")
num_samples = 1000
num_steps = 60
loss_f = torch.nn.MSELoss()
representation_size = 10
hidden_sizes = [64, 32, 16]
n_obs = 13
window_size = 20
VAE = VAE(n_obs * window_size, representation_size, hidden_sizes).to(device)
optimizer = torch.optim.Adam(VAE.parameters(), lr=0.001)

losses = []

if False:
    for data in datas:
        print(data)
        data = np.load(data)
        for i in tqdm(range(num_samples)):
            sample = data[i, :, :]
            for s in range(window_size, num_steps):
                step = sample[s - window_size : s, :].flatten()

                if np.isclose(step[0], 0).all():
                    break

                step_tensor = torch.tensor(
                    step, dtype=torch.float32, requires_grad=True, device=device
                )

                f = VAE.forward(step_tensor)
                loss = loss_f(f, step_tensor)
                losses.append(loss.item())

                optimizer.zero_grad()
                with torch.no_grad():
                    VAE.backward(loss)

                optimizer.step()

    # Compute running average of losses
    running_losses = []
    avg_loss = 0
    for i in range(len(losses)):
        avg_loss += losses[i]
        if i % 100 == 0:
            running_losses.append(avg_loss / 100)
            avg_loss = 0
    plt.plot(running_losses)
    plt.show()

    torch.save(VAE.state_dict(), "VAE.pt")

if False:
    VAE.load_state_dict(torch.load("VAE.pt"))
    VAE.eval()

    # Test the model
    data = np.load("mass_1.npy")
    offset = 6000
    test_data = data[offset]
    test_data = test_data[30 - window_size : 30, :].flatten()
    # print(test_data.shape)
    tensor = torch.tensor(test_data, dtype=torch.float32, device=device)
    embed = VAE.encode(tensor).detach().cpu().numpy()
    # print(embed)

    TJ = TrajectoryPredictor(n_obs, representation_size, hidden_sizes).to(device)
    optimizer = torch.optim.Adam(TJ.parameters(), lr=0.001)

    losses = []
    for i in tqdm(range(2 * num_samples)):
        sample = data[i, :, :]
        for s in range(1, num_steps):
            step = sample[s - 1, :].flatten()

            if np.isclose(step, 0).all():
                break

            x = np.hstack((step, embed))
            # print(x)

            step_tensor = torch.tensor(
                x,
                dtype=torch.float32,
                requires_grad=True,
                device=device,
            )

            next_tensor = torch.tensor(
                sample[s, :], dtype=torch.float32, requires_grad=True, device=device
            )

            f = TJ.forward(step_tensor)
            loss = loss_f(f, next_tensor)
            losses.append(loss.item())

            optimizer.zero_grad()
            with torch.no_grad():
                TJ.backward(loss)

            optimizer.step()
    # Compute running average of losses
    print(losses[-1])
    running_losses = []
    avg_loss = 0
    for i in range(len(losses)):
        avg_loss += losses[i]
        if i % 100 == 0:
            running_losses.append(avg_loss / 100)
            avg_loss = 0
    plt.plot(running_losses)
    plt.show()

    torch.save(TJ.state_dict(), "TJ.pt")


if True:
    VAE.load_state_dict(torch.load("VAE.pt"))
    VAE.eval()

    # Test the model
    data = np.load("mass_33.npy")
    offset = 6000
    test_data = data[offset]
    test_data = test_data[30 - window_size : 30, :].flatten()
    # print(test_data.shape)
    tensor = torch.tensor(test_data, dtype=torch.float32, device=device)
    embed = VAE.encode(tensor).detach().cpu().numpy()
    # print(embed)

    TJ = TrajectoryPredictor(n_obs, representation_size, hidden_sizes).to(device)
    TJ.load_state_dict(torch.load("TJ.pt"))
    TJ.eval()

    losses = []
    for i in tqdm(range(2 * num_samples)):
        sample = data[i, :, :]
        for s in range(1, num_steps):
            step = sample[s - 1, :].flatten()

            if np.isclose(step, 0).all():
                break

            x = np.hstack((step, embed))
            # print(x)

            step_tensor = torch.tensor(
                x,
                dtype=torch.float32,
                requires_grad=True,
                device=device,
            )

            next_tensor = torch.tensor(
                sample[s, :], dtype=torch.float32, requires_grad=True, device=device
            )

            f = TJ.forward(step_tensor)
            loss = loss_f(f, next_tensor)
            losses.append(loss.item())

    # Compute running average of losses
    print(losses[-1])
    print(np.mean(losses))
    running_losses = []
    avg_loss = 0
    for i in range(len(losses)):
        avg_loss += losses[i]
        if i % 100 == 0:
            running_losses.append(avg_loss / 100)
            avg_loss = 0
    plt.plot(running_losses)
    plt.show()
