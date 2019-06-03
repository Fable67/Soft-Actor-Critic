import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Hyperparameters import DEVICE


class SoftQNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size):
        super(SoftQNetwork, self).__init__()

        self.ih = nn.Linear(obs_size + action_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
        self.ho = nn.Linear(hidden_size, 1)

        self.ho.weight.data.uniform_(-3e-3, 3e-3)
        self.ho.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        o = torch.cat([state, action], 1)
        o = F.relu(self.ih(o))
        o = F.relu(self.hh(o))
        o = self.ho(o)
        return o

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class PolicyNetwork(nn.Module):

    def __init__(self, obs_size, action_size,
                 hidden_size, log_std_min, log_std_max):
        super(PolicyNetwork, self).__init__()

        self.lsmin = log_std_min
        self.lsmax = log_std_max

        self.ih = nn.Linear(obs_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
        self.hm = nn.Linear(hidden_size, action_size)
        self.hl = nn.Linear(hidden_size, action_size)

        self.hm.weight.data.uniform_(-3e-3, 3e-3)
        self.hm.bias.data.uniform_(-3e-3, 3e-3)
        self.hl.weight.data.uniform_(-3e-3, 3e-3)
        self.hl.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        h = F.relu(self.ih(state))
        h = F.relu(self.hh(h))
        mean = self.hm(h)
        log_std = self.hl(h)
        log_std = torch.clamp(log_std, self.lsmin, self.lsmax)

        return mean, log_std

    def forward_action(self, state, deterministic=False):
        # Get mean and std
        state = torch.FloatTensor(state).unsqueeze(
            0).to(DEVICE, non_blocking=True)
        mean, log_std = self.forward(state)
        if not deterministic:
            std = log_std.exp()
        else:
            std = torch.zeros(log_std.size(), device=DEVICE)

        #normal = Normal(mean, std)
        #z = normal.sample()
        z = (
            mean + std *
            Normal(
                torch.zeros(mean.size(), device=DEVICE),
                torch.ones(std.size(), device=DEVICE)
            ).sample()
        )
        z.requires_grad_()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        #z = normal.sample().detach()
        z = (
            mean + std *
            Normal(
                torch.zeros(mean.size(), device=DEVICE),
                torch.ones(std.size(), device=DEVICE)
            ).sample()
        )
        z.requires_grad_()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
