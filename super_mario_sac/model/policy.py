#
# File: super_mario_sac/model/policy.py
# Auth: Taken from: https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665
# Desc: The policy used in the SAC algorithm
#
################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """
    The network responsible for deriving a policy from a state
    """
    def __init__(self, feature_extractor, num_actions, hidden_size=32*6*6, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.backbone = feature_extractor

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, state):
        """
        Forward pass function
        :param state:
        :return:
        """
        # extract features
        x = self.backbone(state)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(self.device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()
        return action[0]
