#
# File: super_mario_sac/model/value_network.py
# Auth: Taken from: https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665
# Desc: The value networks in the SAC algorithm
#
################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, feature_extractor, device):
        super(ValueNetwork, self).__init__()

        # use backbone
        self.backbone = feature_extractor

        # implement linear
        self.linear = nn.Sequential(nn.Linear(32 * 6 * 6, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(),
                                    nn.Linear(64, 1))

        # store device
        self.device = device

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        """
        Initialiser for weights, can initialise weights of all modules in a net, where net is a nn.Module.
        All weights in a net can be recursively initialised in its constructor via "self.apply(self.weights_init_)",
        where net is a nn.Module.
        :param m:
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward call of the network
        @param state: batch of states.
        """
        # run state through backbone
        x = self.backbone(state)

        # run extracted features through linear
        out = self.linear(x)

        return out


class SoftQNetwork(nn.Module):
    def __init__(self, feature_extractor, num_actions, device):
        super(SoftQNetwork, self).__init__()

        # use backbone
        self.backbone = feature_extractor

        # implement linear
        self.linear = nn.Sequential(nn.Linear((32 * 6 * 6) + num_actions, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(),
                                    nn.Linear(64, 1))

        # store device
        self.device = device

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        """
        Initialiser for weights, can initialise weights of all modules in a net, where net is a nn.Module.
        All weights in a net can be recursively initialised in its constructor via "self.apply(self.weights_init_)",
        where net is a nn.Module.
        :param m:
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        """
        Forward call of the network
        @param state: batch of states.
        """
        # run state through backbone
        x = self.backbone(state)

        # concat action to feature vector
        x = torch.cat([x, action], 1)

        # run extracted features through linear
        out = self.linear(x)

        return out

