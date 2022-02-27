import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FeatureExtractor(nn.Module):

    def __init__(self, input_channels):
        super(FeatureExtractor, self).__init__()

        # Convolutional Network to extract feature vector from game screen
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, state):
        """
        Forward method for the forward pass
        :param state: The screen at a specific point in time
        :return:
        """
        return self.cnn(state)
