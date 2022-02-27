#
# File: super_mario_ac/models/policy.py
# Auth: Taken from: https://github.com/uvipen/Super-mario-bros-A3C-pytorch/blob/master/src/model.py
# Desc: The policy used in the SAC algorithms
#
################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        # self._initialize_weights()

    def _initialize_weights(self):
        """
        Method to correctly initialize weights
        :return:
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        """
        Forward pass through the model
        :param x:  input
        :param hx: LSTM hidden state
        :param cx: LSTM cell state
        :return:
        """
        # run the forward pass though the shared backbone
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.flatten(x, start_dim=1)
        hx, cx = self.lstm(x, (hx, cx))

        # run the forward pass for the actor
        actor_x = self.actor_linear(hx)
        actor_out = F.softmax(actor_x, dim=1)

        # run the forward pass for the critic
        critic_out = self.critic_linear(hx)

        return actor_out, critic_out, hx, cx
