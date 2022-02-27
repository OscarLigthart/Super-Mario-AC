import torch

import torch.nn as nn
import random

from .memory import ReplayMemory
from ..models.DQN import DQN


class Agent:
    """
    The agent performing the Super Mario Bros game
    """

    def __init__(self, state_space, action_space, params):

        # Define DQN structure
        self.state_space = state_space
        self.action_space = action_space

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # initialize networks
        self.policy_net = DQN(state_space, action_space).to(self.device)
        self.target_net = DQN(state_space, action_space).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=params.lr)
        self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
        self.step = 0

        # Create memory
        self.memory = ReplayMemory(params.memory_size)

        # set exploration rate
        self.exploration_rate = params.exploration_max

        # loss function
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss

        # remember parameters
        self.params = params

    def act(self, state):
        """
        Act according to a given state
        :param state: The state the agent is currently dealing with
        :return:
        """
        # add one to the step every action
        self.step += 1

        # Epsilon-greedy action
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])

        # Local net is used for the policy
        return torch.argmax(self.policy_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def copy_model(self):
        """
        Method to copy the weights of the policy network into the target network
        :return:
        """
        # Copy local net weights into target net
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Method to remember a specific experience
        :param state: the state in the form of the game screen
        :param action: the action chosen by the agent
        :param reward: the reward given the state and action pair
        :param next_state: the next state resulting from the action
        :param done: boolean denoting whether the game is still running
        :return:
        """
        # combine input into a transition
        transition = (
            state.float(),
            action.float(),
            reward.float(),
            next_state.float(),
            done.float()
        )

        # push transition into memory
        self.memory.push(transition)

    def experience_replay(self):
        """
        Method to let the agent perform experience replay
        :return:
        """

        # copy the policy network to the target network if we reach a set amount of steps
        if self.step % self.copy == 0:
            self.copy_model()

        if len(self.memory) < self.params.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.params.batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        self.optimizer.zero_grad()

        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
        target = reward + torch.mul((self.params.gamma *
                                     self.target_net(next_state).max(1).values.unsqueeze(1)),
                                    1 - done)

        current = self.policy_net(state).gather(1, action.long())  # Local net approximation of Q-value

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.params.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.params.exploration_min)

        return loss
