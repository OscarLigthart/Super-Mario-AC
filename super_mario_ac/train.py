#
# File: super_mario_ac/train.py
# Auth: Oscar Ligthart
# Desc: The train script for the Super Mario agent
#
####################################################

import torch
import numpy as np
import argparse
from env.env import create_env
from agent import Agent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="right")
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size used for training')
    parser.add_argument('--exploration_max', type=float, default=1.0, help='determine max fraction of exploration')
    parser.add_argument('--exploration_min', type=float, default=0.02, help='determine min fraction of exploration')
    parser.add_argument('--exploration_decay', type=float, default=0.99, help='decay rate of the exploration')
    parser.add_argument('--memory_size', type=int, default=30000, help='The maximum size of the replay buffer')
    parser.add_argument("--num_episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num_steps", type=int, default=1000, help="max number of steps per episode")

    args = parser.parse_args()
    return args


def train(args):
    """
    Main method to train the Actor Critic model
    :param args:
    :return:
    """

    # create the environment
    env = create_env(1, 1, args.action_type)

    # get environment variables
    num_inputs = env.observation_space.shape
    num_actions = env.action_space.n

    # initialize agent
    agent = Agent(num_inputs, num_actions, args)

    # run for selected amount of episodes
    for episode in range(args.num_episodes):

        # retrieve starting state from env
        state = torch.Tensor([env.reset()])

        # keep track of values
        total_reward = 0

        # run episode
        for steps in range(args.num_steps):

            # allow the agent to act
            action = agent.act(state)

            # perform this action in the environment
            next_state, reward, done, _ = env.step(int(action[0]))

            total_reward += reward
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward]).unsqueeze(0)
            done = torch.Tensor([int(done)]).unsqueeze(0)

            # make sure the agent remembers this experience
            agent.remember(state, action, reward, next_state, done)

            # run experience replay - This is where the learning happens
            loss = agent.experience_replay()

            # overwrite state
            state = next_state

            # render the environment
            env.render()

            if done or steps == args.num_steps - 1:
                # todo print some information here
                break

        print(f"Episode {episode} \t - Reward: {total_reward} \t - Loss: {loss}")

    return


if __name__ == "__main__":
    args = get_args()
    train(args)
