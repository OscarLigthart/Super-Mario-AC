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
from algorithms import A2C
from models import ActorCritic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--num_steps", type=int, default=100, help="number of episodes")

    args = parser.parse_args()
    return args


def train(args):
    """
    Main method to train the Actor Critic model
    :param args:
    :return:
    """

    # create the environment
    env = create_env(1, 1, "simple")

    # get environment variables
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # initialize Actor-Critic model
    actor_critic = ActorCritic(num_inputs, num_actions)

    # initialize algorithm
    a2c = A2C(actor_critic, args)

    entropy_term = 0

    for episode in range(args.num_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()

        # run episode
        for steps in range(args.num_steps):

            # select an action according to the a2c algorithm
            action, value, log_prob, entropy = a2c.select_action(state)

            # perform this action in the environment
            new_state, reward, done, _ = env.step(action)

            # keep track of episode variables
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy

            # overwrite state
            state = new_state

            # render the environment
            env.render()

            if done or steps == args.num_steps - 1:
                # todo print some information here
                break

        # after an episode we need to update the parameters
        a2c.update(state, values, rewards, log_probs, entropy_term)

    return


if __name__ == "__main__":
    args = get_args()
    train(args)