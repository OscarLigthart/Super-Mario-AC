#
# File: super_mario_ac/env/env.py
# Auth: Oscar Ligthart
# Desc: An environment wrapper around the super mario gym package
#
############################

import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


from .wrappers import CustomReward, CustomSkipFrame


def create_env(world, stage, action_type, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))

    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    env = JoypadSpace(env, actions)
    env = CustomReward(env, None)
    env = CustomSkipFrame(env)
    return env, env.observation_space.shape[0], len(actions)
