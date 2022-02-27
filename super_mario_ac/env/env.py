#
# File: super_mario_ac/env/env.py
# Auth: Oscar Ligthart
# Desc: An environment wrapper around the super mario gym package
#
############################

import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


from .wrappers import MaxAndSkipEnv, ProcessFrame84, ImageToPyTorch, BufferWrapper, ScaledFloatFrame


def create_env(world, stage, action_type):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))

    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return JoypadSpace(env, actions)

