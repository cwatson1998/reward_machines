"""
This code add event detectors to the Ant3 Environment
"""
import gym
import numpy as np
from reward_machines.rm_environment import RewardMachineEnv
from client import GymClient


class IlgRMEnv(RewardMachineEnv):
    ''' The only reason to use this would be if I wanted to do some special cacheing of the events. '''

    def __init__(self, env, rm_files):
        # Note that the current position is key for our tasks
        super().__init__(env, rm_files)
        self.info = {}

    def render(self, mode='human'):
        return

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = super().step(action)
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        ''' This could be optimized to perform less IPC. '''
        return self.env.get_events()


class MyDiag3x3SparseEnv(IlgRMEnv):
    def __init__(self):
        # Set this up using IPC. The thing on the
        env = GymClient()
        env.make("point_maze-3x3-diagonal-one-sparse-10goals")
        rm_files = ["./envs/ilg_rm/reward_machines/diag3x3_sparse_rewards.txt"]
        super().__init__(env, rm_files)