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
        # print('debug IlgRmEnv')
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        ''' This could be optimized to perform less IPC. '''
        return self.env.get_events()


class MyDiag3x3SparseEnv(IlgRMEnv):
    def __init__(self, **kwargs):
        # Set this up using IPC. The thing on the
        port = kwargs.get("port", 5000)
        print("Got port as "+str(port))
        env = GymClient(port=port)
        env.make("point_maze-3x3-diagonal-one-sparse-10goals")
        rm_files = ["./envs/ilg_rm/reward_machines/diag3x3_sparse_rewards.txt"]
        super().__init__(env, rm_files)

class MyDiag3x3DenseEnv(IlgRMEnv):
    def __init__(self, **kwargs):
        # Set this up using IPC. The thing on the
        
        port = kwargs.get("port", 5000)
        print("Got port as "+str(port))
        env = GymClient(port=port)
        env.make("point_maze-3x3-diagonal-one-sparse-10goals")
        rm_files = ["./envs/ilg_rm/reward_machines/diag3x3_dense_rewards.txt"]
        super().__init__(env, rm_files)

class MyDiag7x7DenseEnv(IlgRMEnv):
    def __init__(self, **kwargs):
        # Set this up using IPC. The thing on the
        
        port = kwargs.get("port", 5000)
        print("Got port as "+str(port))
        env = GymClient(port=port)
        env.make("point_maze-7x7-diagonal-one-sparse-10goals")
        rm_files = ["./envs/ilg_rm/reward_machines/diag7x7_dense_rewards.txt"]
        super().__init__(env, rm_files)

class MyStackChoiceOutwardviewEnv(IlgRMEnv):
    def __init__(self, **kwargs):
        # Set this up using IPC. The thing on the
        
        port = kwargs.get("port", 5000)
        print("Got port as "+str(port))
        env = GymClient(port=port)
        env.make("stack_twoblock_choice_50_examples_outwardview")
        rm_files = ["./envs/ilg_rm/reward_machines/stack_choice_outwardview.txt"]
        super().__init__(env, rm_files)


class MyStackABEnv(IlgRMEnv):
    def __init__(self, **kwargs):
        # Set this up using IPC. The thing on the
        
        port = kwargs.get("port", 5000)
        print("Got port as "+str(port))
        env = GymClient(port=port)
        env.make("stack_twoblock_A_B_50_examples")
        rm_files = ["./envs/ilg_rm/reward_machines/stack_AB.txt"]
        super().__init__(env, rm_files)