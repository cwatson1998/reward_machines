import math
import numpy as np

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c

class RewardControl(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "ctrl"

    def get_reward(self, s_info):
        return s_info['reward_ctrl']

class RewardForward(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "forward"

    def get_reward(self, s_info):
        return s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah


class RewardBackwards(RewardFunction):
    """
    Gives a reward for moving backwards
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "backwards"

    def get_reward(self, s_info):
        return -s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah
    

class IlgVertexRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, source_vertex_index):
        super().__init__()
        self.source_vertex_index = source_vertex_index

    def get_type(self):
        return f"IlgVertexReward{self.source_vertex_index}"

    def get_reward(self, s_info):
        dist = s_info['reward_value_array'][self.source_vertex_index]
        return np.exp(-dist)
        # return the exponential negative distance 
    
class IlgInfoRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, source_vertex_index):
        super().__init__()
        self.source_vertex_index = source_vertex_index

    def get_type(self):
        return f"IlgInfoReward{self.source_vertex_index}"

    def get_reward(self, s_info):
        return s_info[f'r{self.source_vertex_index}']
    






