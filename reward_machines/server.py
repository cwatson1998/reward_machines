import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
from gym import spaces

import socket
import pickle


from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import parse_unknown_args
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module


# Importing our environments and auxiliary functions
import envs
from envs.water.water_world import Ball, BallAgent
from reward_machines.rm_environment import RewardMachineWrapper
from envs.grids.office_world import OfficeWorld
from envs.grids.grid_environment import GridEnv
from cmd_util import make_vec_env, make_env, common_arg_parser
from client import GymClient

# below this line is bad stuff


class GymServer():
    def __init__(self, host='localhost', port=5000):
        # assert env_name is not None
        self.host = host
        self.port = port
        self.server_socket = None
        # self.env_name = env_name
        self.env = None
        self.start_server()

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}...")
        self.conn, self.addr = self.server_socket.accept()
        print(f"Connection established with {self.addr}")
        self.env = None
        # self.env = gym.make('CartPole-v1')  # You can replace this with any Gym environment

    def handle_client(self):
        try:
            while True:
                # Receive message from client
                data = self.conn.recv(4096)
                if not data:
                    break

                message = pickle.loads(data)

                if message['command'] == 'make':
                    if message['name'] == 'GridEnv_OfficeWorld':
                        self.env = OfficeWorld()
                        self.env = GridEnv(self.env)

                    else:
                        raise NotImplementedError(
                            "RM Server received unknown env name "+str(message['name']))
                    # Space construction dict needs to be manually added to each env.
                    # See GridEnv for an example.
                    response = self.env.space_construction_dict

                    # I got rid of this.
                    # # RM specific. Change this for ILG Learn
                    # message.pop('command')
                    # # Pop the args.
                    # # Technically, I think it would be ok to not do this and
                    # # just pass the remaining message.
                    # env_id = message.pop('env_id')
                    # # Now the message should just have the remaining kwargs
                    # self.env = gym.make(env_id, **message)
                    # if self.env is None:
                    #     print("Making env returned None")
                    #     self.conn.close()
                    # response = {'ok': True}
                elif message['command'] == 'get_events':
                    assert self.env is not None, "Need to make env first"
                    # Propositions are returned as strings
                    response = {'events': self.env.get_events()}
                elif message['command'] == 'reset':
                    assert self.env is not None, "Need to make env first"
                    # obs, info = self.env.reset()
                    # response = {'obs': obs, 'info': info}
                    obs = self.env.reset()
                    response = {'obs': obs}
                elif message['command'] == 'step':
                    assert self.env is not None, "Need to make env first"
                    action = message['action']
                    obs, reward, done, info = self.env.step(
                        action)
                    response = {'obs': obs, 'reward': reward,
                                'done': done, 'info': info}
                    # obs, reward, terminate, truncate, info = self.env.step(
                    #     action)
                    # response = {'obs': obs, 'reward': reward,
                    #             'terminate': terminate, 'truncate': truncate, 'info': info}
                elif message['command'] == 'close':
                    assert self.env is not None, "Need to make env first"
                    self.env.close()
                    response = {'status': 'closed'}
                else:
                    response = {'error': 'Unknown command'}

                # Send response back to client
                self.conn.sendall(pickle.dumps(response))

        except Exception as e:
            print(f"Error while handling client: {e}")
        finally:
            self.conn.close()

    def run(self):
        self.handle_client()


if __name__ == "__main__":
    server = GymServer()
    server.run()


# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python server.py <env_name>")
#         sys.exit(1)
#     env_name = sys.argv[1]
#     server = GymServer(env_name=env_name)
#     server.run()
