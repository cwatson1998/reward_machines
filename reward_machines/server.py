import socket
import gym
import pickle
import numpy as np
import sys
from cmd_util import make_env
from reward_machines.envs.grids.office_world import OfficeWorld
from reward_machines.envs.grids.grid_environment import GridEnv


class GymServer:
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
                    message = {'events': self.env.get_events()}

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
