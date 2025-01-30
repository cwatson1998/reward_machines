import socket
import pickle
import numpy as np
from gym import spaces


def space_from_dict(construction_dict):
    args = construction_dict['space_args']
    kwargs = construction_dict['space_primitive_kwargs']
    if construction_dict['space_dtype'] is not None:
        if construction_dict['space_dtype'] == 'np.uint8':
            kwargs['dtype'] = np.uint8
        else:
            # I learned that Gym spaces can consume things like "float32" or "uint8"
            kwargs['dtype'] = construction_dict['space_dtype']
            # raise NotImplementedError(
            #     "A space_construction_dict provided the unknown dtype "+str(construction_dict['space_dtype']))
    # Now kwargs includes the dtype.
    if construction_dict['space_type'] == 'Discrete':
        return spaces.Discrete(*args, **kwargs)
    elif construction_dict['space_type'] == 'Box':
        return spaces.Box(*args, **kwargs)


class GymClient:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.client_socket = None
        self.connect_to_server()

    def connect_to_server(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")

    def send_command(self, command, **kwargs):
        # If you add more to the message, make will break, because I pass message with 'command' removed to the env's make function on serverside.
        # Actually I think the little note above is not true?
        message = {'command': command}
        message.update(kwargs)
        self.client_socket.sendall(pickle.dumps(message))

        # Receive response from server
        data = self.client_socket.recv(4096)
        return pickle.loads(data)

    def seed(self, *args, **kwargs):
        print("Client does not pass seed.")
        return

    def make(self, name):
        response = self.send_command("make", name=name)
        self.action_space = space_from_dict(
            response['action_space_construction_dict'])
        self.observation_space = space_from_dict(
            response['observation_space_construction_dict'])
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None
        self.unwrapped = self

    def get_events(self):
        # This is too much ipc. I should send the events along with every step and reset.
        response = self.send_command('get_events')
        # print(f"debug. client receives events {response['events']}")
        # print(f'debug: gymclient sends events as {response["events"]}')
        return response['events']

    # def make(self, **kwargs):
    #     """ Must pass all positional args as kwargs also. """
    #     self.send_command("make", **kwargs)
    #     return None

    def reset(self):
        response = self.send_command('reset')
        return response['obs']

    def step(self, action):
        # For some reason, it seems to send shape (1,N) when it should be (N,).
        # I flatten it on the other side.
        # print("action")
        # print(action)
        # print(type(action))
        # print(action.shape)
        # print(self.action_space)
        response = self.send_command('step', action=action)
        # The info field that we return needs to have any data that is needed for recreating the reward function.
        # raise NotImplementedError("chris")
        # print(response['info'])
        # print('debug: client receives done as '+str(response['done']))
        # print(f'debug: gymclient sends info as {response["info"]}')
        return response['obs'], response['reward'], response['done'], response['info']

    def close(self):
        self.send_command('close')
        self.client_socket.close()
        print("Closed connection to server")


# if __name__ == "__main__":
#     client = GymClient()
#     obs = client.reset()

#     for _ in range(100):
#         action = np.random.choice([0, 1])  # Random action for CartPole
#         obs, reward, done, info = client.step(bool(action))
#         print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

#         if done:
#             break

#     client.close()
