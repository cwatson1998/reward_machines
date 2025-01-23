import socket
import pickle
import numpy as np


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
        message = {'command': command}
        message.update(kwargs)
        self.client_socket.sendall(pickle.dumps(message))

        # Receive response from server
        data = self.client_socket.recv(4096)
        return pickle.loads(data)

    def make(self, name):
        self.send_command("make", name=name)

    def get_events(self):
        response = self.send_command('get_events')
        return response['events']

    # def make(self, **kwargs):
    #     """ Must pass all positional args as kwargs also. """
    #     self.send_command("make", **kwargs)
    #     return None

    def reset(self):
        response = self.send_command('reset')
        return response['obs']

    def step(self, action):
        response = self.send_command('step', action=action)
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
