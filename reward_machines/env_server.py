import socket
import pickle
import gym
import numpy as np
import argparse
import envs
from reward_machines.rm_environment import RewardMachineWrapper, HierarchicalRMWrapper
from cmd_util import make_env, common_arg_parser

# Maximum buffer size for socket communications (in bytes)
BUFFER_SIZE = 40960


class EnvServer:
    def __init__(self, host='localhost', port=4096):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.env = None
        self.current_obs = None

    def handle_client(self, client_socket):
        while True:
            try:
                # Receive command and data
                data = client_socket.recv(BUFFER_SIZE)
                if not data:
                    break

                command, args = pickle.loads(data)
                print(f"Received command: {command} with args: {args}")

                if command == 'make':
                    try:
                        # Create new environment
                        env_id = args['env_id']
                        env_type = args.get('env_type')
                        use_rs = args.get('use_rs', False)
                        use_crm = args.get('use_crm', False)
                        gamma = args.get('gamma', 0.9)
                        rs_gamma = args.get('rs_gamma', 0.9)
                        use_self_loops = args.get('use_self_loops', False)
                        r_min = args.get('r_min', -1.0)
                        r_max = args.get('r_max', 1.0)

                        print(f"Creating environment: {env_id}")
                        # Create environment

                        # Create environment
                        self.env = make_env(env_id, env_type, args, seed=None)
                        print(f"Environment created successfully")
                        
                        # Get initial observation
                        print("Resetting environment")
                        self.current_obs = self.env.reset()
                        print(f"Initial observation: {self.current_obs}")
                        
                        # Add RM wrappers if needed
                        alg = args.get('alg', '')
                        if isinstance(alg, str) and alg.endswith(("hrm", "dhrm")):
                            print("Adding HierarchicalRMWrapper")
                            self.env = HierarchicalRMWrapper(
                                self.env, r_min, r_max, use_self_loops, 
                                use_rs, gamma, rs_gamma
                            )
                        elif use_rs or use_crm:
                            print("Adding RewardMachineWrapper")
                            self.env = RewardMachineWrapper(
                                self.env, use_crm, use_rs, gamma, rs_gamma
                            )

                        # Get initial observation
                        print("Resetting environment")
                        self.current_obs = self.env.reset()
                        print(f"Initial observation: {self.current_obs}")
                        
                        # Send back observation space and action space info
                        response = {
                            'obs_space': self.env.observation_space,
                            'action_space': self.env.action_space,
                            'obs': self.current_obs
                        }
                        print("Sending response back to client")
                        print(response)
                        client_socket.sendall(pickle.dumps(response))
                        print("Response sent successfully")
                        
                    except Exception as e:
                        print(f"Error creating environment: {e}")
                        import traceback
                        traceback.print_exc()
                        # Send error response
                        error_response = {'error': str(e)}
                        client_socket.send(pickle.dumps(error_response))

                elif command == 'step':
                    # Step environment
                    action = args['action']
                    obs, reward, done, info = self.env.step(action)
                    self.current_obs = obs
                    
                    response = {
                        'obs': obs,
                        'reward': reward,
                        'done': done,
                        'info': info
                    }
                    #print(f"Step response: {response}")
                    client_socket.sendall(pickle.dumps(response))

                elif command == 'reset':
                    # Reset environment
                    obs = self.env.reset()
                    self.current_obs = obs
                    response = {'obs': obs}
                    # print(f"Reset response: {response}")
                    client_socket.sendall(pickle.dumps(response))

                elif command == 'close':
                    # Close environment
                    if self.env:
                        self.env.close()
                    break

            except Exception as e:
                print(f"Error handling client: {e}")
                print(f"Command was: {command}")
                print(f"Args were: {args}")
                import traceback
                traceback.print_exc()
                # Send error response
                error_response = {'error': str(e)}
                print(f"Sending error response: {error_response}")
                client_socket.sendall(pickle.dumps(error_response))
                break

        client_socket.close()

    def serve(self):
        print(f"Server listening on {self.host}:{self.port}")
        while True:
            client_socket, address = self.server_socket.accept()
            print(f"Connected to client at {address}")
            self.handle_client(client_socket)

    def close(self):
        if self.env:
            self.env.close()
        self.server_socket.close()

def parse_args():
    parser = common_arg_parser()
    parser.add_argument('--host', type=str, default='localhost',
                      help='Host to run the server on')
    parser.add_argument('--port', type=int, default=4096,
                      help='Port to run the server on')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    server = EnvServer(host=args.host, port=args.port)
    try:
        server.serve()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.close() 