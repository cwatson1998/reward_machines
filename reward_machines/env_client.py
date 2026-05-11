import socket
import pickle
import numpy as np
import gym
import argparse
from cmd_util import common_arg_parser
from baselines.common.cmd_util import parse_unknown_args
from importlib import import_module
import sys

# Maximum buffer size for socket communications (in bytes)
BUFFER_SIZE = 40960

class EnvClient:
    def __init__(self, host='localhost', port=4096):
        self.host = host
        self.port = port
        self.socket = None
        self.obs_space = None
        self.action_space = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def make(self, env_id, env_type=None, use_rs=False, use_crm=False, gamma=0.9, rs_gamma=0.9, use_self_loops=False, r_min=-1.0, r_max=1.0, alg=None):
        args = {
            'env_id': env_id,
            'env_type': env_type,
            'use_rs': use_rs,
            'use_crm': use_crm,
            'gamma': gamma,
            'rs_gamma': rs_gamma,
            'use_self_loops': use_self_loops,
            'r_min': r_min,
            'r_max': r_max,
            'alg': alg
        }
        print(f"Sending make command with args: {args}")
        self.socket.sendall(pickle.dumps(('make', args)))
        
        try:
            data = self.socket.recv(BUFFER_SIZE)
            print(f"Received {len(data)} bytes of data")
            response = pickle.loads(data)
            print(f"Received response: {response}")
            
            if 'error' in response:
                raise Exception(f"Server error: {response['error']}")
                
            self.obs_space = response['obs_space']
            self.action_space = response['action_space']
            return response['obs']
        except EOFError:
            print("Error: Server closed connection unexpectedly")
            raise
        except Exception as e:
            print(f"Error in make: {e}")
            raise

    def step(self, action):
        self.socket.send(pickle.dumps(('step', {'action': action})))
        return pickle.loads(self.socket.recv(BUFFER_SIZE))

    def reset(self):
        self.socket.send(pickle.dumps(('reset', {})))
        return pickle.loads(self.socket.recv(BUFFER_SIZE))['obs']

    def close(self):
        if self.socket:
            self.socket.send(pickle.dumps(('close', {})))
            self.socket.close()

def get_alg_module(alg, submodule=None):
    library = 'rl_agents'
    submodule = submodule or alg
    try:
        # first try to import the alg module from rl_agents
        alg_module = import_module('.'.join([library, alg, submodule]))
    except ImportError:
        # then from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    return alg_module

def get_learn_function(alg):
    return get_alg_module(alg).learn

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}

def main(args):
    arg_parser = common_arg_parser()
    arg_parser.add_argument('--host', type=str, default='localhost',
                      help='Host to connect to')
    arg_parser.add_argument('--port', type=int, default=4096,
                      help='Port to connect to')
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    #extra_args = parse_cmdline_kwargs(unknown_args)
    
    # Create a proxy environment that communicates with the server
    class RemoteEnv:
        def __init__(self, client):
            self.client = client
            self.observation_space = client.obs_space
            self.action_space = client.action_space
        
        def step(self, action):
            result = self.client.step(action)
            return result['obs'], result['reward'], result['done'], result['info']
        
        def reset(self):
            return self.client.reset()
        
        def close(self):
            self.client.close()
        
        # Add methods needed by HRM/DHRM algorithms
        def get_valid_options(self):
            # This would need to be implemented based on the environment
            # For now, return empty to avoid errors
            return []
        
        def get_option_observation(self, option_id):
            # This would need to be implemented based on the environment
            return []
        
        def get_experience(self):
            # This would need to be implemented based on the environment
            return []
        
        def did_option_terminate(self, option_id):
            # This would need to be implemented based on the environment
            return False
    
    # Connect to server
    client = EnvClient(host=args.host, port=args.port)
    client.connect()
    
    # Create environment
    obs = client.make(
        env_id=args.env,
        env_type=args.env_type,
        use_rs=args.use_rs,
        use_crm=args.use_crm,
        gamma=args.gamma,
        rs_gamma=args.rs_gamma,
        use_self_loops=args.use_self_loops,
        r_min=args.r_min,
        r_max=args.r_max,
        alg=args.alg
    )
    
    # Create proxy environment
    env = RemoteEnv(client)
    
    # Get learning function and defaults
    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, args.env_type or 'grid')
    alg_kwargs.update(extra_args)
    # Add RM-related parameters
    alg_kwargs['use_rs'] = args.use_rs
    alg_kwargs['use_crm'] = args.use_crm
    alg_kwargs['gamma'] = args.gamma
    
    # Override with command line arguments
    total_timesteps = int(float(args.num_timesteps))
    
    print(f'Training {args.alg} on {args.env} with arguments \n{alg_kwargs}')
    
    # Run the learning algorithm
    model = learn(
        env=env,
        seed=args.seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )
    
    # Close connection
    client.close()
    
    print("\nTraining finished!")
    return model

if __name__ == "__main__":
    main(sys.argv) 