"""
Value Iteration based method for environments with known models
"""

import pickle, os
from baselines import logger
import numpy as np


class ValueIterationWrapper(object):
    """Wrapper for value iteration optimal policies that provides save/load functionality"""
    
    def __init__(self, optimal_policies, actions, env_supports_model=True):
        self.optimal_policies = optimal_policies  # List of policies, one per reward machine
        self.actions = actions
        self.env_supports_model = env_supports_model
        self.initial_state = None
    
    def __call__(self, observation, **kwargs):
        """Main action selection method using optimal policy"""
        # For value iteration, we need environment context which should be provided via kwargs
        env = kwargs.get('env', None)
        if env is not None and hasattr(env, 'current_u_id') and hasattr(env, 'current_rm_id'):
            # Extract the actual environment state from the flattened observation
            # The observation includes both env features and RM state, we need just the env features
            env_state = self._extract_env_state(observation, env)
            rm_state = env.current_u_id
            rm_id = env.current_rm_id
            return self.get_action(env_state, rm_state, rm_id)
        else:
            # Fallback: return first action if we don't have environment context
            return self.actions[0] if self.actions else 0
    
    def step(self, observation, **kwargs):
        """Step method for compatibility with other agents"""
        # Remove unused parameters for compatibility but keep env if present
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        action = self.__call__(observation, **kwargs)
        return action, None, None, None
    
    
    
    def _extract_env_state(self, observation, env):
        """Extract the environment state from the flattened RM observation"""
        # For grid environments, the underlying state is always (x, y) coordinates
        # These are always the first 2 elements of the observation, regardless of RM state encoding
        try:
            # Grid environments always have (x, y) as the first 2 elements
            x, y = int(round(observation[0])), int(round(observation[1]))
            return (x, y)
        except Exception as e:
            print(f"VI: Error extracting state from observation: {e}")
            # Fallback
            return (0, 0)
    
    def get_action(self, state, rm_state, rm_id):
        """Get action using optimal policy for given state, RM state, and RM id"""
        if rm_id < len(self.optimal_policies):
            policy = self.optimal_policies[rm_id]
            state_key = (tuple(state) if not isinstance(state, tuple) else state, rm_state)
            
            if state_key in policy:
                action = policy[state_key]
                print(f"VI: Optimal action {action} for state {state} in RM state {rm_state}")
                return action
            else:
                print(f"VI: Warning - state {state_key} not found in policy for RM {rm_id}")
                # Show some nearby states to help debug
                similar_states = [k for k in policy.keys() if k[0] == state]
                print(f"VI DEBUG: found {len(similar_states)} states with same env position")
                if len(similar_states) > 0:
                    print(f"VI DEBUG: available RM states for position {state}: {[k[1] for k in similar_states]}")
        else:
            print(f"VI: Warning - RM id {rm_id} >= number of policies {len(self.optimal_policies)}")
        
        # Fallback to first action if policy doesn't have this state
        fallback_action = self.actions[0] if self.actions else 0
        print(f"VI: Using fallback action {fallback_action}")
        return fallback_action
    
    def save(self, path):
        """Save the optimal policies and parameters"""
        save_data = {
            'optimal_policies': self.optimal_policies,
            'actions': self.actions,
            'env_supports_model': self.env_supports_model
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "value_iteration_model.pkl")
        self.save(path)
    
    @staticmethod
    def load_act(path):
        """Load a saved value iteration model"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        return ValueIterationWrapper(
            optimal_policies=save_data['optimal_policies'],
            actions=save_data['actions'],
            env_supports_model=save_data.get('env_supports_model', True)
        )


def learn(env,
          network=None,
          seed=None,
          total_timesteps=100000,  # Ignored for value iteration
          gamma=0.9,
          **kwargs):
    """Compute optimal policies using value iteration.

    Parameters
    -------
    env: gym.Env
        environment to compute optimal policy for (must support get_model())
    network: string or a function
        Ignored for value iteration
    seed: int or None
        Ignored for value iteration
    total_timesteps: int
        Ignored for value iteration
    gamma: float
        discount factor
    **kwargs
        Other arguments (ignored)
    """
    
    # Check if environment supports model-based methods
    if not hasattr(env, 'get_model'):
        raise ValueError("Value iteration requires an environment that supports get_model(). "
                        "This is currently only implemented for grid environments.")
    
    if not hasattr(env, 'reward_machines'):
        raise ValueError("Value iteration requires an environment with reward machines. "
                        "Make sure you're using a reward machine environment.")
    
    print("Computing optimal policies using value iteration...")
    
    # Get the environment model
    try:
        S, A, L, T = env.get_model()
        actions = list(A)
        print(f"Environment model: {len(S)} states, {len(A)} actions")
    except Exception as e:
        raise ValueError(f"Failed to get environment model: {e}")
    
    # Import value iteration function
    try:
        from envs.grids.value_iteration import value_iteration
    except ImportError as e:
        raise ImportError(f"Could not import value iteration: {e}")
    
    # Compute optimal policy for each reward machine
    reward_machines = env.reward_machines
    print(f"Computing optimal policies for {len(reward_machines)} reward machines...")
    
    optimal_policies = []
    for i, rm in enumerate(reward_machines):
        print(f"Computing optimal policy for RM {i+1}/{len(reward_machines)}")
        try:
            policy = value_iteration(S, A, L, T, rm, gamma)
            optimal_policies.append(policy)
        except Exception as e:
            raise ValueError(f"Failed to compute optimal policy for RM {i}: {e}")
    
    print("Value iteration completed successfully!")
    print(f"Computed {len(optimal_policies)} optimal policies")
    
    # Return the wrapper with optimal policies
    return ValueIterationWrapper(optimal_policies, actions)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ValueIterationWrapper
        wrapper that takes observations and returns actions.
    """
    return ValueIterationWrapper.load_act(path) 