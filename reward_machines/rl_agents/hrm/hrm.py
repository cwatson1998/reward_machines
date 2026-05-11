"""
Q-Learning based method
"""

import random, pickle, os
from baselines import logger
import numpy as np


class HRMWrapper(object):
    """Wrapper for tabular HRM that provides save/load functionality"""
    
    def __init__(self, Q_controller, Q_options, actions, q_init, epsilon):
        self.Q_controller = Q_controller
        self.Q_options = Q_options
        self.actions = actions
        self.q_init = q_init
        self.epsilon = epsilon
        self.initial_state = None
        
        # State for ongoing option execution
        self.current_option_id = None
        self.option_start_state = None
    
    def add_state_if_needed(self, Q, s, actions):
        if s not in Q:
            Q[s] = dict([(a, self.q_init) for a in actions])
    
    def get_qmax(self, Q, s, actions):
        self.add_state_if_needed(Q, s, actions)
        return max(Q[s].values())
    
    def get_best_action(self, Q, s, actions):
        qmax = self.get_qmax(Q, s, actions)
        best = [a for a in actions if Q[s][a] == qmax]
        return random.choice(best)
    
    def __call__(self, observation, **kwargs):
        """Main action selection method for HRM"""
        # This is a simplified version for playing - in practice HRM needs environment interaction
        # For now, we'll use the meta-controller to select an option, then the option policy
        s = tuple(observation) if isinstance(observation, (list, np.ndarray)) else observation
        if isinstance(observation, np.ndarray) and observation.ndim > 1:
            s = tuple(observation[0])  # Handle batch input
        
        # Simplified action selection - in practice this would need more environment context
        # For playing purposes, we'll select actions from the option policies
        # Note: This is a simplified implementation for basic compatibility
        if s in self.Q_options:
            if random.random() < self.epsilon:
                return random.choice(self.actions)
            else:
                return self.get_best_action(self.Q_options, s, self.actions)
        else:
            # If state not seen, choose random action
            return random.choice(self.actions)
    
    def step(self, observation, **kwargs):
        """Step method for compatibility with other agents"""
        # Remove unused parameters for compatibility
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        action = self.__call__(observation, **kwargs)
        return [action], None, None, None  # Return as list for step method compatibility
    
    def save(self, path):
        """Save the Q-tables and parameters"""
        save_data = {
            'Q_controller': self.Q_controller,
            'Q_options': self.Q_options,
            'actions': self.actions,
            'q_init': self.q_init,
            'epsilon': self.epsilon
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "hrm_model.pkl")
        self.save(path)
    
    @staticmethod
    def load_act(path):
        """Load a saved HRM model"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        return HRMWrapper(
            Q_controller=save_data['Q_controller'],
            Q_options=save_data['Q_options'],
            actions=save_data['actions'],
            q_init=save_data['q_init'],
            epsilon=save_data['epsilon']
        )


def add_state_if_needed(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])

def get_qmax(Q,s,actions,q_init):
    add_state_if_needed(Q,s,actions,q_init)
    return max(Q[s].values())

def get_best_action(Q,s,actions,q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def evaluate_episode(env, Q_controller, Q_options, actions, q_init):
    """Evaluate a single HRM episode deterministically (no exploration)"""
    s = tuple(env.reset())
    episode_reward = 0
    option_id = None
    
    while True:
        # Selecting an option if needed (deterministic)
        if option_id is None:
            valid_options = env.get_valid_options()
            add_state_if_needed(Q_controller, s, valid_options, q_init)
            option_id = get_best_action(Q_controller, s, valid_options, q_init)
        
        # Selecting and executing an action (deterministic)
        option_obs = tuple(env.get_option_observation(option_id))
        add_state_if_needed(Q_options, option_obs, actions, q_init)
        a = get_best_action(Q_options, option_obs, actions, q_init)
        sn, r, done, info = env.step(a)
        sn = tuple(sn)
        
        episode_reward += r
        
        # Check if option terminates
        if env.did_option_terminate(option_id):
            option_id = None
        
        if done:
            break
        s = sn
    
    return episode_reward

def learn(env,
          network=None,
          seed=None,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.9,
          q_init=2.0,
          hrm_lr=0.1,
          use_rs=False,
          eval_env=None,
          eval_episodes=20,
          **others):
    """Train a tabular HRM method.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        This is just a placeholder to be consistent with the openai-baselines interface, but we don't really use state-approximation in tabular q-learning
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate
    total_timesteps: int
        number of env steps to optimizer for
    epsilon: float
        epsilon-greedy exploration
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    gamma: float
        discount factor
    q_init: float
        initial q-value for unseen states
    hrm_lr: float
        learning rate for the macro-controller
    use_rs: bool
        use reward shaping
    eval_env: gym.Env or None
        environment to use for deterministic evaluation (if None, no evaluation is performed)
    eval_episodes: int
        number of deterministic evaluation episodes to run at each logging interval
    """

    # Running Q-Learning
    step         = 0
    num_episodes = 0
    num_nonzero_reward_episodes = 0
    num_episodes_this_log_interval = 0
    num_nonzero_reward_episodes_this_log_interval = 0
    reward_total = 0
    actions      = list(range(env.action_space.n))
    Q_controller = {}   # Q-values for the meta-controller
    Q_options    = {}   # Q-values for the option policies
    option_s     = None # State where the option initiated
    option_id    = None # Id of the current option being executed
    option_rews  = []   # Rewards obtained by the current option
    while step < total_timesteps:
        s = tuple(env.reset())
        episode_reward = 0
        while True:
            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = s
                add_state_if_needed(Q_controller,option_s,valid_options,q_init)
                option_id   = random.choice(valid_options) if random.random() < epsilon else get_best_action(Q_controller,s,valid_options,q_init)
                option_rews = []

            # Selecting and executing an action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(Q_options,tuple(env.get_option_observation(option_id)),actions,q_init)
            sn, r, done, info = env.step(a)
            sn = tuple(sn)

            # Saving the real reward that the option is getting
            if use_rs:
                option_rews.append(info["rs-reward"])
            else:
                option_rews.append(r)

            # Updating the option policies
            for _s,_a,_r,_sn,_done in env.get_experience():
                _s,_sn = tuple(_s), tuple(_sn)
                add_state_if_needed(Q_options,_s,actions,q_init)
                if _done: _delta = _r - Q_options[_s][_a]
                else:     _delta = _r + gamma*get_qmax(Q_options,_sn,actions,q_init) - Q_options[_s][_a]
                Q_options[_s][_a] += lr*_delta

            # Updating the meta-controller if needed 
            # Note that this condition always hold if done is True
            if env.did_option_terminate(option_id):
                option_sn = sn
                option_reward = sum([_r*gamma**_i for _i,_r in enumerate(option_rews)])
                if done: _delta = option_reward - Q_controller[option_s][option_id]
                else:    _delta = option_reward + gamma**(len(option_rews)) * get_qmax(Q_controller,option_sn,env.get_valid_options(),q_init) - Q_controller[option_s][option_id]
                Q_controller[option_s][option_id] += hrm_lr*_delta
                option_id = None

            # Moving to the next state
            reward_total += r
            episode_reward += r
            step += 1
            if step%print_freq == 0:
                # Run deterministic evaluation episodes
                if eval_env is not None:
                    eval_rewards = []
                    for _ in range(eval_episodes):
                        eval_reward = evaluate_episode(eval_env, Q_controller, Q_options, actions, q_init)
                        eval_rewards.append(eval_reward)
                    
                    eval_mean_reward = np.mean(eval_rewards)
                    eval_success_rate = np.mean([1 if r > 0.01 else 0 for r in eval_rewards])
                    logger.record_tabular("eval mean reward", eval_mean_reward)
                    logger.record_tabular("eval success rate", eval_success_rate)
                
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("total reward", reward_total)
                logger.record_tabular("nonzero reward episodes", num_nonzero_reward_episodes)
                logger.record_tabular("success rate this interval", num_nonzero_reward_episodes_this_log_interval / num_episodes_this_log_interval)
                
                logger.dump_tabular()
                reward_total = 0
                num_episodes_this_log_interval = 0
                num_nonzero_reward_episodes_this_log_interval = 0
            if done:
                num_episodes += 1
                num_episodes_this_log_interval += 1
                if episode_reward > 0.01:
                    num_nonzero_reward_episodes += 1
                    num_nonzero_reward_episodes_this_log_interval += 1
                break
            s = sn

    # Return the trained model wrapped in a saveable object
    return HRMWrapper(Q_controller, Q_options, actions, q_init, epsilon)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: HRMWrapper
        wrapper that takes observations and returns actions.
    """
    return HRMWrapper.load_act(path)
