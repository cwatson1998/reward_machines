"""
Q-Learning based method
"""

import random, time, pickle, os
from baselines import logger
import numpy as np


class QlearningWrapper(object):
    """Wrapper for tabular Q-learning that provides save/load functionality"""
    
    def __init__(self, Q, actions, q_init, epsilon):
        self.Q = Q
        self.actions = actions
        self.q_init = q_init
        self.epsilon = epsilon
        self.initial_state = None
    
    def get_qmax(self, s):
        if s not in self.Q:
            self.Q[s] = dict([(a, self.q_init) for a in self.actions])
        return max(self.Q[s].values())
    
    def get_best_action(self, s):
        qmax = self.get_qmax(s)
        best = [a for a in self.actions if self.Q[s][a] == qmax]
        return random.choice(best)
    
    def __call__(self, observation, **kwargs):
        """Main action selection method"""
        s = tuple(observation) if isinstance(observation, (list, np.ndarray)) else observation
        if isinstance(observation, np.ndarray) and observation.ndim > 1:
            s = tuple(observation[0])  # Handle batch input
        
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.get_best_action(s)
    
    def step(self, observation, **kwargs):
        """Step method for compatibility with other agents"""
        # Remove unused parameters for compatibility
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        action = self.__call__(observation, **kwargs)
        return action, None, None, None  # Return as list for step method compatibility
    
    def save(self, path):
        """Save the Q-table and parameters"""
        save_data = {
            'Q': self.Q,
            'actions': self.actions,
            'q_init': self.q_init,
            'epsilon': self.epsilon
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "qlearning_model.pkl")
        self.save(path)
    
    @staticmethod
    def load_act(path):
        """Load a saved Q-learning model"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        return QlearningWrapper(
            Q=save_data['Q'],
            actions=save_data['actions'],
            q_init=save_data['q_init'],
            epsilon=save_data['epsilon']
        )


def get_qmax(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q,s,actions,q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def evaluate_episode(env, Q, actions, q_init):
    """Evaluate a single episode deterministically (no exploration)"""
    s = tuple(env.reset())
    episode_reward = 0
    t = 0
    first_success = None
    
    while True:
        # Select best action deterministically (no exploration)
        a = get_best_action(Q, s, actions, q_init)
        sn, r, done, info = env.step(a)
        t += 1
        if r > 0.5 and first_success is None:
            first_success = t
        sn = tuple(sn)
        episode_reward += r
        if done:
            break
        s = sn
    
    eval_info = {}
    eval_info['episode_reward'] = episode_reward
    eval_info['episode_success'] = int(episode_reward > 0.5)
    eval_info['episode_length'] = t
    eval_info['first_success'] = first_success
    return eval_info

def learn(env,
          network=None,
          seed=None,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.9,
          q_init=2.0,
          use_crm=False,
          use_rs=False,
          eval_env=None,
          eval_episodes=20):
    """Train a tabular q-learning model.

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
    use_crm: bool
        use counterfactual experience to train the policy
    use_rs: bool
        use reward shaping
    """

    # Running Q-Learning
    reward_total = 0
    step = 0
    num_episodes = 0
    num_nonzero_reward_episodes = 0
    num_episodes_this_log_interval = 0
    num_nonzero_reward_episodes_this_log_interval = 0
    Q = {}
    actions = list(range(env.action_space.n))

    while step < total_timesteps:
        s = tuple(env.reset())
        episode_reward = 0
        if s not in Q: Q[s] = dict([(a,q_init) for a in actions])
        while True:
            # Selecting and executing the action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(Q,s,actions,q_init)
            sn, r, done, info = env.step(a)
            sn = tuple(sn)

            # Updating the q-values
            experiences = []
            if use_crm:
                # Adding counterfactual experience (this will alrady include shaped rewards if use_rs=True)
                for _s,_a,_r,_sn,_done in info["crm-experience"]:
                    experiences.append((tuple(_s),_a,_r,tuple(_sn),_done))
            elif use_rs:
                # Include only the current experince but shape the reward
                experiences = [(s,a,info["rs-reward"],sn,done)]
            else:
                # Include only the current experience (standard q-learning)
                experiences = [(s,a,r,sn,done)]

            for _s,_a,_r,_sn,_done in experiences:
                if _s not in Q: Q[_s] = dict([(b,q_init) for b in actions])
                if _done: _delta = _r - Q[_s][_a]
                else:     _delta = _r + gamma*get_qmax(Q,_sn,actions,q_init) - Q[_s][_a]
                Q[_s][_a] += lr*_delta

            # moving to the next state
            reward_total += r
            episode_reward += r
            step += 1
            if step%print_freq == 0:
                # Run 20 deterministic evaluation episodes
                if eval_env is not None:
                    eval_infos = []
                    eval_rewards = []
                    eval_first_successes = []
                    eval_episode_lengths = []
                    for _ in range(eval_episodes):
                        eval_info = evaluate_episode(eval_env, Q, actions, q_init)
                        eval_infos.append(eval_info)
                        eval_reward = eval_info['episode_reward']
                        eval_rewards.append(eval_reward)
                        eval_first_successes.append(eval_info['first_success'])
                        eval_episode_lengths.append(eval_info['episode_length'])

                    eval_mean_reward = np.mean(eval_rewards)
                    eval_success_rate = np.mean([1 if r > 0.5 else 0 for r in eval_rewards])
                    logger.record_tabular("eval mean reward", eval_mean_reward)
                    logger.record_tabular("eval success rate", eval_success_rate)
                    logger.record_tabular("eval episode length", np.mean(eval_episode_lengths))
                    filtered_first_successes = [elt for elt in eval_first_successes if elt is not None]
                    if len(filtered_first_successes) > 0:
                        avg_first_success = np.mean(filtered_first_successes)
                    else:
                        avg_first_success = np.nan
                    logger.record_tabular("eval first success", avg_first_success)
                    
                    
                    
                
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
    return QlearningWrapper(Q, actions, q_init, epsilon)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: QlearningWrapper
        wrapper that takes observations and returns actions.
    """
    return QlearningWrapper.load_act(path)

