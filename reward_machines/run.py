import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import imageio

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
from cmd_util import make_vec_env, make_env, common_arg_parser

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    eval_env = build_env(args)
    
    # if args.show_env:
    #    print("about to try to show")
    #    env.show()
    #    return None, None

    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # Adding RM-related parameters
    alg_kwargs['use_rs']   = args.use_rs
    alg_kwargs['use_crm']  = args.use_crm
    alg_kwargs['gamma']    = args.gamma

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    if args.no_learn:
        print("no-learn so no model")
        model = None
    else:
        try:
            model = learn(
                env=env,
                seed=seed,
                total_timesteps=total_timesteps,
                eval_env=eval_env,
                eval_episodes=20,
                **alg_kwargs
            )
        except (TypeError, AttributeError):
            print("learn function does not accept an eval env.")
            # This exception happens if learn does not expect some of the kwargs
            model = learn(
                env=env,
                seed=seed,
                total_timesteps=total_timesteps,
                **alg_kwargs
            )
    print("model type is ")
    print(type(model))
    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)
    
    # Get the environment spec to access its kwargs
    env_spec = gym.envs.registry.env_specs[env_id]
    env_kwargs = env_spec._kwargs if hasattr(env_spec, '_kwargs') else {}
    
    print(f"DEBUG: env_id = {env_id}")
    print(f"DEBUG: env_spec = {env_spec}")
    print(f"DEBUG: env_kwargs = {env_kwargs}")
    print(f"DEBUG: env_spec attributes = {dir(env_spec)}")

    if alg in ['deepq', 'qlearning', 'hrm', 'dhrm', 'value_iteration']:
        env = make_env(env_id, env_type, args, seed=seed, logger_dir=logger.get_dir(), env_kwargs=env_kwargs)
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, args, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations, env_kwargs=env_kwargs)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    library = 'rl_agents'
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join([library, alg, submodule]))
    except ImportError:
        # then from rl_algs
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


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

def parse_rm_state(gridworld_obs):
    raise NotImplementedError("The observation renames the states.")

        


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)
    print("about to show in main")
    print(type(env))
    # env.show()  # Commented out to avoid interactive mode during play
    
    # Render the environment and save as JPEG
    env.reset()
    rgb_array = env.env.render(mode='rgb_array')
    if rgb_array is not None:
        image = Image.fromarray(rgb_array)
        image.save('./scratch_render.jpg')
        print("Saved render to ./scratch_render.jpg")

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        print(f"Model type: {type(model)}")
        print("DEBUG: forcing epsilon to 0")
        model.epsilon = 0
        print("Starting play mode with trained model...")
        
        obs = env.reset()
        print(f"Initial observation: {obs}")

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        step_count = 0
        episode_count = 0
        frames = []  # Store frames for current episode
        
        # Capture initial frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
        
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                # Pass environment to value iteration models for optimal action selection
                if hasattr(model, '__class__') and 'ValueIteration' in model.__class__.__name__:
                    actions, _, _, _ = model.step(obs, env=env)
                else:
                    actions, _, _, _ = model.step(obs)

            print(f"Step {step_count}: obs={obs}, action={actions}")
            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            step_count += 1
            
            # Capture frame after taking action
            rm_state = env.current_u_id
            frame = env.render(mode='rgb_array')
            if frame is not None:
                cv2.putText(frame, f"RM State: {rm_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 20, 147), 2)
                cv2.putText(frame, f"Reward: {rew}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 20, 147), 2)
                cv2.putText(frame, f"Ep Reward: {episode_rew}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 20, 147), 2)
                cv2.putText(frame, f"Step: {step_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 20, 147), 2)
                cv2.putText(frame, f"Events: {env.get_events()}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 20, 147), 2)
                frames.append(frame)
                
                
            
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                episode_count += 1
                for i in np.nonzero(done)[0]:
                    print(f'Episode {episode_count} finished with reward={episode_rew[i]}')
                    episode_rew[i] = 0
                
                # Save video for this episode
                if frames:
                    gif_path = f"episode_{episode_count}_rollout.gif"
                    print(f"Saving video with {len(frames)} frames to {gif_path}")
                    try:
                        imageio.mimsave(gif_path, frames, duration=1)
                        print(f"GIF saved successfully to {gif_path}")
                    except Exception as e:
                        print(f"Error saving GIF: {e}")
                        print("Skipping video save for this episode.")
                    
                    # Save individual frames as JPEGs if requested
                    if args.save_play_jpgs:
                        import os
                        jpg_dir = f"episode_{episode_count}_frames"
                        os.makedirs(jpg_dir, exist_ok=True)
                        for i, frame in enumerate(frames):
                            jpg_path = os.path.join(jpg_dir, f"frame_{i:03d}.jpg")
                            try:
                                image = Image.fromarray(frame)
                                image.save(jpg_path)
                                print(f"Saved frame {i} to {jpg_path}")
                            except Exception as e:
                                print(f"Error saving frame {i}: {e}")
                
                obs = env.reset()
                step_count = 0
                frames = []  # Reset frames for next episode
                
                # Capture initial frame of new episode
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
                
                print(f"New episode {episode_count + 1} started")
                
                # Stop after a few episodes for testing
                if episode_count >= 3:
                    print("Stopping after 3 episodes")
                    break

    env.close()

    return model

if __name__ == '__main__':

    # Examples over the office world:
    #    cross-product baseline: 
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 
    #    cross-product baseline with reward shaping: 
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_rs
    #    CRM: 
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_crm
    #    CRM with reward shaping: 
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_crm --use_rs
    #    HRM: 
    #        >>> python3.6 run.py --alg=hrm --env=Office-v0 --num_timesteps=1e5 --gamma=0.9
    #    HRM with reward shaping: 
    #        >>> python3.6 run.py --alg=hrm --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_rs
    #    Value Iteration (optimal policy): 
    #        >>> python3.6 run.py --alg=value_iteration --env=Office-v0 --gamma=0.9 --play
    # NOTE: The complete list of experiments (that we reported in the paper) can be found on '../scripts' 

    import time
    t_init = time.time()
    main(sys.argv)
    logger.log("Total time: " + str(time.time() - t_init) + " seconds")