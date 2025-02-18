import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import time

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

from rl_agents.dhrm.options import OptionDQN, OptionDDPG
from rl_agents.dhrm.controller import ControllerDQN

import wandb

def learn(env,
          use_ddpg=False,
          gamma=0.9,
          use_rs=False,
          controller_kargs={},
          option_kargs={},
          seed=None,
          total_timesteps=100000,
          print_freq=100,
          callback=None,
          checkpoint_path=None,
          checkpoint_freq=10000,
          load_path=None,
          **others):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    use_ddpg: bool
        whether to use DDPG or DQN to learn the option's policies
    gamma: float
        discount factor
    use_rs: bool
        use reward shaping
    controller_kargs
        arguments for learning the controller policy.
    option_kargs
        arguments for learning the option policies.
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    total_timesteps: int
        number of env steps to optimizer for
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    load_path: str
        path to load the model from. (default: None)

    Returns
    -------
    act: ActWrapper (meta-controller)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    act: ActWrapper (option policies)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    assert checkpoint_path is not None
    print(f"checkpoint path: {checkpoint_path}")
    wandb.init(project=others['wandb_project'],
               name=others['wandb_name'],
               entity=others['wandb_entity'],
            #    id=None, 
            #    resume =False
               tags=[others['wandb_tag']] if others['wandb_tag'] is not None else None)
    wandb.config.update(dict(
          use_ddpg=use_ddpg,
          gamma=gamma,
          use_rs=use_rs,
          seed=seed,
          total_timesteps=total_timesteps,
          print_freq=print_freq,
          checkpoint_path=checkpoint_path,
          checkpoint_freq=checkpoint_freq,
          load_path=load_path))
    # Create all the functions necessary to train the model
    
    

    controller  = ControllerDQN(env, **controller_kargs)
    # saver = tf.train.Saver()
    # print('debug made it past controller and saver')
    sess = get_session()
    if use_ddpg:
        options = OptionDDPG(env, gamma, total_timesteps, **option_kargs)
    else:
        options = OptionDQN(env, gamma, total_timesteps, **option_kargs)
    saver = tf.train.Saver()
    sess.graph.finalize()
    options.agent.reset()

    
    
    set_global_seeds(seed)
        
    option_s    = None # State where the option initiated
    option_id   = None # Id of the current option being executed
    option_rews = []   # Rewards obtained by the current option

    episode_rewards = [0.0]
    tracked_events = ['a','b','c','d','e']
    episode_data_lists = {e: [] for e in tracked_events}
    episode_data_lists['best_dense'] = []
    episode_data_lists['best_is_success'] = []
    episode_data_lists['ep_len'] = []
    episode_data_lists['option_transitions'] = []
    episode_data_lists['missing_data'] = []
    prev_time = time.time()
    prev_t = 0
    def prepare_episode_data_lists_for_new_episode():
        for e in tracked_events:
            episode_data_lists[e].append(0)
        episode_data_lists['best_dense'].append(-9999)
        episode_data_lists['best_is_success'].append(0)
        episode_data_lists['ep_len'].append(0)
        episode_data_lists['option_transitions'].append(0)
        episode_data_lists['missing_data'].append(0)
    def update_episode_data_lists(info):
        try:
            for e in tracked_events:
                if e in info['events']:
                    episode_data_lists[e][-1] = 1
        except AttributeError:
            episode_data_lists['missing_data'][-1] = 1
        try:
            episode_data_lists['best_dense'][-1] = max(episode_data_lists['best_dense'][-1], info['dense'])
        except AttributeError:
            episode_data_lists['missing_data'][-1] = 1
        try:
            episode_data_lists['best_is_success'][-1] = max(episode_data_lists['best_is_success'][-1], info['is_success'])
        except AttributeError:
            episode_data_lists['missing_data'][-1] = 1
        try:
            episode_data_lists['ep_len'][-1] += 1
        except AttributeError:
            episode_data_lists['missing_data'][-1] = 1
    prepare_episode_data_lists_for_new_episode()
    saved_mean_reward = None
    # print(f"debug type of env is {type(env)}")
    obs = env.reset()
    options.reset()
    reset = True
    loaded_model=False
    
    with tempfile.TemporaryDirectory() as td:
        td = td
        cd = checkpoint_path

        # model_file = os.path.join(td, "model")
        # model_checkpoint_file = os.path.join(checkpoint_path, "model")
        model_saved = False

        # if tf.train.latest_checkpoint(td) is not None:
        #     load_variables(model_file)
        #     logger.log('Loaded model from {}'.format(model_file))
        #     loaded_model=True
        #     model_saved = True
        # elif load_path is not None:
        #     load_variables(load_path)
        #     logger.log('Loaded model from {}'.format(load_path))

        
        for t in range(total_timesteps):
            if loaded_model:
                t + t + 10_000_000
            if callback is not None:
                if callback(locals(), globals()):
                    break

            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = obs
                option_id   = controller.get_action(option_s, valid_options)
                episode_data_lists['option_transitions'][-1] += 1
                option_rews = []

            # Take action and update exploration to the newest value
            action = options.get_action(env.get_option_observation(option_id), t, reset)
            reset = False
            new_obs, rew, done, info = env.step(action)
            # print(f"debug: in the loop we get done as {done}")
            

            # Saving the real reward that the option is getting
            if use_rs:
                option_rews.append(info["rs-reward"])
            else:
                option_rews.append(rew)

            # Store transition for the option policies
            for _s,_a,_r,_sn,_done in env.get_experience():
                options.add_experience(_s,_a,_r,_sn,_done)

            # Learn and update the target networks if needed for the option policies
            options.learn(t)
            options.update_target_network(t)

            # Update the meta-controller if needed 
            # Note that this condition always hold if done is True
            if env.did_option_terminate(option_id):
                option_sn = new_obs
                option_reward = sum([_r*gamma**_i for _i,_r in enumerate(option_rews)])
                valid_options = [] if done else env.get_valid_options()
                controller.add_experience(option_s, option_id, option_reward, option_sn, done, valid_options,gamma**(len(option_rews)))
                controller.learn()
                controller.update_target_network()
                controller.increase_step()
                option_id = None

            obs = new_obs
            episode_rewards[-1] += rew
            # print(f"debug info {info}")
            update_episode_data_lists(info)
            
            
    
            

            if done:
                obs = env.reset()
                options.reset()
                episode_rewards.append(0.0)
                prepare_episode_data_lists_for_new_episode()
                reset = True

            # General stats
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                
#                print(f"debug. we have {episode_data_lists}")
                # Compute mean over 100 items!!!!
                wandb_log_dict = {
                    f"{k}_mean100": round(np.mean(v[-101:-1]), 1) for k,v in episode_data_lists.items()
                }
                
                wandb_log_dict['custom_step']=t
                wandb_log_dict['episodes']=num_episodes
                wandb_log_dict['mean_100ep_reward_hrm']=mean_100ep_reward
                cur_time = time.time()
                wandb_log_dict['sps'] = (t - prev_t)/(cur_time - prev_time)
                prev_t = t
                prev_time = cur_time
                
                # Log the prepared dictionary with the step
                wandb.log(wandb_log_dict, step=t)
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("sps", wandb_log_dict['sps'])
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.dump_tabular()

            if (checkpoint_freq is not None and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    # save_variables(model_file)
                    # save_checkpoint_name = f"{model_checkpoint_file}_{mean_100ep_reward}_"
                    saver_stem = f'{cd}/saver_model.ckpt'
                    print(f"Trying to use saver to save at {saver_stem}")
                    saver.save(sess, saver_stem, global_step=t)
                    
                    # print(f"Tried to save to {model_file}")
                    # print(f"Tried to save to {save_checkpoint_name}")
                    
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            #load_variables(model_file)
        print("Now I'm going to try to do some more rollouts to eval right here!")
        
        final_eval_dict = gym_eval(
            env,
            use_ddpg=use_ddpg,
            use_rs=use_rs,
            seed=seed,
            premade_controller=controller,
            premade_options=options,
            eval_episodes=1000
        )
        print("the final eval completed.")
        final_eval_dict = {f"final_{k}": v for k, v in final_eval_dict.items()}
        wandb.log(final_eval_dict, step=t)
        print("I uploaded final eval to wandb final eval completed.")
        
        
        wandb.finish()





    return controller.act, options.act


def gym_eval(env,
          use_ddpg=False,
          gamma=0.9,
          use_rs=False,
          controller_kargs={},
          option_kargs={},
          seed=None,
          total_timesteps=100000,
          print_freq=100,
          callback=None,
          checkpoint_path=None,
          checkpoint_freq=10000,
          load_path=None,
          save_path=None,
          premade_controller=None,
          premade_options=None,
          eval_episodes=100,
          **others):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    use_ddpg: bool
        whether to use DDPG or DQN to learn the option's policies
    gamma: float
        discount factor
    use_rs: bool
        use reward shaping
    controller_kargs
        arguments for learning the controller policy.
    option_kargs
        arguments for learning the option policies.
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    total_timesteps: int
        number of env steps to optimizer for
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    load_path: str
        path to load the model from. (default: None)

    Returns
    -------
    act: ActWrapper (meta-controller)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    act: ActWrapper (option policies)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # eval_episodes = others['eval_episodes']
    

    
    print(f"hardcoded eval episodes in dhrm to {eval_episodes}")
    # wandb.init(project=others['wandb_project'],
    #            name=others['wandb_name'],
    #            entity=others['wandb_entity'],
    #         #    id=None, 
    #         #    resume =False
    #            tags=[others['wandb_tag']] if others['wandb_tag'] is not None else None)
    # wandb.config.update(dict(
    #       use_ddpg=use_ddpg,
    #       gamma=gamma,
    #       use_rs=use_rs,
    #       seed=seed,
    #       total_timesteps=total_timesteps,
    #       print_freq=print_freq,
    #       checkpoint_path=checkpoint_path,
    #       checkpoint_freq=checkpoint_freq,
    #       load_path=load_path))
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    if premade_controller is None:
        assert premade_options is None
        controller  = ControllerDQN(env, **controller_kargs)
        print("after making controller but before making options it looks like this")
        for var in tf.global_variables():
            print(f"{var.name}: mean={sess.run(var).mean()}, std={sess.run(var).std()}")
        if use_ddpg:
            options = OptionDDPG(env, gamma, total_timesteps, **option_kargs)
        else:
            options = OptionDQN(env, gamma, total_timesteps, **option_kargs)
        # This 
        print("after making options it looks like this:")
        saver = tf.train.Saver()
        for var in tf.global_variables():
            print(f"{var.name}: mean={sess.run(var).mean()}, std={sess.run(var).std()}")

        logger.log('trying to load model from {}'.format(save_path))
        # load_variables(save_path)
        saver.restore(sess, save_path)
        print("after loading it looks like")
        for var in tf.global_variables():
            print(f"{var.name}: mean={sess.run(var).mean()}, std={sess.run(var).std()}")
        sess.graph.finalize()
    else:
        assert save_path is None
        controller = premade_controller
        options= premade_options
    
    # This would be an ok place to try to load.
    
    
    
    
    
    # print("about to try to unfinalize the graph")
    # tf.get_default_graph()._unsafe_unfinalize()
    # load_variables(save_path)
    # sess.run(tf.assign(is_training, False))
    
    
    
    options.agent.reset()


        
    option_s    = None # State where the option initiated
    option_id   = None # Id of the current option being executed
    option_rews = []   # Rewards obtained by the current option

    episode_rewards = [0.0]
    tracked_events = ['a','b','c','d','e']
    episode_data_lists = {e: [] for e in tracked_events}
    episode_data_lists['best_dense'] = []
    episode_data_lists['best_is_success'] = []
    episode_data_lists['ep_len'] = []
    episode_data_lists['option_transitions'] = []
    episode_data_lists['missing_data'] = []
    prev_time = time.time()
    prev_t = 0
    def prepare_episode_data_lists_for_new_episode():
        for e in tracked_events:
            episode_data_lists[e].append(0)
        episode_data_lists['best_dense'].append(-9999)
        episode_data_lists['best_is_success'].append(0)
        episode_data_lists['ep_len'].append(0)
        episode_data_lists['option_transitions'].append(0)
        episode_data_lists['missing_data'].append(0)
    def update_episode_data_lists(info):
        try:
            for e in tracked_events:
                if e in info['events']:
                    episode_data_lists[e][-1] = 1
        except AttributeError:
            episode_data_lists['missing_data'][-1] = 1
        try:
            episode_data_lists['best_dense'][-1] = max(episode_data_lists['best_dense'][-1], info['dense'])
        except AttributeError:
            episode_data_lists['missing_data'][-1] = 1
        try:
            episode_data_lists['best_is_success'][-1] = max(episode_data_lists['best_is_success'][-1], info['is_success'])
        except AttributeError:
            episode_data_lists['missing_data'][-1] = 1
        try:
            episode_data_lists['ep_len'][-1] += 1
        except AttributeError:
            episode_data_lists['missing_data'][-1] = 1
    prepare_episode_data_lists_for_new_episode()
    saved_mean_reward = None
    # print(f"debug type of env is {type(env)}")
    obs = env.reset()
    options.reset()
    reset = True
    loaded_model=False
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        # model_file = os.path.join(td, "model")
        # model_saved = False

        # if tf.train.latest_checkpoint(td) is not None:
        #     load_variables(model_file)
        #     logger.log('Loaded model from {}'.format(model_file))
        #     loaded_model=True
        #     model_saved = True
        

        t = -1
        while len(episode_data_lists['best_is_success']) < eval_episodes:
            t = t + 1
            
            # if loaded_model:
            #     t + t + 10_000_000
            if callback is not None:
                if callback(locals(), globals()):
                    break

            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = obs
                option_id   = controller.get_action(option_s, valid_options)
                episode_data_lists['option_transitions'][-1] += 1
                option_rews = []

            # Take action and update exploration to the newest value
            action = options.get_action_no_noise(env.get_option_observation(option_id), t, reset)
            reset = False
            new_obs, rew, done, info = env.step(action)
            # print(f"debug: in the loop we get done as {done}")
            

            # Saving the real reward that the option is getting
            # if use_rs:
            #     option_rews.append(info["rs-reward"])
            # else:
            #     option_rews.append(rew)

            # Store transition for the option policies
            # for _s,_a,_r,_sn,_done in env.get_experience():
            #     options.add_experience(_s,_a,_r,_sn,_done)

            # Learn and update the target networks if needed for the option policies
            # options.learn(t)
            # options.update_target_network(t)

            # Update the meta-controller if needed 
            # Note that this condition always hold if done is True
            if env.did_option_terminate(option_id):
                option_sn = new_obs
                option_reward = sum([_r*gamma**_i for _i,_r in enumerate(option_rews)])
                valid_options = [] if done else env.get_valid_options()
                # controller.add_experience(option_s, option_id, option_reward, option_sn, done, valid_options,gamma**(len(option_rews)))
                # controller.learn()
                # controller.update_target_network()
                # controller.increase_step()
                option_id = None

            obs = new_obs
            episode_rewards[-1] += rew
            # print(f"debug info {info}")
            update_episode_data_lists(info)
            
            
    
            

            if done:
                if len(episode_data_lists['best_is_success'])+1 % 50 == 0:
                    print(f"We completed {len(episode_data_lists['best_is_success'])} episodes in {t} steps")
                obs = env.reset()
                options.reset()
                episode_rewards.append(0.0)
                prepare_episode_data_lists_for_new_episode()
                reset = True

            # General stats
#             mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
#             num_episodes = len(episode_rewards)
#             if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                
# #                print(f"debug. we have {episode_data_lists}")
                # Compute mean over 100 items!!!!
                # wandb_log_dict = {
                #     f"{k}_mean100": round(np.mean(v[-101:-1]), 1) for k,v in episode_data_lists.items()
                # }
                
#                 wandb_log_dict['custom_step']=t
#                 wandb_log_dict['episodes']=num_episodes
#                 wandb_log_dict['mean_100ep_reward_hrm']=mean_100ep_reward
#                 cur_time = time.time()
#                 wandb_log_dict['sps'] = (t - prev_t)/(cur_time - prev_time)
#                 prev_t = t
#                 prev_time = cur_time
                
#                 # Log the prepared dictionary with the step
#                 wandb.log(wandb_log_dict, step=t)
#                 logger.record_tabular("steps", t)
#                 logger.record_tabular("episodes", num_episodes)
#                 logger.record_tabular("sps", wandb_log_dict['sps'])
#                 logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
#                 logger.dump_tabular()

#             if (checkpoint_freq is not None and
#                     num_episodes > 100 and t % checkpoint_freq == 0):
#                 if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
#                     if print_freq is not None:
#                         logger.log("Saving model due to mean reward increase: {} -> {}".format(
#                                    saved_mean_reward, mean_100ep_reward))
#                     save_variables(model_file)
#                     model_saved = True
#                     saved_mean_reward = mean_100ep_reward
#         if model_saved:
#             if print_freq is not None:
#                 logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
#             #load_variables(model_file)
#         wandb.finish()
    # for k,v in episode_data_lists.items():
        # print(f"{k}: {v}")
    
    wandb_log_dict = {
                    f"{k}_mean100": round(np.mean(v[:-1]), 5) for k,v in episode_data_lists.items()
                }
    wandb_log_dict['hrm_reward_mean100'] = round(np.mean(episode_rewards[:-1]), 5)
    print(wandb_log_dict)
    for k,v in wandb_log_dict.items():
        print(f"{k}: {v}")
        # for vv in v:
        #     print(f"{k}: {vv}")
    print("that concludes the eval")
    return wandb_log_dict
