import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import datetime
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from rl_games.algos_torch.players import PpoPlayerContinuous
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv

from rlgames_train import RLGTrainer
from rl_games.torch_runner import Runner
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from torch._C import fork

#from utils..plot_experiment import plot_episode_data_virtual """I can comment because it is for multiple agents"""
import wandb

import os

def eval_single_agent(cfg_dict, cfg, env):
    base_dir = "./evaluations/"
    evaluation_dir = base_dir + str(cfg_dict["task"]["name"]) + "/"
    os.makedirs(evaluation_dir, exist_ok=True)

    player = PpoPlayerContinuous(cfg_dict['train']['params'])
    player.restore(cfg.checkpoint)
    # _____Run Evaluation_____
    num_episodes = 1 
    obs = env.reset()
    ep_data = {'act': [], 'obs': [], 'rews': [], 'info': []}
    info = {}
    for episode in range(num_episodes):
        done = 0
        step = 0
        rew_sum = torch.zeros(1, device=cfg_dict['train']['params']['config']['device'])

        while not done:
            action = player.get_action(obs['obs'], is_deterministic=True)
            #env._task.pre_physics_step(actions)
            obs, rews, dones, info =  env.step(action)
            done = dones[0]
            rew_sum += rews[0]
            step += 1
            ep_data['act'].append(action.cpu().numpy())
            ep_data['obs'].append(obs['obs'].cpu().numpy().flatten())
            ep_data['rews'].append(rew_sum.cpu().numpy())
            ep_data['info'].append(info)
        ep_data['obs'] = np.array(ep_data['obs'])
            #print(f'Step {step}: action={action}, obs={obs}, rews={rews}, dones={dones}, info={info} \n')
        print(f'Episode {episode}: rew_sum={rew_sum}info \n')
        print(ep_data)
        #plot_episode_data(ep_data, evaluation_dir)

def eval_multi_agents(cfg):

    base_dir = "./evaluations/"
    experiment_name = cfg.checkpoint.split("/")[1]
    print(f'Experiment name: {experiment_name}')
    evaluation_dir = base_dir + experiment_name + "/"
    os.makedirs(evaluation_dir, exist_ok=True)

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    print(rlg_config_dict)
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    agent = runner.create_player()
    agent.restore(cfg.checkpoint)

    store_all_agents = True # store all agents generated data, if false only the first agent is stored
    is_done = False
    env = agent.env
    obs = env.reset()

    ep_data = {'act': [], 'obs': [], 'rews': [], 'info': [], 'all_dist': []}
    total_reward = 0
    num_steps = 0
    
    total_num_steps = 800
    for _ in range(total_num_steps):
        actions = agent.get_action(obs['obs'], is_deterministic=True)
        obs, reward, done, info = env.step(actions)
        
        #print(f'Step {num_steps}: obs={obs["obs"]}, rews={reward}, dones={done}, info={info} \n')
        if store_all_agents:
            ep_data['act'].append(actions.cpu().numpy())
            ep_data['obs'].append(obs['obs']['state'].cpu().numpy())
            ep_data['rews'].append(reward.cpu().numpy())  
        else:
            ep_data['act'].append(actions[0].cpu().numpy())
            ep_data['obs'].append(obs['obs']['state'][0].cpu().numpy())
            ep_data['rews'].append(reward[0].cpu().numpy())
        #ep_data['info'].append(info)
        x_pos = obs['obs']['state'][:,6].cpu().numpy()
        y_pos = obs['obs']['state'][:,7].cpu().numpy()
        ep_data['all_dist'].append(np.linalg.norm(np.array([x_pos, y_pos]), axis=0))
        total_reward += reward[0]
        num_steps += 1
        is_done = done.any()
    ep_data['obs'] = np.array(ep_data['obs'])
    ep_data['act'] = np.array(ep_data['act'])
    ep_data['rews'] = np.array(ep_data['rews'])
    ep_data['all_dist'] = np.array(ep_data['all_dist'])

    print(f'\n Episode: rew_sum={total_reward:.2f}, tot_steps={num_steps} \n')
    #print(f'Episode data: {ep_data} \n')
    print(f'Episode data obs shape: {ep_data["obs"].shape} \n')
    plot_episode_data_virtual(ep_data, evaluation_dir, store_all_agents)


def activate_wandb(cfg, cfg_dict, task):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"evaluate_{task.name}_{time_str}"
    cfg.wandb_entity = "matteohariry"
    wandb.tensorboard.patch(root_logdir="evaluations")
    wandb.init(
        project=task.name,
        entity=cfg.wandb_entity,
        config=cfg_dict,
        sync_tensorboard=True,
        name=run_name,
        resume="allow",
        group="eval_agents"
    )
    
@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    if cfg.checkpoint is None:
        print("No checkpoint specified. Exiting...")
        return

    # set congig params for evaluation
    """  cfg.task.env.maxEpisodeLength = 802
    cfg.task.env.clipObservations['state'] = 20.0
    cfg.task.env.task_parameters[0]['max_spawn_dist'] = 5.0
    cfg.task.env.task_parameters[0]['min_spawn_dist'] = 4.5  
    cfg.task.env.task_parameters[0]['kill_dist'] = 6.0
    cfg.task.env.task_parameters[0]['kill_after_n_steps_in_tolerance'] = 400
    # TODO: check error with visualizer of thrusters....
    #cfg.task.env.platform.configuration.visualize = False """


    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # _____Create environment_____

    headless = cfg.headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)

    #env.task._task_parameters.max_spawn_dist = 5.0
    
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    task = initialize_task(cfg_dict, env)
    # Activate wandb logging
    if cfg.wandb_activate:
        activate_wandb(cfg, cfg_dict, task)

    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    # _____Create players (model)_____
    
    #eval_single_agent(cfg_dict, cfg, env)
    eval_multi_agents(cfg)

    if cfg.wandb_activate:
        wandb.finish()
    env.close()

if __name__ == '__main__':
    parse_hydra_configs()