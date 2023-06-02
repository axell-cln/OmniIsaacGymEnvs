from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

import hydra
from omegaconf import DictConfig

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import datetime
import os
import torch
import numpy as np
import pandas as pd

class RLGTrainer():
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self, env):
        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        self.cfg_dict["task"]["test"] = self.cfg.test

        # register the rl-games adapter to use inside the runner
        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: env
        })

        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self):
        # create runner and set the settings
        runner = Runner(RLGPUAlgoObserver())
        runner.load(self.rlg_config_dict)
        runner.reset()

        agent = runner.create_player()
        agent.restore(self.cfg.checkpoint)

        is_done = False
        env = agent.env
        obs = env.reset()
        print(obs)
        #input()
        #prev_screen = env.render(mode='rgb_array')
        #plt.imshow(prev_screen)
        total_reward = 0
        num_steps = 0
        headings=[]
        distances=[]
        while not is_done:
            
            heading=obs["obs"].cpu().numpy()[0][0]
            distance=obs["obs"].cpu().numpy()[0][1]

            headings.append(heading)
            distances.append(distance)

            action = agent.get_action(obs['obs'], is_deterministic=True)
            obs, reward, done, info = env.step(action)

            
            print(f'Step {num_steps}: obs={obs["obs"].cpu().numpy()}, rews={reward}, dones={done}, info={info} \n')

            total_reward += reward
            num_steps += 1
            is_done = done

        print(total_reward, num_steps)
        df = pd.DataFrame({'Heading': headings, 'Distance': distances})
        output_file = 'player_observations.xlsx'

        # Save the DataFrame to Excel
        df.to_excel(output_file, index=False)



@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg.checkpoint = "/home/axelcoulon/projects/OmniIsaacGymEnvs/runs/Jetbot/save_64_64/nn/Jetbot.pth"

    headless = cfg.headless
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    task = initialize_task(cfg_dict, env)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)


    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg_trainer.run()
    env.close()


if __name__ == '__main__':
    parse_hydra_configs()