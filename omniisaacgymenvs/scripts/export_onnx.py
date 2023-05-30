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

import onnx
import onnxruntime as ort

import datetime
import os
import torch
import numpy as np



class ModelWrapper(torch.nn.Module):
    '''
    Main idea is to ignore outputs which we don't need from model
    '''
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
        
        
    def forward(self,input_dict):
        input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
        '''
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        '''
        #print(input_dict)
        #output_dict = self._model.a2c_network(input_dict)
        #input_dict['is_train'] = False
        #return output_dict['logits'], output_dict['values']
        return self._model.a2c_network(input_dict)
    
    
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

        # dump config dict
        experiment_dir = os.path.join('runs', self.cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        agent = runner.create_player()
        agent.restore(self.cfg.checkpoint)

        import rl_games.algos_torch.flatten as flatten
        inputs = {
            'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
            'rnn_states' : agent.states
        }
        with torch.no_grad():
            adapter = flatten.TracingAdapter(ModelWrapper(agent.model), inputs,allow_non_tensor=True)
            traced = torch.jit.trace(adapter, adapter.flattened_inputs,check_trace=False)
            flattened_outputs = traced(*adapter.flattened_inputs)
            print(flattened_outputs)
        
        torch.onnx.export(traced, *adapter.flattened_inputs, "two_wheeled_robot.onnx", verbose=True, input_names=['obs'], output_names=['mu', 'sigma', 'value'])

        onnx_model = onnx.load("two_wheeled_robot.onnx")

        # Check that the model is well formed
        onnx.checker.check_model(onnx_model)

        ort_model = ort.InferenceSession("two_wheeled_robot.onnx")
        env = agent.env
        total_reward = 0
        num_steps = 0
        for i in range(5):
            obs=env.reset()
            is_done=False
            while not is_done:
                print("obs: ",obs["obs"].cpu().numpy())
                obs=obs["obs"].cpu().numpy()
                
                """ save1=obs[0][0]
                save2=obs[0][1]
                obs[0][0]=save2
                obs[0][1]=save1 """
                
                outputs = ort_model.run(None, {"obs": obs},)
                print("outputs received: ",outputs)
                
                """ np_outputs0=np.array(outputs[0])
                np_outputs1=np.array(outputs[1])
               
                mu_lin_vel=np_outputs0[0][0]
                mu_ang_vel=np_outputs0[0][1]

                print(mu_lin_vel)
                print(mu_ang_vel)

                sigma_lin_vel=np.exp(np_outputs1[0][0])
                sigma_ang_vel=np.exp(np_outputs1[0][1])

                action_lin=np.random.normal(mu_lin_vel,sigma_lin_vel)
                action_ang=np.random.normal(mu_ang_vel,sigma_ang_vel)  """
                

                action=outputs[0]

                """ 
                mu = outputs[0].squeeze(0)
                sigma = np.exp(outputs[1].squeeze(0))
                action = np.random.normal(mu, sigma)
                action=-action
                action[1]=-action[1] """

                #action=torch.tensor([[action_lin,action_ang]])
                action=torch.tensor(action)
                #action = torch.tensor(outputs[0])

                action[0]=abs(action[0])

                print("action sent: ", action)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                num_steps += 1
                is_done = done

            print(total_reward, num_steps)
       


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = rank
        cfg.rl_device = f'cuda:{rank}'
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    task = initialize_task(cfg_dict, env)

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run_name = f"{cfg.wandb_name}_{time_str}"

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
        )


    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg_trainer.run()
    env.close()

    if cfg.wandb_activate and rank == 0:
            wandb.finish()

if __name__ == '__main__':
    parse_hydra_configs()