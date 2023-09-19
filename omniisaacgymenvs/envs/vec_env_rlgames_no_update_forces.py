from omni.isaac.gym.vec_env import VecEnvBase

import torch
import numpy as np

from datetime import datetime


# VecEnv Wrapper for RL training
class VecEnvRLGames(VecEnvBase):

    def _process_data(self):
        if type(self._obs) is dict:
            if type(self._task.clip_obs) is dict:
                for k,v in self._obs.items():
                    if k in self._task.clip_obs.keys():
                        self._obs[k] = v.float() / 255.0
                        self._obs[k] = torch.clamp(v, -self._task.clip_obs[k], self._task.clip_obs[k]).to(self._task.rl_device).clone()
                    else:
                        self._obs[k] = v
        else:
            self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
            self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()

        self._rew = self._rew.to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

    def step(self, actions):
        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)

        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()

        self._task.pre_physics_step(actions)
        
        for _ in range(self._task.control_frequency_inv - 1):
            self._world.step(render=False)
            #self._task.update_state()
            #self._task.apply_forces()
            self.sim_frame_count += 1

        self._world.step(render=self._render)
        self.sim_frame_count += 1

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(
                observations=self._obs.to(device=self._task.rl_device), reset_buf=self._task.reset_buf)

        self._states = self._task.get_states()
        self._process_data()
        
        obs_dict = {"obs": self._obs, "states": self._states}

        return obs_dict, self._rew, self._resets, self._extras

    def reset(self):
        """ Resets the task and applies default zero actions to recompute observations and states. """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        self._task.reset()
        actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.rl_device)
        obs_dict, _, _, _ = self.step(actions)

        return obs_dict