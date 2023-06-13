# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.heron import Heron
from omniisaacgymenvs.robots.articulations.views.heron_view import HeronView

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math


class HeronTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self.thrust_limit = 2000
        self.thrust_lateral_component = 0.2

        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 4 
        self._num_actions = 2

        self._boat_position = torch.tensor([0, 0, 0.05])
        self._goal_position = torch.tensor([2.0, 2.0, 0.0])

        RLTask.__init__(self, name=name, env=env)

        self.force_indices = torch.tensor([0, 2], device=self._device)
        self.spinning_indices = torch.tensor([0, 2], device=self._device)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
     
     
        """ self.target_positions[:, 0] = 2.0
        self.target_positions[:, 1] = 2.0
        self.target_positions[:, 2] = 0.0 """
        
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        return

    def set_up_scene(self, scene) -> None:
        self.get_heron()
        self.get_target()
        RLTask.set_up_scene(self, scene)
        self._herons = HeronView(prim_paths_expr="/World/envs/.*/heron", name="heron_view")
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False)
        scene.add(self._herons)
        scene.add(self._targets)
        

    def get_heron(self):
        heron = Heron(prim_path=self.default_zero_env_path + "/heron", name="heron", translation=self._boat_position)
        self._sim_config.apply_articulation_settings("heron", get_prim_at_path(heron.prim_path), self._sim_config.parse_actor_config("heron"))

    def get_target(self):
        radius = 0.1
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball", 
            translation=self._goal_position, 
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:

        self.positions, self.rotations = self._herons.get_world_poses()
        self.target_positions, _ = self._targets.get_world_poses()

        self.herons_velocities = self._herons.get_velocities(clone=False)

        self.positions[:,2]=0.0
        self.target_positions[:,2]=0.0

        yaws = []
        for rot in self.rotations:
            yaws.append(quat_to_euler_angles(rot)[2])
        yaws = torch.tensor(yaws).to(self._device)

        goal_angles = torch.atan2(self.target_positions[:,1] - self.positions[:,1], self.target_positions[:,0] - self.positions[:,0])

        self.headings = goal_angles - yaws
        self.headings = torch.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
        self.headings = torch.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings)

        self.goal_distances = torch.linalg.norm(self.positions - self.target_positions, dim=1).to(self._device)
        

        to_target = self.target_positions - self.positions
        to_target[:, 2] = 0.0

        self.prev_potentials[:] = self.potentials.clone()
        self.potentials[:] = -torch.norm(to_target, p=2, dim=-1) / self.dt

        obs = torch.hstack((self.headings.unsqueeze(1), self.goal_distances.unsqueeze(1), self.herons_velocities.unsqueeze(1)))
        self.obs_buf[:] = obs

        observations = {
            self._herons.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)
        

        self.thrusts[:, 0] = 1.0
        self.thrusts[:, 1] = 1.0
    
        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0

        # spin spinning rotors
        self.dof_vel[:, self.spinning_indices[0]] = 50
        self.dof_vel[:, self.spinning_indices[1]] = 50
        self._herons.set_joint_velocities(self.dof_vel)

        # apply actions
        for i in range(2):
            self._herons.thrusters[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)

    def post_reset(self):
        """This is run when first starting the simulation before first episode."""
      
        # get some initial poses
        self.initial_root_pos, self.initial_root_rot = self._herons.get_world_poses()
        self.initial_target_pos, _ = self._targets.get_world_poses()

        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # randomize all envs
        indices = torch.arange(self._herons.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        #in case of randomizing
        """ if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self) """

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()

        target_pos=torch.zeros((num_sets, 3), device=self._device)
        
        for i in range(num_sets):
            # randomize goal location in circle around robot
            alpha = 2 * math.pi * np.random.rand()
            r = 1.50 * math.sqrt(np.random.rand()) + 0.20
            random_target_pos=torch.tensor([math.sin(alpha) * r, math.cos(alpha) * r, 0.025]).to(self._device)
            target_pos[i,:] = self.initial_target_pos[i,:] + random_target_pos

        self._targets.set_world_poses(target_pos, indices=env_ids)

    def reset_idx(self, env_ids):
        """Resetting the environment at the beginning of episode."""
        
        num_resets = len(env_ids)

        self.goal_reached = torch.zeros(self._num_envs, device=self._device)

        # apply resets
        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        
        self._herons.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._herons.set_velocities(root_vel, indices=env_ids)

        target_pos=torch.zeros((num_resets, 3), device=self._device)
        
        for i in range(num_resets):
            # randomize goal location in circle around robot
            alpha = 2 * math.pi * np.random.rand()
            r = 1.50 * math.sqrt(np.random.rand()) + 0.20
            random_target_pos=torch.tensor([math.sin(alpha) * r, math.cos(alpha) * r, 0.025]).to(self._device)
            target_pos[i,:] = self.initial_target_pos[i,:] + random_target_pos

        self._targets.set_world_poses(target_pos, indices=env_ids)

        to_target = target_pos - self.initial_root_pos[env_ids]
        to_target[:, 2] = 0.0

        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:

        heron_positions = self.heron_pos - self._env_pos
        root_quats = self.root_rot
        root_angvels = self.root_velocities[:, 3:]

        # distance to target
        target_dist = torch.sqrt(torch.square(self.target_positions - heron_positions).sum(-1))
        pos_reward = 1.0 / (1.0 + 2.5 * target_dist * target_dist)
        self.target_dist = target_dist
        self.heron_positions = heron_positions

        # uprightness
        ups = quat_axis(root_quats, 2)
        
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 1.0 / (1.0 + 30 * tiltage * tiltage)
  
        # spinning
        spinnage = torch.abs(root_angvels[..., 2])
        spinnage_reward = 1.0 / (1.0 + 10 * spinnage * spinnage)

        # combined reward
        # uprightness and spinning only matter when close to the target
        self.rew_buf[:] = pos_reward + pos_reward * (up_reward + spinnage_reward)

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 20.0, ones, die)
        die = torch.where(self.heron_positions[..., 2] < 0.5, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
