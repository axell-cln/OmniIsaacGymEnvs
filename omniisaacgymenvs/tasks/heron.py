from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.views.heron_view import HeronView
from omniisaacgymenvs.robots.articulations.heron import Heron
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import RigidPrimView
from omniisaacgymenvs.envs.BuoyancyPhysics.Buoyancy_physics import BuoyantObject

import numpy as np
import torch
import math
from gym import spaces

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

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength_s"]

        self.dt = self._task_cfg["sim"]["dt"]

        #physics
        self.gravity=self._task_cfg["sim"]["gravity"][2]
        self.box_density=self._task_cfg["sim"]["material_density"]
        self.water_density=1000 # kg/m^3

        
        #approximation of the submerged volume
        self.box_width=self._task_cfg["sim"]["box_width"]
        self.box_large=self._task_cfg["sim"]["box_large"]
        self.box_high=self._task_cfg["sim"]["box_high"]
        self.box_volume=self.box_width*self.box_large*self.box_high
        self.half_box_size=self.box_high/2

        #task specifications 
        self._num_observations =  2 
        self._num_actions = 2

        RLTask.__init__(self, name, env)

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # init tensors that need to be set to correct device

        #task learning
        self.prev_goal_distance = torch.zeros(self._num_envs).to(self._device)
        self.prev_heading = torch.zeros(self._num_envs).to(self._device)

        #positions 
        self._heron_position=torch.tensor([0.0, 0.0, 0.035]).to(self._device)
        self.target_position = torch.tensor([0.0, 0.0, 0.025]).to(self._device)
        self.left_thruster_position = torch.tensor([-0.53, 0.37765, -0.16])
        self.right_thruster_position = torch.tensor([-0.53, -0.37765, -0.16])

        #forces
        self.archimedes=torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.drag=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self.thrusters=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self.high_submerged=torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self.submerged_volume=torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self.stable_torque=torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)


        return

    def set_up_scene(self, scene) -> None:

        """Add prims to the scene and add views for interracting with them. Views are useful to interract with multiple prims at once."""
        
        self.get_heron()
        self.get_buoyancy()
        self.get_target(scene)
        RLTask.set_up_scene(self, scene)
        self._targets= GeometryPrimView(prim_paths_expr="/World/envs/.*/target_cube", name="target_view")
        self._thrusters_left= RigidPrimView(prim_paths_expr="/World/envs/.*/heron/thruster_1", name="left_thruster_view", reset_xform_properties=False)
        self._thrusters_right= RigidPrimView(prim_paths_expr="/World/envs/.*/heron/thruster_0", name="right_thruster_view", reset_xform_properties=False)
        self._herons= RigidPrimView(prim_paths_expr="/World/envs/.*/heron/dummy_link", name="body_view", reset_xform_properties=False)
        scene.add(self._herons)
        scene.add(self._thrusters_left)
        scene.add(self._thrusters_right)

    def get_target(self, scene):

        #create target and import
        scene.add(
            VisualCuboid(
                prim_path=self.default_zero_env_path + "/target_cube",
                name="target_cube",
                position=self.target_position,
                size=0.1,
                color=np.array([1.0, 0, 0]),
            )
        )

    def get_buoyancy(self):
        self.buoyancy_physics=BuoyantObject(self.num_envs)
    
    def get_heron(self):
        heron = Heron(prim_path=self.default_zero_env_path + "/heron", name="heron", translation=self._heron_position)
           
    def get_observations(self) -> dict:

        self.positions, self.rotations = self._herons.get_world_poses()
        self.target_positions, _ = self._targets.get_world_poses()
        
        self.positions[:,2]=0.0
        self.target_positions[:,2]=0.0

        self.yaws = []
        for rot in self.rotations:
            self.yaws.append(quat_to_euler_angles(rot)[2])
        self.yaws = torch.tensor(self.yaws).to(self._device)

        goal_angles = torch.atan2(self.target_positions[:,1] - self.positions[:,1], self.target_positions[:,0] - self.positions[:,0])

        self.headings = goal_angles - self.yaws
        self.headings = torch.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
        self.headings = torch.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings)

        self.goal_distances = torch.linalg.norm(self.positions - self.target_positions, dim=1).to(self._device)
        
        to_target = self.target_positions - self.positions
        to_target[:, 2] = 0.0

        self.prev_potentials[:] = self.potentials.clone()
        self.potentials[:] = -torch.norm(to_target, p=2, dim=-1) / self.dt

        obs = torch.hstack((self.headings.unsqueeze(1), self.goal_distances.unsqueeze(1)))
        self.obs_buf[:] = obs

        observations = {
            self._herons.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def get_angles(self, quaternions):
        
        yaws = np.zeros((self.num_envs,3), dtype=float)
        for i in range(self._num_envs):
            yaws[i,:]=quat_to_euler_angles(quaternions[i,:])
        return torch.tensor(yaws).to(self._device)
    
    def pre_physics_step(self, actions) -> None:

        """Perform actions to move the robot."""

        if not self._env._world.is_playing():
            return
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)
        indices = torch.arange(self._herons.count, dtype=torch.int64, device=self._device)
        
        heron_poses, heron_quaternions = self._herons.get_world_poses(clone=False)
        heron_velocities = self._herons.get_velocities(clone=False)

        yaws = self.get_angles(heron_quaternions)

        
        #body underwater
        self.high_submerged[:]=torch.clamp(self.half_box_size-heron_poses[:,2], 0, self.box_high)
        self.submerged_volume[:]= torch.clamp(self.high_submerged * self.box_width * self.box_large, 0, self.box_volume)
        self.archimedes[:,:]=self.buoyancy_physics.compute_archimedes(self.water_density, self.submerged_volume, -self.gravity)
        self.stable_torque[:,:]=self.buoyancy_physics.stabilize_boat(yaws)
        self.thrusters[:,:]=self.buoyancy_physics.command_to_thrusters_force(0.2,0.2)
        self.drag[:,:]=self.buoyancy_physics.compute_drag(heron_velocities[:,:])
                   
        forces_applied_on_center= self.archimedes + self.drag[:,:3]
        self._herons.apply_forces_and_torques_at_pos(forces=forces_applied_on_center, torques=self.drag[:,3:] + self.stable_torque)
        self._thrusters_left.apply_forces_and_torques_at_pos(self.thrusters[:,:3])
        self._thrusters_right.apply_forces_and_torques_at_pos(self.thrusters[:,3:])

    
    def reset_herons(self, env_ids):

        num_sets = len(env_ids)
        envs_long = env_ids.long()

        heron_pos = self.initial_root_pos[envs_long] + self._env_pos[envs_long]

        self._herons.set_world_poses(heron_pos[:, 0:3], self.initial_root_rot[envs_long].clone(), indices=env_ids)


    def reset_targets(self, env_ids):

        num_resets = len(env_ids)

        target_pos=torch.zeros((num_resets, 3), device=self._device)
        
        for i in range(num_resets):
            # randomize goal location in circle around robot
            alpha = 2 * math.pi * np.random.rand()
            r = 3.50 * math.sqrt(np.random.rand()) + 0.20
            random_target_pos=torch.tensor([math.sin(alpha) * r, math.cos(alpha) * r, 0.025]).to(self._device)
            target_pos[i,:] = self.initial_target_pos[i,:] + random_target_pos

        self._targets.set_world_poses(target_pos, indices=env_ids)

        to_target = target_pos - self.initial_root_pos[env_ids]
        to_target[:, 2] = 0.0

        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()



    def reset_idx(self, env_ids):
        """Resetting the environment at the beginning of episode."""
        
        self.goal_reached = torch.zeros(self._num_envs, device=self._device)

        # apply resets
        self.reset_herons(env_ids)

        self.reset_targets(env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

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
        

    def calculate_metrics(self) -> None:
        """Calculate rewards for the RL agent."""
        rewards = torch.zeros_like(self.rew_buf)

        self.prev_goal_distance = self.goal_distances
        self.goal_reached = torch.where(self.goal_distances < 0.15, 1, 0).to(self._device)

        self.prev_heading = self.headings

        progress_reward = self.potentials - self.prev_potentials
     
        episode_end = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, 0.0)
      
        rewards -= 10 * episode_end
        rewards += 0.1 * progress_reward
        rewards += 20 * self.goal_reached

        self.rew_buf[:] = rewards

    def is_done(self) -> None:
        """Flags the environnments in which the episode should end."""
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, self.reset_buf.double())
        self.reset_buf[:] = resets