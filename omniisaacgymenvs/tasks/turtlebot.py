from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.nucleus import get_assets_root_path

import numpy as np
import torch
import math
from gym import spaces


class TurtlebotTask(RLTask):
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
        self._max_episode_length = self._task_cfg["env"]["learn"]["episodeLength_s"]
        
        self._num_observations =  2 
        self._num_actions = 2

        self.dt = 1/120

        self._diff_controller = DifferentialController(name="simple_control",wheel_radius=0.033, wheel_base=2*0.178)

        RLTask.__init__(self, name, env)

        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # init tensors that need to be set to correct device
        self.prev_goal_distance = torch.zeros(self._num_envs).to(self._device)
        self.prev_heading = torch.zeros(self._num_envs).to(self._device)
        self.weird_offset=0.35
        self.target_position = torch.tensor([0.0, 0.0, self.weird_offset]).to(self._device)
        self.robot_position=torch.tensor([0.0, 0.0, 0.00]).to(self._device)

        return

    def set_up_scene(self, scene) -> None:

        """Add prims to the scene and add views for interracting with them. Views are useful to interract with multiple prims at once."""
        
        self.add_prims_to_stage(scene)
        super().set_up_scene(scene)
        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/wheeled_robot", name="wheeled_robot_view")
        self._targets = GeometryPrimView(prim_paths_expr="/World/envs/.*/target_cube", name="target_view")
        scene.add(self._robots)

    def add_prims_to_stage(self, scene):

        #here specify the asset path of the wheeled robot
        turtlebot_asset_path = "/home/axelcoulon/projects/assets/turtlebot/turtlebot.usd"

        #import wheeled robot
        scene.add(
            WheeledRobot(
                prim_path= self.default_zero_env_path + "/wheeled_robot",
                name="wheeled_robot",
                wheel_dof_indices=[0,1],
                wheel_dof_names=["wheel_left_joint", "wheel_right_joint"],
                create_robot=True,
                usd_path=turtlebot_asset_path,
                position=self.robot_position,
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
        )
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
       
    def get_observations(self) -> dict:

        self.positions, self.rotations = self._robots.get_world_poses()
        self.target_positions, _ = self._targets.get_world_poses()

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

        obs = torch.hstack((self.headings.unsqueeze(1), self.goal_distances.unsqueeze(1)))
        self.obs_buf[:] = obs

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:

        """Perform actions to move the robot."""
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        controls = torch.zeros((self._num_envs, 4))

         
        
        for i in range(self._num_envs):

            #for rl_games librairy player
            if(actions.size()==torch.Size([2])):
                lin_vel=actions[0].item()
                ang_vel=actions[1].item()

            if(actions.size()==torch.Size([1,2])):
                lin_vel=actions[i][0].item()
                ang_vel=actions[i][1].item() 
                
            """ lin_vel=actions[i][0].item()
            ang_vel=actions[i][1].item() """
            controls[i][:2] = torch.tensor([1.0,1.0])
            controls[i][2:] = self._diff_controller.forward([lin_vel,ang_vel])


        
            
        self._robots.set_joint_velocities(controls)
       

    def reset_idx(self, env_ids):
        """Resetting the environment at the beginning of episode."""
        num_resets = len(env_ids)

        self.goal_reached = torch.zeros(self._num_envs, device=self._device)

        # apply resets
        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        
        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

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

    def post_reset(self):
        """This is run when first starting the simulation before first episode."""
      
        # get some initial poses
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_target_pos, _ = self._targets.get_world_poses()

        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
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