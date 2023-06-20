from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.ingenuity import Ingenuity
from omniisaacgymenvs.robots.articulations.views.ingenuity_view import IngenuityView

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.envs.buoyancy_physics import BuoyantObject
from omni.isaac.core.utils.stage import add_reference_to_stage

import omni
from omni.physx.scripts import utils
from pxr import UsdPhysics
from pxr import Gf

import numpy as np
import torch
import math


class BuoyancyTask(RLTask):
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
        self._max_episode_length = self._task_cfg["env"]["episodeLength_s"]

        self.dt = self._task_cfg["sim"]["dt"]

        #boxes physics
        self.gravity=self._task_cfg["sim"]["gravity"][2]
        
        self.box_density=self._task_cfg["sim"]["material_density"]
        
        self.box_width=self._task_cfg["sim"]["box_width"]
        self.box_large=self._task_cfg["sim"]["box_large"]
        self.box_high=self._task_cfg["sim"]["box_high"]

        self.box_volume=self.box_width*self.box_large*self.box_high
        self.box_mass=self.box_volume*self.box_density

        self.half_box_size=self.box_high/2

        self._num_observations = 7
        self._num_actions = 1

        self._box_position = torch.tensor([0, 0, 0.025])

        RLTask.__init__(self, name=name, env=env)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        return

    def set_up_scene(self, scene) -> None:
        
        self.get_target()
        self.get_buoyancy()
        RLTask.set_up_scene(self, scene)
        self._boxes = RigidPrimView(prim_paths_expr="/World/envs/.*/box/body", name="box_view", reset_xform_properties=False)
        self._thrusters_left= RigidPrimView(prim_paths_expr="/World/envs/.*/box/left_thruster", name="left_thruster_view", reset_xform_properties=False)
        self._thrusters_right= RigidPrimView(prim_paths_expr="/World/envs/.*/box/right_thruster", name="right_thruster_view", reset_xform_properties=False)
        scene.add(self._boxes)
        scene.add(self._thrusters_left)
        scene.add(self._thrusters_right)
        return


    def get_buoyancy(self):

        self.buoyancy_physics=BuoyantObject()

    def get_target(self):
    

        box_usd_path="/home/axelcoulon/projects/assets/box_thrusters.usd"
        box_prim_path=self.default_zero_env_path + "/box"
        add_reference_to_stage(prim_path=box_prim_path, usd_path=box_usd_path, prim_type="Xform")

        ############# this script below is provided in case you need to have a box.usd 

        """    box = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/box", 
            translation=self._box_position, 
            name="target_0",
            scale=np.array([0.2, 0.3, 0.05]),
            color=np.array([0.0, 0.0, 1.0])

        )
        
        stage = omni.usd.get_context().get_stage()
        # Get the prim
        cube_prim = stage.GetPrimAtPath(self.default_zero_env_path + "/box")
        # Enable physics on prim
        # If a tighter collision approximation is desired use convexDecomposition instead of convexHull
        utils.setRigidBody(cube_prim, "convexHull", False)
        mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
        mass_api.CreateMassAttr(self.box_mass)
        ### Alternatively set the density
        mass_api.CreateDensityAttr(self.box_density)
        # Same with COM
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        box.set_collision_enabled(False)
        #omni.usd.get_context().save_as_stage("/home/axelcoulon/projects/assets/box_saved.usd", None)  """


    def get_observations(self) -> dict:

        self.root_pos, self.root_rot = self._boxes.get_world_poses(clone=False)
        
        self.obs_buf[..., 0:3] = self.root_pos
        self.obs_buf[..., 3:7] = self.root_rot
      
        observations = {
            self._boxes.name: {
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


        actions = actions.clone().to(self._device)
        indices = torch.arange(self._boxes.count, dtype=torch.int64, device=self._device)
        box_pos, _= self._boxes.get_world_poses(clone=False)
        box_velocities=self._boxes.get_velocities(clone=False)
        archimedes=torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        drag_z=torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        drag_y=torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        thrusters=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

        for i in range(self._num_envs):
            if box_pos[i,2] < self.half_box_size:
                #body underwater
                water_density=1000 # kg/m^3
                high_submerged=self.half_box_size-box_pos[i,2]
                submerged_volume= high_submerged * self.box_width * self.box_large
                submerged_volume=torch.clamp(submerged_volume,0,self.box_volume).item()
                #print("submerged_volume",submerged_volume)
                archimedes[i,:]=self.buoyancy_physics.compute_archimedes(water_density, submerged_volume, -self.gravity)
                drag_z[i,:]=self.buoyancy_physics.compute_drag_archimedes_underwater(box_velocities[i,2])
                thrusters[i,:]=self.buoyancy_physics.compute_thrusters_force()
                drag_y[i,:]=self.buoyancy_physics.compute_drag_thrusters(box_velocities[i,1])
            
            else:
                archimedes[i,:]=self.buoyancy_physics.compute_archimedes(0.0,0.0,0.0)
                drag_z[i,:]=0.0
                drag_y[i,:]=0.0
                thrusters[i,:]=0.0
            
        #thrusters[:,:]=self.buoyancy_physics.compute_thrusters_force()
        
        #print("archimedes first box: ",archimedes[0,:])
        #print("drag_z first box: ", drag_z)

        forces= archimedes + drag_z + drag_y
        #print("forces: ", forces)
        #print("thrusters: ", thrusters)clea

        self._boxes.apply_forces_and_torques_at_pos(forces,indices=indices)
        self._thrusters_left.apply_forces_and_torques_at_pos(thrusters[:,:3],indices=indices)
        self._thrusters_right.apply_forces_and_torques_at_pos(thrusters[:,3:],indices=indices)

        print(thrusters[0,:])
        

    def post_reset(self):
        
        self.initial_box_pos, self.initial_box_rot = self._boxes.get_world_poses()
       

    def set_targets(self, env_ids):

        num_sets = len(env_ids)
        envs_long = env_ids.long()

        # set target position randomly z in (0.5, 1.5)
        self.target_positions[envs_long, 2] = torch.rand(num_sets, device=self._device) + 0.5

        # shift the target up so it visually aligns better
        box_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        
        box_pos[:, 2] += 0.4
        #box_pos[:, 2] = 0.05

        self._boxes.set_world_poses(box_pos[:, 0:3], self.initial_box_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):

        self.set_targets(env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:

        self.rew_buf[:] = 0.0

    def is_done(self) -> None:

        #no resets for now

        """Flags the environnments in which the episode should end."""
        #resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, self.reset_buf.double())
        #self.reset_buf[:] = resets
