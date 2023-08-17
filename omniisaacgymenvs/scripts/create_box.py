############# this script below is provided in case you need to have a box.usd but have to be updated to the boat dimension

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