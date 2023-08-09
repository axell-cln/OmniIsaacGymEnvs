from omniisaacgymenvs.envs.BuoyancyPhysics.ThrusterDynamics import *


time_constant = 0.05
num_envs = 2
dynamics = DynamicsFirstOrder(time_constant, num_envs)

commands = torch.tensor([[1.0,0.0],
                         [0.0, 0.5]]) 

# Obtenir les forces des propulseurs à partir des commandes
thrusters_forces = dynamics.command_to_thrusters_force_lookup_table(commands)

print("Forces des propulseurs:")
print(thrusters_forces)