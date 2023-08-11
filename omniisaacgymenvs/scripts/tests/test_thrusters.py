from omniisaacgymenvs.envs.BuoyancyPhysics.ThrusterDynamics import *
import matplotlib.pyplot as plt

time_constant = 0.05
num_envs = 1
dynamics = DynamicsFirstOrder(time_constant, num_envs)

commands = torch.tensor([1.0,-1.0]) 
intervals = 100
zero_command= intervals // 10

dt = [0.01 * i for i in range(intervals)]


# Obtenir les forces des propulseurs à partir des commandes
thrusters_forces = dynamics.command_to_thrusters_force(commands)

full_forward_force_applied = []
for j in range(intervals):
    if(j >= zero_command):
        full_forward_force_applied.append(dynamics.command_to_thrusters_force(commands)[:,3].item())
    else:
        full_forward_force_applied.append(0.0)

commands_for_plot = torch.full((intervals,), full_forward_force_applied[-1])
commands_for_plot[:10]=0.0


print("real_forces_applied:", full_forward_force_applied)
# Tracer le graphique
plt.plot(dt, commands_for_plot, label='commanded force')
plt.plot(dt, full_forward_force_applied, label='real force')

# Ajouter des titres et des légendes
plt.title('Thruster first order model')
plt.xlabel("Simuling time dt(secondes)")
plt.ylabel("Thrusters force (N)")
plt.legend()

# Afficher le graphique
plt.savefig('test_thrusters_dynamics_backward_command.png.png')
