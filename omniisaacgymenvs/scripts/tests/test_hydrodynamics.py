import torch
from omniisaacgymenvs.envs.BuoyancyPhysics.Hydrodynamics import *

# Valeurs de test
num_envs = 3
squared_drag_coefficients = [5.0, 5.0, 0.002]
linear_damping = [-16.44998712, -15.79776044, -100,-13,-13, -6]
quadratic_damping = [-2.942, -2.7617212, -10, -5, -5, -5]
linear_damping_forward_speed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
offset_linear_damping = 0.0
offset_lin_forward_damping_speed = 0.0
offset_nonlin_damping = 0.0
scaling_damping = 1.0
offset_added_mass = 0.0
scaling_added_mass = 1.0
world_velocities = torch.tensor([[1.0,1.0,0.0,0.0,0.0,0.0],
                                [0.5, 1.0, 0.0, 0.4, 0.1, 0.0],
                                [1.0,0.5,0.0,0.0,0.0,0.0]])

time= 0.01

hydrodynamics = HydrodynamicsObject(num_envs, squared_drag_coefficients, linear_damping, quadratic_damping, linear_damping_forward_speed, offset_linear_damping, offset_lin_forward_damping_speed, offset_nonlin_damping, scaling_damping, offset_added_mass, scaling_added_mass)

#correspond à une rotation de 45 degrés
roll_pitch_yaws = torch.tensor([[0.785, 0.0, 0.0], [0.0, 0.785, 0.0], [0.785, 0.785, 0.0]], dtype=torch.float32)
quaternions = torch.tensor([[ 0.924, 0.383, 0.0, 0.0], [0.924, 0.0, 0.383, 0.0], [  0.854,  0.354,  0.354, 0.146]], dtype=torch.float32)


drag = hydrodynamics.ComputeHydrodynamicsEffects(time, quaternions, world_velocities)

print("drag:", drag)