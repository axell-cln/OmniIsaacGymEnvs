import torch
from omniisaacgymenvs.envs.BuoyancyPhysics.Buoyancy_physics import *

# Valeurs de test
num_envs = 3
water_density = 1000.0
gravity = 9.81
metacentric_width = 0.5
metacentric_length = 0.7
submerged_volume = 0.14

roll_pitch_yaws = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]], dtype=torch.float32)
quaternions = torch.tensor([[0.969, 0.247, 0.0, 0.0], [0.969, 0.0, 0.247, 0.0], [0.969, 0.0, 0.0, 0.247]], dtype=torch.float32)

# Créer un objet de la classe BuoyantObject
buoyant_obj = BuoyantObject(num_envs, water_density, gravity, metacentric_width, metacentric_length)

# Appeler la fonction compute_archimedes_metacentric_global avec des valeurs de test
""" archimedes_force_global, archimedes_torque_global = buoyant_obj.compute_archimedes_metacentric_global(submerged_volume, roll_pitch_yaws)
print("Archimedes force (global):\n", archimedes_force_global)
print("Archimedes torque (global):\n", archimedes_torque_global) """

# Appeler la fonction compute_archimedes_metacentric_local avec des valeurs de test
archimedes_force_local, archimedes_torque_local = buoyant_obj.compute_archimedes_metacentric_local(submerged_volume, roll_pitch_yaws, quaternions)
print("Archimedes force (local):\n", archimedes_force_local)
print("Archimedes torque (local):\n", archimedes_torque_local)
