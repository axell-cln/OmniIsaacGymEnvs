import torch
import math

class BuoyantObject:
    def __init__(self):

        return
         
    def compute_archimedes_simple(self, mass, gravity):

        archimedes=torch.zeros(3)
        archimedes[2] = - gravity * mass
        return archimedes
        
    def compute_archimedes(self, density_water, submerged_volume, gravity):

        archimedes=torch.zeros(3)
        archimedes[2] = density_water * gravity * submerged_volume
        return archimedes
    
        
    def compute_thrusters_force(self):
        
        thrusters=torch.zeros(6)
        thrusters[1]=-5.0
        thrusters[4]=-5.0

        return thrusters
        
        
    def compute_drag(self, boat_velocities):

        x_velocity=boat_velocities[0].item()
        y_velocity=boat_velocities[1].item()
        z_velocity=boat_velocities[2].item()

        norme = math.sqrt(x_velocity**2 + y_velocity**2 + z_velocity**2)

        x_unit= x_velocity / norme
        y_unit = y_velocity / norme
        z_unit = z_velocity / norme

        coeff_drag_x = 1.0
        coeff_drag_y = 1.0
        coeff_drag_z = 10.0

        drag=torch.zeros(3)
        
        drag[0]=-x_unit*coeff_drag_x*x_velocity*x_velocity
        drag[1]=-y_unit*coeff_drag_y*y_velocity*y_velocity
        drag[2]=-z_unit*coeff_drag_z*z_velocity*z_velocity

        return drag

    


