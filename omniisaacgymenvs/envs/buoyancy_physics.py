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
    
    def compute_drag_archimedes_underwater(self, z_velocity):
        
        coeff=10.0
        drag=torch.zeros(3)
        drag[2]=coeff*z_velocity*z_velocity
        if z_velocity<0:
            return drag
        else:
            return -drag
        
    def compute_thrusters_force(self):
        
        thrusters=torch.zeros(6)
        thrusters[1]=5.0
        thrusters[4]=5.0

        return thrusters
    
    def compute_drag_thrusters(self, y_velocity):

        coeff=1.0
        drag=torch.zeros(3)
        drag[1]=coeff*y_velocity*y_velocity
        if y_velocity<0:
            return drag
        else:
            return -drag
    


