import torch

class BuoyantObject:
    def __init__(self, num_envs):
            self._num_envs = num_envs
    
            return
        
    def compute_archimedes_simple(self, mass, gravity):

        archimedes=torch.zeros(3)
        archimedes[2] = - gravity * mass
        return archimedes
        
    def compute_archimedes(self, density_water, submerged_volume, gravity):

        archimedes=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        archimedes[:,2] = density_water * gravity * submerged_volume
        return archimedes
    
        
    def compute_thrusters_force(self):
        
        thrusters=torch.zeros((self._num_envs, 6), dtype=torch.float32)
        thrusters[:,1]=5.0
        thrusters[:,4]=-5.0

        return thrusters
    
    def rl_compute_thrusters_force(self, left_thruster_force, right_thruster_force):

        thrusters=torch.zeros(6)
        thrusters[1]=left_thruster_force
        thrusters[4]=right_thruster_force

        return thrusters
        
    def compute_drag(self, boat_velocities):

        eps= 0.00000001
        
        x_velocity=boat_velocities[:,0]
        y_velocity=boat_velocities[:,1]
        z_velocity=boat_velocities[:,2]

        x_rotation = boat_velocities[:,3]
        y_rotation = boat_velocities[:,4]
        z_rotation = boat_velocities[:,5]

        norme = torch.sqrt(x_velocity**2 + y_velocity**2 + z_velocity**2)

        x_unit= x_velocity / (norme + eps)
        y_unit = y_velocity / (norme + eps)
        z_unit = z_velocity / (norme + eps) 

        coeff_drag_x = 3.0
        coeff_drag_y = 3.0
        coeff_drag_z = 10.0

        drag=torch.zeros((self._num_envs, 6), dtype=torch.float32)
        
        """ drag[:,0]=-x_unit*coeff_drag_x*x_velocity*x_velocity
        drag[:,1]=-y_unit*coeff_drag_y*y_velocity*y_velocity
        drag[:,2]=-z_unit*coeff_drag_z*z_velocity*z_velocity  """

        drag[:,0]=-coeff_drag_x*abs(x_velocity)*x_velocity
        drag[:,1]=-coeff_drag_y*abs(y_velocity)*y_velocity
        drag[:,2]=-coeff_drag_z*abs(z_velocity)*z_velocity 

        drag[:,3]=-coeff_drag_x*abs(x_rotation)*x_rotation
        drag[:,4]=-coeff_drag_y*abs(y_rotation)*y_rotation
        drag[:,5]=-coeff_drag_z*abs(z_rotation)*z_rotation 

        return drag
    
    def stabilize_boat(self,yaws):
         
        K=5
        drag=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        
        drag[:,0] = - K * yaws[:,0]
        drag[:,1] = - K * yaws[:,1]
        #drag[:,2] = - K * yaws[:,2]

        return drag

    


