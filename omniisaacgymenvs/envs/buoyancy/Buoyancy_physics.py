import torch
import numpy as np 

class BuoyantObject:
    def __init__(self, num_envs):
            self._num_envs = num_envs
    
            return
        
        
    def compute_archimedes(self, density_water, submerged_volume, gravity):
        """This function apply the archimedes force to the center of the boat"""

        """Ideally, this function should not be applied only at the center of the boat, but at the center of the volume submerged underwater.
        In this case, if the boat start to rotate around x and y axis, the part of the boat who isn't underwater anymore have no force except gravity applied,
        it automatically balance the boat. But that would require to create 4 rigid body at each corner of the boat and then check which one of them is underwater.
        """
        archimedes=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        archimedes[:,2] = density_water * gravity * submerged_volume
        return archimedes
    
    def compute_archimedes_metacentric(self, density_water, submerged_volume, gravity, angles, metacentric_width, metacentric_length):
        """This function apply the archimedes force to the center of the boat"""

        """Ideally, this function should not be applied only at the center of the boat, but at the center of the volume submerged underwater.
        In this case, if the boat start to rotate around x and y axis, the part of the boat who isn't underwater anymore have no force except gravity applied,
        it automatically balance the boat. But that would require to create 4 rigid body at each corner of the boat and then check which one of them is underwater.
        """

        #print(angles[:,0])
        #print(metacentric_width)
        roll = angles[:,0].cpu()
        pitch = angles[:,1].cpu()

        archimedes=torch.zeros((self._num_envs, 6), dtype=torch.float32)

        archimedes[:,2] = density_water * gravity * submerged_volume
        archimedes[:,3] = -1 * metacentric_width * np.sin(roll) * archimedes[:,2]
        archimedes[:,4] = -1 * metacentric_length * np.sin(pitch) * archimedes[:,2]
        

        return archimedes
    
        

    def compute_drag(self, boat_velocities):

        """this function implements the drag, rotation drag is needed because of where archimedes is applied. if the boat start to rate around x for 
        exemple, since archimedes is applied onto the center, isaac sim will believe that the boat is still under water and so the boat is free to rotate around and
        y. So to prevent this behaviour, if we don't want to create 4 rigid bodies as talked above, we are forced to add a drag + stabilizer to the simulation."""
        
        
        
        eps= 0.00000001 #to avoid division by zero
        
        x_velocity=boat_velocities[:,0]
        y_velocity=boat_velocities[:,1]
        z_velocity=boat_velocities[:,2]

        x_rotation = boat_velocities[:,3]
        y_rotation = boat_velocities[:,4]
        z_rotation = boat_velocities[:,5]


        coeff_drag_x = 5.0   # Linear Drag  = 0.5 * ρ * v^2 * A * Cd
        coeff_drag_y = 5.0
        coeff_drag_z = 0.002

        """ρ (rho) is the density of the surrounding fluid in kg/m³.
        v is the velocity of the object relative to the fluid in m/s.
        A is the reference area of the object perpendicular to the direction of motion in m². This is usually the frontal area of the object exposed to the fluid flow.
        Cd is the drag coefficient (dimensionless) that depends on the shape and roughness of the object. This coefficient is often determined experimentally."""

        """for our boxes, A ~ 0.2 , ρ ~ 1000, Cd ~ 0.05 """
        
        drag=torch.zeros((self._num_envs, 6), dtype=torch.float32)
        
        drag[:,0]=-coeff_drag_x*abs(x_velocity)*x_velocity
        drag[:,1]=-coeff_drag_y*abs(y_velocity)*y_velocity
        drag[:,2]=-coeff_drag_z*abs(z_velocity)*z_velocity 

        drag[:,3]=-coeff_drag_x*abs(x_rotation)*x_rotation
        drag[:,4]=-coeff_drag_y*abs(y_rotation)*y_rotation
        drag[:,5]=-coeff_drag_z*abs(z_rotation)*z_rotation 

        return drag
    
    def stabilize_boat(self,yaws):
        """ Roll Stabilizing Force = -k_roll * θ_x, Yaw Stabilizing Force = -k_yaw * θ_z """

        K=5.0 #by hand
        force=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        
        force[:,0] = - K * yaws[:,0]
        force[:,1] = - K * yaws[:,1]

        return force

    


