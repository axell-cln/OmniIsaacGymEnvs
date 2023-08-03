import torch

class HydrodynamicsObject:

    def __init__(self, num_envs, drag_coefficients, linear_damping, quadratic_damping, linear_damping_forward_speed, offset_linear_damping, offset_lin_forward_damping_speed, offset_nonlin_damping, scaling_damping):
            self._num_envs = num_envs
            self.drag=torch.zeros((self._num_envs, 6), dtype=torch.float32)
            self.drag_coefficients = drag_coefficients
            self.linear_damping=torch.eye(6) * linear_damping
            self.quadratic_damping=torch.eye(6) * quadratic_damping
            self.linear_damping_forward_speed = torch.eye(6) * linear_damping_forward_speed
            self.offset_linear_damping = offset_linear_damping
            self.offset_lin_forward_damping_speed = offset_lin_forward_damping_speed
            self.offset_nonlin_damping = offset_nonlin_damping
            self.scaling_damping = scaling_damping
            
            return
        
    def compute_drag(self, boat_velocities):

        """this function implements the drag, rotation drag is needed because of where archimedes is applied. if the boat start to rate around x for 
        exemple, since archimedes is applied onto the center, isaac sim will believe that the boat is still under water and so the boat is free to rotate around and
        y. So to prevent this behaviour, if we don't want to create 4 rigid bodies as talked above, we are forced to add a drag + stabilizer to the simulation."""
        
        #coefficients  = 0.5 * ρ * v^2 * A * Cd

        """ρ (rho) is the density of the surrounding fluid in kg/m³.
        v is the velocity of the object relative to the fluid in m/s.
        A is the reference area of the object perpendicular to the direction of motion in m². This is usually the frontal area of the object exposed to the fluid flow.
        Cd is the drag coefficient (dimensionless) that depends on the shape and roughness of the object. This coefficient is often determined experimentally."""

        """for our boxes, A ~ 0.2 , ρ ~ 1000, Cd ~ 0.05 """

        xyz_velocities = boat_velocities[:,:3]
        xyz_rotations = boat_velocities[:,3:]

        self.drag[:,:3]= - self.drag_coefficients * torch.abs(xyz_velocities) * xyz_velocities
        self.drag[:, 3:]= - self.drag_coefficients * torch.abs(xyz_rotations) * xyz_rotations

        return self.drag
    
    def ComputeDampingMatrix(self, vel):
        """
        // From Antonelli 2014: the viscosity of the fluid causes
        // the presence of dissipative drag and lift forces on the
        // body. A common simplification is to consider only linear
        // and quadratic damping terms and group these terms in a
        // matrix Drb
        """
        self.drag =  -1 * (self.linear_damping+ self.offset_linear_damping * torch.eye(6))\
            - vel[0] * (self.linear_damping_forward_speed + self.offset_lin_forward_damping_speed * torch.eye(6))
        
        # Nonlinear damping matrix is considered as a diagonal matrix
        self.drag += -1 * (self.quadratic_damping + self.offset_nonlin_damping * torch.eye(6))* torch.abs(vel)
        
        # adding both matrices 
        self.drag = self.drag*self.scaling_damping

        return self.drag
    

    def computeCoriolis(self):

        """Implementation of coriolis force"""
        
        return
    
    #Only if archimedes torque is not applied 

    """
    def stabilize_boat(self,yaws):
        # Roll Stabilizing Force = -k_roll * θ_x, Yaw Stabilizing Force = -k_yaw * θ_z 

        K=5.0 #by hand
        force=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        
        force[:,0] = - K * yaws[:,0]
        force[:,1] = - K * yaws[:,1]

        return force
    """

    


