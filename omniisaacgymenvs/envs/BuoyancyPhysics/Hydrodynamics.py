import torch
import pytorch3d

class HydrodynamicsObject:

    def __init__(self, num_envs, drag_coefficients, linear_damping, quadratic_damping, linear_damping_forward_speed, offset_linear_damping, offset_lin_forward_damping_speed, offset_nonlin_damping, scaling_damping, offset_added_mass, scaling_added_mass ):
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
            self._Ca = torch.zeros([6,6]) #dk what does it represents
            self.added_mass = torch.zeros([6,6])
            self.offset_added_mass = offset_added_mass
            self.scaling_added_mass=scaling_added_mass
            
            return
        
    def compute_squared_drag(self, boat_velocities):

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
        #adding both matrices 
        self.drag += -1 * (self.quadratic_damping + self.offset_nonlin_damping * torch.eye(6))* torch.abs(vel)
        
        # scaling 
        self.drag = self.drag*self.scaling_damping

        return self.drag
    

    def ComputeAddedCoriolisMatrix(self, vel):
        """
        // This corresponds to eq. 6.43 on p. 120 in
        // Fossen, Thor, "Handbook of Marine Craft and Hydrodynamics and Motion
        // Control", 2011
        """

        #Sa = torch.cross(ab[:3], torch.zeros([3,3]), dim=1)

        ab = torch.matmul(self.GetAddedMass(), vel)
        Sa = -1 * self.CrossProductOperator(ab[:3])
        self._Ca[-3:,:3] = Sa
        self._Ca[:3,-3:] = Sa
        self._Ca[-3:,-3:] = -1 * self.CrossProductOperator(ab[-3:])
        
        return
    
    def applyDrag(self, time, world_velocities, quaternions, vel):
         
        alpha = 0.3

        rot_mat = pytorch3d.transforms.quaternion_to_matrix(quaternions)
        rot_mat_inv = torch.linalg.inv(rot_mat)  #transpose 
        
        self.linVel = self.getRelativeLinearVel(vel[:,:3], rot_mat_inv)
        self.angVel = self.getRelativeAngularVel(vel[:,3:], rot_mat_inv)
        
        # Transform the flow velocity to the body frame
        flowVel = torch.matmul(rot_mat,world_velocities)
        # Compute the relative velocity
        velRel = torch.hstack([self.linVel - flowVel, self.angVel])
        # Update added Coriolis matrix
        self.ComputeAddedCoriolisMatrix(velRel)
        # Update damping matrix
        self.ComputeDampingMatrix(velRel)
        # Filter acceleration (see issue explanation above)
        self.ComputeAcc(velRel, time, alpha)
        # We can now compute the additional forces/torques due to this dynamic
        # effects based on Eq. 8.136 on p.222 of Fossen: Handbook of Marine Craft ...
        # Damping forces and torques
        damping =  torch.matmul(-self.drag, velRel)  #minus?
        # Added-mass forces and torques
        added = torch.matmul(-self.GetAddedMass(), self._filtered_acc)
        # Added Coriolis term
        cor = torch.matmul(-self._Ca, velRel)
        
        # All additional (compared to standard rigid body) Fossen terms combined.
        tau = damping + added + cor

        #utils.Assert(not math.isnan(np.linalg.norm(tau)), "Hydrodynamic forces vector is nan")
        
        return tau


    def ComputeAcc(self, velRel, time, alpha):
        #Compute Fossen's nu-dot numerically. This is mandatory as Isaac does
        #not report accelerations

        if self._last_time < 0:
            self._last_time = time
            self._last_vel_rel = velRel
            return

        dt = time #time - self._last_time
        if dt <= 0.0:
            return

        acc = (velRel - self._last_vel_rel) / dt

        #   TODO  We only have access to the acceleration of the previous simulation
        #       step. The added mass will induce a strong force/torque counteracting
        #       it in the current simulation step. This can lead to an oscillating
        #       system.
        #       The most accurate solution would probably be to first compute the
        #       latest acceleration without added mass and then use this to compute
        #       added mass effects. This is not how gazebo works, though.

        self._filtered_acc = (1.0 - alpha) * self._filtered_acc + alpha * acc
        self._last_time = time
        self._last_vel_rel = velRel.copy()


    def getRelativeLinearVel(linearVel, rotWR):

        robot_velocity = torch.matmul(rotWR, linearVel)
        return robot_velocity # m/s

    def getRelativeAngularVel(angularVel, rotWR):
        
        robot_velocity = torch.matmul(rotWR, angularVel)
        return robot_velocity # rad/s

    @staticmethod
    def CrossProductOperator(A):
        B = torch.zeros([3,3])
        B[0,1] = -A[2]
        B[1,0] = A[2]
        B[0,2] = A[1]
        B[2,0] = -A[1]
        B[2,1] = A[0]
        B[1,2] = -A[0]
        return B
    
    def GetAddedMass(self):
        return self.scaling_added_mass * (self.added_mass + self.offset_added_mass * torch.eye(6))
    

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

    


