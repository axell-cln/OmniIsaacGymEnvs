import torch

class BuoyantObject:
    def __init__(self, num_envs):
            self._num_envs = num_envs
    
            return
        
        
    def compute_archimedes(self, density_water, submerged_volume, gravity):

        """This function apply the archimedes force to the center of the boat"""
        archimedes=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        archimedes[:,2] = density_water * gravity * submerged_volume
        return archimedes
    
        
    def compute_thrusters_constant_force(self):
        
        thrusters=torch.zeros((self._num_envs, 6), dtype=torch.float32)
        
        #turn
        thrusters[:,0]=400
        thrusters[:,3]=-400

        return thrusters
    
  
    def command_to_thrusters_force(self, left_thruster_command, right_thruster_command):
         
        """This function implement the non-linearity of the thrusters according to a command"""

        if left_thruster_command<-1 or left_thruster_command>1:
            print("error left command")
            return 
        if right_thruster_command<-1 or right_thruster_command>1:
            print("error right command")
            return 
        
        T_left=0
        T_right=0
        coeff_neg_commands=[88.61013986, 163.99545455, 76.81641608, 11.9476958, 0.20374615]
        coeff_pos_commands=[-197.800699, 334.050699, -97.6197902, 7.59341259, -0.0301846154]
        n=len(coeff_neg_commands)-1

        if left_thruster_command<0:
            for i in range(n):
                T_left+=(left_thruster_command**(n-i))*coeff_neg_commands[i]
            T_left+=coeff_neg_commands[n]
        elif left_thruster_command>=0:
            for i in range(n):
                T_left+=(left_thruster_command**(n-i))*coeff_pos_commands[i]
            T_left+=coeff_pos_commands[n]
        
        if right_thruster_command<0:
            for i in range(n):
                T_right+=(right_thruster_command**(n-i))*coeff_neg_commands[i]
            T_right+=coeff_neg_commands[n]
        elif right_thruster_command>=0:
            for i in range(n):
                T_right+=(right_thruster_command**(n-i))*coeff_pos_commands[i]
            T_right+=coeff_pos_commands[n]

        thrusters=torch.zeros((self._num_envs, 6), dtype=torch.float32)
        thrusters[:,0]=T_left
        thrusters[:,3]=T_right

        return thrusters
        

    def compute_drag(self, boat_velocities):

        eps= 0.00000001
        
        x_velocity=boat_velocities[:,0]
        y_velocity=boat_velocities[:,1]
        z_velocity=boat_velocities[:,2]

        x_rotation = boat_velocities[:,3]
        y_rotation = boat_velocities[:,4]
        z_rotation = boat_velocities[:,5]


        coeff_drag_x = 10.0
        coeff_drag_y = 10.0
        coeff_drag_z = 0.002

        drag=torch.zeros((self._num_envs, 6), dtype=torch.float32)
        
        drag[:,0]=-coeff_drag_x*abs(x_velocity)*x_velocity
        drag[:,1]=-coeff_drag_y*abs(y_velocity)*y_velocity
        drag[:,2]=-coeff_drag_z*abs(z_velocity)*z_velocity 

        drag[:,3]=-coeff_drag_x*abs(x_rotation)*x_rotation
        drag[:,4]=-coeff_drag_y*abs(y_rotation)*y_rotation
        drag[:,5]=-coeff_drag_z*abs(z_rotation)*z_rotation 

        return drag
    
    def stabilize_boat(self,yaws):
         
        K=2.5
        drag=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        
        drag[:,0] = - K * yaws[:,0]
        drag[:,1] = - K * yaws[:,1]

        return drag

    


