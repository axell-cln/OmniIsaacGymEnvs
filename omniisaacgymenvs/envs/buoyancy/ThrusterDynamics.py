"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

import math
import torch

class DynamicsZeroOrder:
    def __init__(self):
        return
    def update(self, cmd):
        return cmd


#add inheritance

class DynamicsFirstOrder:
    def __init__(self, timeConstant, num_envs):
        self.num_envs = num_envs
        self.tau = timeConstant
        self.cmd_updated = 0

    def update(self, cmd, dt):
    
        alpha = math.exp(-dt/self.tau)
        self.cmd_updated = self.cmd_updated*alpha + (1.0 - alpha)*cmd

        #print(self.cmd_updated)
        return self.cmd_updated
    
    def compute_thrusters_constant_force(self):
            """for testng purpose"""
            thrusters=torch.zeros((self._num_envs, 6), dtype=torch.float32)
            
            #turn
            thrusters[:,0]=400
            thrusters[:,3]=-400

            return thrusters
    
  
    def command_to_thrusters_force(self, left_thruster_command, right_thruster_command):
         
        """This function implement the non-linearity of the thrusters according to a command""" 
        
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

        thrusters=torch.zeros((self.num_envs, 6), dtype=torch.float32)

        T_left_real = self.update(T_left, 0.01)
        T_right_real = self.update(T_right, 0.01)

        thrusters[:,0]= T_left_real
        thrusters[:,3]= T_right_real

        return thrusters

