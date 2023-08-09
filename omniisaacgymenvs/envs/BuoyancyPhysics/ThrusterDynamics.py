import math
import torch

class Dynamics:
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.thrusters=torch.zeros((self.num_envs, 6), dtype=torch.float32)

        #interpolate
        self.commands = torch.linspace(-1, 1, steps=21)
        self.thrusters_forces = torch.tensor([-10.1043, -10.0062,  -8.5347,  -4.4145,  -3.3354,  -1.4715,  -0.9810,
         -0.2943,  -0.1962,   0.0000,   0.0000,  -0.0000,   0.2943,   0.5886,
          2.1582,  10.3986,  18.3447,  21.5820,  31.7844,  44.7336,  45.1260])
        
        #lsm
        self.coeff_neg_commands=[88.61013986, 163.99545455, 76.81641608, 11.9476958, 0.20374615]
        self.coeff_pos_commands=[-197.800699, 334.050699, -97.6197902, 7.59341259, -0.0301846154]

        self.cmd_updated = torch.zeros((self.num_envs, 2), dtype=torch.float32)
        self.Reset()

    def update(self, cmd, dt):
      raise NotImplementedError()

    def Reset(self):
        self.cmd_updated = 0.

class DynamicsZeroOrder(Dynamics):
    def __init__(self, num_envs):
        super().__init__(num_envs)
        return
    def update(self, cmd):
        return cmd

class DynamicsFirstOrder(Dynamics):
    def __init__(self, timeConstant, num_envs):

        super().__init__(num_envs)
        self.tau = timeConstant
        self.cmd_updated = 0.0
        self.interpolate_on_field_data()


    def update(self, cmd, dt):
    
        alpha = math.exp(-dt/self.tau)

        self.cmd_updated = self.cmd_updated*alpha + (1.0 - alpha)*cmd

        return self.cmd_updated
    
    def compute_thrusters_constant_force(self):
            """for testng purpose"""
    
            #turn
            self.thrusters[:,0]=400
            self.thrusters[:,3]=-400

            return self.thrusters
    
  
    def interpolate_on_field_data(self):

        self.x_linear_interp = torch.linspace(min(self.commands), max(self.commands), 1000)
        self.y_linear_interp = torch.nn.functional.interpolate(self.thrusters_forces.unsqueeze(0).unsqueeze(0), size=1000, mode='linear', align_corners=False)

        self.y_linear_interp.squeeze(0).squeeze(0)

        self.n = len(self.y_linear_interp) - 1
        
        #self.lookup_table = {round(x.item(),3): round(y.item(),3) for x, y in zip(self.x_linear_interp, self.y_linear_interp)}
        
    
    def get_cmd_interpolated(self, cmd_value):
        
        #cmd_value is size (num_envs,2)
        idx = torch.round((cmd_value + 1) * self.n/2).long()

        interpolated_values = self.y_linear_interp[idx[:, 0] * self.n + idx[:, 1]]

        print("interpolated_values: ", interpolated_values)

        return interpolated_values

    def command_to_thrusters_force_lookup_table(self, commands):

        #size (num_envs,2)
        thruster_forces = self.get_cmd_interpolated(commands)

        self.thrusters = self.update(thruster_forces, 0.01)
        
        return self.thrusters


    def command_to_thrusters_force_lsm(self, left_thruster_command, right_thruster_command):
         
        """This function implement the non-linearity of the thrusters according to a command""" 
        
        T_left=0
        T_right=0
    
        n=len(self.coeff_neg_commands)-1

        if left_thruster_command<0:
            for i in range(n):
                T_left+=(left_thruster_command**(n-i))*self.coeff_neg_commands[i]
            T_left+=self.coeff_neg_commands[n]
        elif left_thruster_command>=0:
            for i in range(n):
                T_left+=(left_thruster_command**(n-i))*self.coeff_pos_commands[i]
            T_left+=self.coeff_pos_commands[n]
        
        if right_thruster_command<0:
            for i in range(n):
                T_right+=(right_thruster_command**(n-i))*self.coeff_neg_commands[i]
            T_right+=self.coeff_neg_commands[n]
        elif right_thruster_command>=0:
            for i in range(n):
                T_right+=(right_thruster_command**(n-i))*self.coeff_pos_commands[i]
            T_right+=self.coeff_pos_commands[n]


        self.thrusters[:,0]= self.update(T_left, 0.01)
        self.thrusters[:,3]= self.update(T_right, 0.01)

        return self.thrusters

