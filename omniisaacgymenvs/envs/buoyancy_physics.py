import torch

class BuoyantObject:
    def __init__(self):

        return
         
    def compute_archimedes_simple(self, mass, gravity):

        archimedes=torch.zeros(3)
        archimedes[2] = - gravity * mass
        return archimedes
        
