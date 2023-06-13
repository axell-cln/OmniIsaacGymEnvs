import numpy as np
import math

class BuoyantObject:
    def __init__(self):

        self.gravity= -9.81
        self.mass = 5.0
        self.archimedes = - self.gravity * self.mass 

        
