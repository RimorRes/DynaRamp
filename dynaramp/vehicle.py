import numpy as np
from core import Basis

class SolidMotor:
    pass

class Shoe:

    def __init__(self, parent: RigidRocket):
        self.deactivation_dist = 0

        # Mechanical properties
        self.e_modulus = 0
        self.surf = 0
        self.l0 = 0

    def strain_energy(self, length):
        return 1/2 * self.surf * self.e_modulus / self.l0 * (length-self.l0)**2


class RigidRocket:

    def __init__(self, basis: Basis, glow: float):
        self.mass = glow

    def add_shoe(self, pos: np.ndarray):
        pass