import numpy as np
from core import Basis

class SolidMotor:
    pass

class Shoe:

    def __init__(self, parent: RigidRocket):
        self.deactivation_dist = 0
        self.dk = 0  # X-axis distance from center of mass
        self.rk = 0  # Y-axis radius from centerline

        # Mechanical properties
        self.e_modulus = 0
        self.surf = 0
        self.l0 = 0  # Relaxed length

        self.spring_const = self.e_modulus * self.surf / self.l0


class RigidRocket:

    def __init__(self, basis: Basis, glow: float):
        self.mass = glow

    def add_shoe(self, pos: np.ndarray):
        pass