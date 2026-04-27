import numpy as np
from core import Basis


class SimpleMotor:

    def __init__(self, thrust: float, isp: float):
        self.thrust = thrust  # Thrust in Newtons
        self.isp = isp  # Specific impulse in seconds

        self.m_dot = self.thrust / (self.isp * 9.81)  # Mass flow rate in kg/s

class Shoe:

    def __init__(self, parent: RigidRocket2D, rel_pos: np.ndarray):
        self.parent = parent

        self.deactivation_dist = 0
        self.dk, self.rk = rel_pos # X-axis distance from center of mass, Y-axis radius from centerline

        # Mechanical properties
        self.e_modulus = 0
        self.surf = 0
        self.l0 = 0  # Relaxed length

        self.spring_const = self.e_modulus * self.surf / self.l0

    @property
    def contact_loc(self):
        # Contact location along the x-axis in rail coordinates
        return self.dk - self.rk * self.parent.theta  + self.parent.pos[0]

    def force(self, beam_displacement: float):
        # Spring normal force
        disp = self.parent.pos[1] + beam_displacement - self.dk * self.parent.theta - self.rk - self.l0
        return self.spring_const * disp * self.parent.basis.uz


class RigidRocket2D:

    def __init__(self, parent_basis: Basis, motor: SimpleMotor, glow: float, inertia: float):
        self.basis = parent_basis  # Rail coordinate system, FUR
        self.motor = motor

        self.mass = glow  # Gross lift off weight, kg
        self.J = inertia  # Mass moment of inertia around Y-axis, kg*m^2

        # Location and orientation, 3D but constrained to XZ plane
        self.theta = 0  # Angle around Z-axis
        self.pos = np.array([0, 0, 0])

        self.shoes = []

    def add_shoe(self, rel_pos: np.ndarray):
        # Add a shoe to the rocket, with the given relative (dk, rk) position in the rocket's local frame
        self.shoes.append(Shoe(self, rel_pos))

    def update(self, pos, theta, dt: float):
        self.pos = pos
        self.theta = theta
        # TODO: update mass based on fuel consumption
        # self.mass -= self.motor.m_dot * dt
