import numpy as np

from geometry import Basis


class SimpleMotor:

    def __init__(self, thrust: float, isp: float):
        self.thrust = thrust  # Thrust in Newtons
        self.isp = isp  # Specific impulse in seconds

        self.m_dot = self.thrust / (self.isp * 9.81)  # Mass flow rate in kg/s

class Shoe:

    def __init__(self, parent: RigidRocket2D, rel_pos: np.ndarray,
                 friction_coef: float, e_modulus: float=210e9, surf: float=0.02, l0: float=0.1):
        self.parent = parent

        self.deactivation_dist = 0
        self.dk, self.rk = rel_pos # X-axis distance from center of mass, Y-axis radius from centerline

        # Mechanical properties
        self.e_modulus = e_modulus
        self.surf = surf
        self.l0 = l0  # Relaxed length

        self.spring_const = self.e_modulus * self.surf / self.l0
        self.f_coef = friction_coef

    @property
    def contact_loc(self):
        # Contact location along the x-axis in rail coordinates
        return self.dk - self.rk * self.parent.theta  + self.parent.pos[0]

    def force_norm(self, beam_displacement: float):
        # Spring normal force
        disp = self.parent.pos[1] + beam_displacement - self.dk * self.parent.theta - self.rk - self.l0
        return self.spring_const * disp


class RigidRocket2D:

    def __init__(self, parent_basis: Basis, mass: float, inertia: float, start_pos: np.ndarray):

        self.mass = mass  # Gross lift off weight, kg
        self.J = inertia  # Mass moment of inertia around Y-axis, kg*m^2

        self.parent_basis = parent_basis

        # Reminder: NED frame
        self.pos = start_pos  # Position of the center of mass (in local rail coordinates)
        self.theta = 0  # Rocket pitch angle, radians
        self.rot_axis = parent_basis.uy  # Rocket pitch axis

