import numpy as np

from geometry import Basis


class SimpleMotor:

    def __init__(self, thrust: float, isp: float):
        self.thrust = thrust  # Thrust in Newtons
        self.isp = isp        # Specific impulse in seconds
        self.m_dot = self.thrust / (self.isp * 9.81)  # Mass flow rate in kg/s


class RigidRocket2D:

    def __init__(self, parent_basis: Basis, motor: SimpleMotor, mass: float, inertia: float,
                 start_pos: np.ndarray = None):

        self.mass = mass   # Gross lift off weight, kg
        self.J = inertia   # Mass moment of inertia around Y-axis, kg*m^2

        self.parent_basis = parent_basis
        self.motor = motor
        self.shoes: list['Shoe'] = []

        # Reminder: NED frame — x along rail, y transverse (away from rail surface), z down
        self.pos = start_pos if start_pos is not None else np.zeros(3)
        self.theta = 0.0   # Rocket pitch angle relative to rail axis, radians

        if parent_basis is not None:
            self.rot_axis = parent_basis.uy

    def update(self, pos: np.ndarray, theta: float) -> None:
        self.pos = pos
        self.theta = theta

    def add_shoe(self, rel_pos: np.ndarray, release_point, friction_coef: float) -> None:
        self.shoes.append(Shoe(self, rel_pos, release_point, friction_coef))

    @property
    def thrust_vec(self) -> np.ndarray:
        if self.motor is None:
            return np.zeros(3)
        T = self.motor.thrust
        # Thrust acts along rocket body axis; theta is pitch relative to rail direction
        return np.array([T * np.cos(self.theta), T * np.sin(self.theta), 0.0])


class Shoe:

    def __init__(self, parent: RigidRocket2D, rel_pos: np.ndarray, release_point: float,
                 friction_coef: float, e_modulus: float = 210e10,
                 surf: float = 0.02, l0: float = 0.1):
        self.parent = parent

        self.dk, self.rk = rel_pos  # axial distance from CG (along x), lateral radius from centerline
        self.x_release = release_point  # Abscissa along rail where shoe releases contact

        self.e_modulus = e_modulus
        self.surf = surf
        self.l0 = l0  # Relaxed length

        self.spring_const = self.e_modulus * self.surf / self.l0
        self.f_coef = friction_coef

    @property
    def contact_loc(self) -> float:
        # Contact location along the rail x-axis, accounting for pitch
        return self.parent.pos[0] + self.dk - self.rk * self.parent.theta

    def force_norm(self, beam_displacement: float) -> float:
        # Spring normal force from Lagrangian: N = k * delta, delta = y + w(x_k) - dk*theta - rk - l0
        delta = (self.parent.pos[1] + beam_displacement
                 - self.dk * self.parent.theta - self.rk - self.l0)
        return self.spring_const * delta
