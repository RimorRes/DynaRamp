import numpy as np

from ..launchrail import LaunchRail
from ..vehicle import RigidRocket2D
from ..geometry import Basis


class PointOnRocket:

    def __init__(self, rocket:RigidRocket2D, loc_point:np.ndarray):
        self.rocket = rocket
        self.point = loc_point  # Static location of the force application in the rocket's local frame

    def point_vec(self, q: np.ndarray) -> np.ndarray:
        """

        :param q: generalized coordinates
        :return:
        """
        s, y, theta = q[:3]

        temp_basis = Basis(self.rocket.parent_basis)
        temp_basis.rotate(theta, self.rocket.rot_axis)

        rocket_pos = self.rocket.parent_basis.global_transform @ np.array([s, 0.0, y])

        # Position relative to rail base (origin) in global coordinates
        return temp_basis.global_transform @ self.point + rocket_pos

class PointOnRail:

    def __init__(self, rail: LaunchRail, abscissa: float):
        self.rail = rail
        self.point = abscissa
