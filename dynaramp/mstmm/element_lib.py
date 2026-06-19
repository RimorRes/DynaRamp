from __future__ import annotations
import logging

from typing import Dict, Sequence

import numpy as np

from .structs import Element
from ..vecmath import skew_sym_mat
from ..common_types import EntityID, Vector, Matrix

logger = logging.getLogger(__name__)


class RigidBody(Element):

    def __init__(
            self,
            e_id,
            mass: float,
            inertia: Matrix,
            com: Vector,
            slot_coords: Dict[EntityID, Vector]
    ):
        slots_pos = {s_id: np.array(pos) for s_id, pos in slot_coords.items()}
        super().__init__(e_id, slots_pos)

        self.mass = mass
        self.inertia = inertia
        self.com_pos = np.array(com)
        r = - self.com_pos
        self.j = inertia + mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))

    def u(self, omega: float, output_pos: Vector) -> Matrix:
        l_io = skew_sym_mat(output_pos)
        l_ic = skew_sym_mat(self.com_pos)
        l_co = l_io - l_ic

        u_mat = np.block([
            [np.identity(3), -l_io, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.identity(3), np.zeros((3, 3)), np.zeros((3, 3))],
            [self.mass * (omega**2) * l_co, -(omega**2) * (self.mass * l_io @ l_ic + self.j), np.identity(3), l_io],
            [self.mass * (omega**2) * np.identity(3), -self.mass * (omega**2) * l_ic, np.zeros((3, 3)), np.identity(3)],
        ])
        return u_mat


class EulerBernoulliBeam(Element):

    def __init__(
            self,
            length: float,
            density: float,
            youngs_mod: float,
            shear_mod: float,
            area: float,
            i_y: Matrix,
            i_z: Matrix
    ):

        self.length = length
        self.rho = density
        self.e = youngs_mod
        self.g = shear_mod
        self.a = area  # Cross-sectional area
        self.iy = i_y  # Second moment of area about y-axis
        self.iz = i_z  # Second moment of area about z-axis
        self.jp = self.iy + self.iz  # Polar moment of inertia, application of the perpendicular axis theorem
        self.mu = self.rho * self.a  # Linear mass density

    @staticmethod
    def _s(z: float) -> float:
        return (np.cosh(z) + np.cos(z)) / 2

    @staticmethod
    def _t(z: float) -> float:
        return (np.sinh(z) + np.sin(z)) / 2

    @staticmethod
    def _u(z: float) -> float:
        return (np.cosh(z) - np.cos(z)) / 2

    @staticmethod
    def _v(z: float) -> float:
        return (np.sinh(z) - np.sin(z)) / 2

    def u(self, omega: float, output_pos: Vector) -> Matrix:
        x, y, z = output_pos

        beta_x = np.sqrt(self.mu * omega**2 / (self.e * self.a))
        lam_y = np.power((self.mu * omega**2 / (self.e * self.iz)), 1/4)
        lam_z = np.power((self.mu * omega**2 / (self.e * self.iy)), 1/4)
        gam_theta_x = np.sqrt(self.rho * omega**2 / self.g)

        u_mat = np.zeros((12, 12))
        u_mat[0, 0] = u_mat[9, 9] = np.cos(beta_x * x)
        u_mat[0, 9] = - np.sin(beta_x * x) / (beta_x * self.e * self.a)
        u_mat[9, 0] = beta_x * self.e * self.a * np.sin(beta_x * x)

        u_mat[3, 3] = u_mat[6, 6] = np.cos(gam_theta_x * x)
        u_mat[3, 6] = np.sin(gam_theta_x * x) / (gam_theta_x * self.g * self.jp)
        u_mat[6, 3] = - gam_theta_x * self.g * self.jp * np.sin(gam_theta_x * x)

        u_mat[1, 1] = u_mat[5, 5] = u_mat[8, 8] = u_mat[10, 10] = self._s(lam_y * x)
        u_mat[2, 2] = u_mat[4, 4] = u_mat[7, 7] = u_mat[11, 11] = self._s(lam_z * x)

        u_mat[1, 5] = u_mat[8, 10] = self._t(lam_y * x) / lam_y
        u_mat[1, 8] = u_mat[5, 10] = self._u(lam_y * x) / (self.e * self.iz * lam_y**2)
        u_mat[1, 10] = self._v(lam_y * x) / (self.e * self.iz * lam_y**3)
        u_mat[4, 7] = self._t(lam_z * x) / (self.e * self.iy * lam_z)

        u_mat[2, 4] = u_mat[7, 11] = - self._t(lam_z * x) / lam_z
        u_mat[2, 7] = u_mat[4, 11] = - self._u(lam_z * x) / (self.e * self.iy * lam_z**2)
        u_mat[2, 11] = self._v(lam_z * x) / (self.e * self.iy * lam_z**3)
        u_mat[5, 8] = self._t(lam_y * x) / (self.e * self.iz * lam_y)

        u_mat[4, 2] = u_mat[11, 7] = - lam_z * self._v(lam_z * x)
        u_mat[5, 1] = u_mat[10, 8] = lam_y * self._v(lam_y * x)
        u_mat[7, 2] = u_mat[11, 4] = - self.e * self.iy * lam_z**2 * self._u(lam_z * x)

        u_mat[7, 4] = self.e * self.iy * lam_z * self._v(lam_z * x)
        u_mat[8, 1] = u_mat[10, 5] = self.e * self.iz * lam_y**2 * self._u(lam_y * x)

        u_mat[8, 5] = self.e * self.iz * lam_y * self._v(lam_y * x)
        u_mat[10, 1] = self.e * self.iz * lam_y**3 * self._t(lam_y * x)
        u_mat[11, 2] = self.e * self.iy * lam_z**3 * self._t(lam_z * x)

        return u_mat


class SpatialElasticHinge(Element):

    def __init__(self, k: Sequence[np.float64], k_rot: Sequence[np.float64]):
        """

        :param k: Linear spring stiffnesses
        :param k_rot: Rotary spring torsional stiffnesses
        """
        k_mat = np.diag(- 1 / np.array(k))
        k_rot_mat = np.diag(1 / np.array(k_rot))

        k_block = np.block([
            [np.zeros((3, 3)), k_mat],
            [k_rot_mat, np.zeros((3, 3))]
        ])
        self._u = np.block([
            [np.identity(6), k_block],
            [np.zeros((6, 6)), np.identity(6)],
        ])

    def u(self, _=None, __=None) -> Matrix:
        return self._u
