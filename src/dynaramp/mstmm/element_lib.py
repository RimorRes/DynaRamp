from __future__ import annotations
import logging

from typing import Tuple

import numpy as np

from .structs import Element
from common.vecmath import skew_sym_mat
from common.types import EntityID, Vector, VectorLike, Matrix

logger = logging.getLogger(__name__)


class RigidBody(Element):

    def __init__(
            self,
            e_id: EntityID,
            mass: float,
            inertia: Matrix,
            com: VectorLike,
    ):
        super().__init__(e_id)

        self.mass = mass
        self.inertia = inertia
        self.com_pos = np.array(com)
        r = - self.com_pos
        self.j = inertia + mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))

    def _u(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        # The vector FROM the input TO the output
        r_io = output_pos - input_pos

        # The vector FROM the input TO the center of mass
        r_ic = self.com_pos - input_pos

        l_io = skew_sym_mat(r_io)
        l_ic = skew_sym_mat(r_ic)
        l_co = l_io - l_ic  # Mathematically equivalent to skew(r_io - r_ic)

        u_mat = np.block([
            [np.identity(3), -l_io, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.identity(3), np.zeros((3, 3)), np.zeros((3, 3))],
            [self.mass * (omega**2) * l_co, -(omega**2) * (self.mass * l_io @ l_ic + self.j), np.identity(3), l_io],
            [self.mass * (omega**2) * np.identity(3), -self.mass * (omega**2) * l_ic, np.zeros((3, 3)), np.identity(3)],
        ])
        return u_mat


class EulerBernoulliBeam(Element):
    # TODO: Add support for non-uniform beams (e.g. tapered, variable cross-section, etc.)
    def __init__(
            self,
            e_id: EntityID,
            length: float,
            density: float,
            youngs_mod: float,
            shear_mod: float,
            area: float,
            i_y: float,
            i_z: float,
    ):
        super().__init__(e_id)

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
    def _krylov_s(z: float) -> float:
        return (np.cosh(z) + np.cos(z)) / 2

    @staticmethod
    def _krylov_t(z: float) -> float:
        return (np.sinh(z) + np.sin(z)) / 2

    @staticmethod
    def _krylov_u(z: float) -> float:
        return (np.cosh(z) - np.cos(z)) / 2

    @staticmethod
    def _krylov_v(z: float) -> float:
        return (np.sinh(z) - np.sin(z)) / 2

    def _u(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        # TODO: Potentially broken logic here because of the fixed XYZ-LWH coordinate system
        x, y, z = output_pos - input_pos

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

        u_mat[1, 1] = u_mat[5, 5] = u_mat[8, 8] = u_mat[10, 10] = self._krylov_s(lam_y * x)
        u_mat[2, 2] = u_mat[4, 4] = u_mat[7, 7] = u_mat[11, 11] = self._krylov_s(lam_z * x)

        u_mat[1, 5] = u_mat[8, 10] = self._krylov_t(lam_y * x) / lam_y
        u_mat[1, 8] = u_mat[5, 10] = self._krylov_u(lam_y * x) / (self.e * self.iz * lam_y ** 2)
        u_mat[1, 10] = self._krylov_v(lam_y * x) / (self.e * self.iz * lam_y ** 3)
        u_mat[4, 7] = self._krylov_t(lam_z * x) / (self.e * self.iy * lam_z)

        u_mat[2, 4] = u_mat[7, 11] = - self._krylov_t(lam_z * x) / lam_z
        u_mat[2, 7] = u_mat[4, 11] = - self._krylov_u(lam_z * x) / (self.e * self.iy * lam_z ** 2)
        u_mat[2, 11] = self._krylov_v(lam_z * x) / (self.e * self.iy * lam_z ** 3)
        u_mat[5, 8] = self._krylov_t(lam_y * x) / (self.e * self.iz * lam_y)

        u_mat[4, 2] = u_mat[11, 7] = - lam_z * self._krylov_v(lam_z * x)
        u_mat[5, 1] = u_mat[10, 8] = lam_y * self._krylov_v(lam_y * x)
        u_mat[7, 2] = u_mat[11, 4] = - self.e * self.iy * lam_z**2 * self._krylov_u(lam_z * x)

        u_mat[7, 4] = self.e * self.iy * lam_z * self._krylov_v(lam_z * x)
        u_mat[8, 1] = u_mat[10, 5] = self.e * self.iz * lam_y**2 * self._krylov_u(lam_y * x)

        u_mat[8, 5] = self.e * self.iz * lam_y * self._krylov_v(lam_y * x)
        u_mat[10, 1] = self.e * self.iz * lam_y**3 * self._krylov_t(lam_y * x)
        u_mat[11, 2] = self.e * self.iy * lam_z**3 * self._krylov_t(lam_z * x)

        return u_mat


class SpatialElasticHinge(Element):

    def __init__(
            self,
            e_id: EntityID,
            k: Tuple[float, float, float],
            k_rot: Tuple[float, float, float],
    ):
        """
        :param k: Linear spring stiffnesses
        :param k_rot: Rotary spring torsional stiffnesses
        """
        super().__init__(e_id)

        k_mat = np.diag(- 1 / np.array(k))
        k_rot_mat = np.diag(1 / np.array(k_rot))

        k_block = np.block([
            [np.zeros((3, 3)), k_mat],
            [k_rot_mat, np.zeros((3, 3))]
        ])
        self._u_mat = np.block([
            [np.identity(6), k_block],
            [np.zeros((6, 6)), np.identity(6)],
        ])

    def _u(self, _=None, __=None, ___=None) -> Matrix:
        return self._u_mat
