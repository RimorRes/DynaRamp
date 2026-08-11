from __future__ import annotations
import logging

from dataclasses import dataclass

import numpy as np

from ..common.types import Vector, VectorLike
from ..common.vecmath import h_matrix, h_rotation_matrix

logger = logging.getLogger(__name__)


@dataclass
class ProjectileState:
    """
    Generalized state of the projectile relative to the launch rail (Eqs. 22-24).

    Configuration
        x = [x_R, y_L, z_L, gamma, psi, phi]^T = [s_B; a_B]
        with s_B the O1 position coordinates in K_L (axial x_R, lateral/right y_L, vertical/
        down z_L) and a_B = [gamma, psi, phi] the intrinsic z-y-x Euler angles. In the NED
        launch frame (+X forward, +Y right, +Z down) these are, physically, gamma = yaw
        (about +z/down), psi = pitch (about +y/right) and phi = roll (about +x/forward). The
        paper's own frame labels gamma/psi as pitch/yaw instead; the z-y-x math is identical.
    Quasi-velocity
        y = [x_R_dot, y_L_dot, z_L_dot, w1, w2, w3]^T = [s_B_dot; L_omega_LB]
        with L_omega_LB the angular velocity of K_B relative to K_L, expressed in K_L.

    The two are linked by y = H(gamma, psi) @ x_dot (Eqs. 25-27).
    """
    x: Vector
    y: Vector

    # --- configuration accessors ---
    @property
    def s_b(self) -> Vector:
        return self.x[0:3]

    @property
    def a_b(self) -> Vector:
        return self.x[3:6]

    @property
    def x_r(self) -> float:
        return float(self.x[0])

    @property
    def y_l(self) -> float:
        return float(self.x[1])

    @property
    def z_l(self) -> float:
        return float(self.x[2])

    @property
    def gamma(self) -> float:
        return float(self.x[3])

    @property
    def psi(self) -> float:
        return float(self.x[4])

    @property
    def phi(self) -> float:
        return float(self.x[5])

    # --- velocity accessors ---
    @property
    def s_b_dot(self) -> Vector:
        return self.y[0:3]

    @property
    def omega_lb(self) -> Vector:
        """L_omega_LB: angular velocity of K_B relative to K_L, expressed in K_L."""
        return self.y[3:6]

    @property
    def v_p_prime(self) -> float:
        """Relative axial slide speed v_P' = x_R_dot (Eq. A2)."""
        return float(self.y[0])

    # --- x <-> y conversions via H ---
    @classmethod
    def from_config_rates(cls, x: VectorLike, x_dot: VectorLike) -> "ProjectileState":
        """
        Build a state from configuration and configuration rates, via ``y = H x_dot``.

        Parameters
        ----------
        x : VectorLike
            Configuration ``[x_R, y_L, z_L, gamma, psi, phi]``.
        x_dot : VectorLike
            Configuration rates.

        Returns
        -------
        ProjectileState
            The state, with its quasi-velocity derived from the rates.
        """
        x = np.array(x, dtype=np.float64).reshape(6)
        x_dot = np.array(x_dot, dtype=np.float64).reshape(6)
        y = h_matrix(x[3], x[4]) @ x_dot
        return cls(x, y)

    def config_rates(self) -> Vector:
        """
        Recover the configuration rates from the quasi-velocity by inverting ``H``.

        ``x_dot = [s_B_dot; H_R^{-1} L_omega_LB]``.

        Returns
        -------
        Vector
            The 6 configuration rates.

        Notes
        -----
        Singular at ``psi = +/- pi/2`` (gimbal lock), where ``H_R`` loses rank.
        """
        x_dot = np.empty(6, dtype=np.float64)
        x_dot[0:3] = self.y[0:3]
        x_dot[3:6] = np.linalg.solve(h_rotation_matrix(self.gamma, self.psi), self.y[3:6])
        return x_dot
