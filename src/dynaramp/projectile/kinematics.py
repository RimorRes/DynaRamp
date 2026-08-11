from __future__ import annotations
import logging

from dataclasses import dataclass

import numpy as np

from ..common.types import Vector, VectorLike, Matrix
from ..common.vecmath import skew_sym_mat, euler_zyx, small_rot
from .modal_field import GuideModalField, ModalShape
from .state import ProjectileState

logger = logging.getLogger(__name__)

# Coordinate convention (NED): +X forward (axial), +Y right (lateral), +Z down (vertical).
# The axial slide is x_R; the lateral/vertical O1 offsets are (y_L, z_L); the body attitude
# uses the intrinsic z-y-x angles (gamma=yaw, psi=pitch, phi=roll) via `euler_zyx`.


@dataclass(frozen=True)
class ProjectileKinematicState:
    """
    Result of the section 3.1-3.2 kinematic analysis at one instant, all projected in the
    inertial frame K_I. The influence blocks map the launch-vehicle modal rates (p_dot,
    p_ddot) and the projectile quasi-velocity (y, y_dot) to the absolute velocity/acceleration
    of O1 and the absolute angular velocity/acceleration of the projectile body K_B:

        I_r_dot_O1  = L_TO1 p_dot  + J_TO1 y
        I_r_ddot_O1 = L_TO1 p_ddot + J_TO1 y_dot + zeta_TO1
        I_omega_IB  = L_RP  p_dot  + J_RO1 y
        I_omega_dot_IB = L_RP p_ddot + J_RO1 y_dot + zeta_RO1
    """
    a_il: Matrix         # 3x3  A_IL (== A_IP)
    a_ib: Matrix         # 3x3  A_IB
    omega_il: Vector     # 3    I_omega_IL
    omega_ib: Vector     # 3    I_omega_IB
    r_o1: Vector         # 3    I_r_O1     absolute position of O1
    r_dot_o1: Vector     # 3    I_r_dot_O1 absolute velocity of O1
    l_to1: Matrix        # 3xn  translational, modal
    j_to1: Matrix        # 3x6  translational, quasi-velocity
    zeta_to1: Vector     # 3    translational convective term
    l_rp: Matrix         # 3xn  rotational, modal (== Phi_theta, i.e. L_RP)
    j_ro1: Matrix        # 3x6  rotational, quasi-velocity
    zeta_ro1: Vector     # 3    rotational convective term


class ProjectileKinematics:
    """
    Kinematics of the projectile within the bending launch rail (sections 3.1-3.2).

    Depends only on the rail modal field (the section 2 -> 3 seam) and the projectile's
    generalized state; it needs no inertial properties, which enter only in the equations
    of motion (section 3.3).
    """

    def __init__(self, field: GuideModalField):
        self.field: GuideModalField = field

    def evaluate(self, state: ProjectileState, p: VectorLike, p_dot: VectorLike) -> ProjectileKinematicState:
        """
        Assemble the influence blocks at the current state and launch-vehicle modal
        coordinates p and rates p_dot.
        """
        shape = self.field.evaluate(state.x_r)
        p = np.asarray(p, dtype=np.float64)
        p_dot = np.asarray(p_dot, dtype=np.float64)
        return self._assemble(shape, self.field.a_ir, state, p, p_dot)

    @staticmethod
    def _assemble(
            shape: ModalShape,
            a_ir: Matrix,
            state: ProjectileState,
            p: Vector,
            p_dot: Vector,
    ) -> ProjectileKinematicState:
        y = state.y
        v = state.v_p_prime               # v_P' = x_R_dot
        e_x = a_ir[:, 0]                   # a_ir @ [1, 0, 0]: axial unit vector in K_I

        # Selector row [1 0 0 0 0 0] picking the axial slide rate out of y (Eqs. 31, 33).
        sel_axial = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # --- Launch cross-section / point P' (Eqs. 15-21, 30-33) ---
        theta_rp = a_ir.T @ (shape.phi_theta @ p)         # R_theta_RP (Eq. 19)
        a_il = a_ir @ small_rot(theta_rp)                 # A_IP = A_IL (Eqs. 18, 29)

        ir1 = e_x + shape.phi_r_d1 @ p                    # I_r'_P'   (Eq. 31)
        l_tp = shape.phi_r                                # L_TP      (Eq. 31)
        l_rp = shape.phi_theta                            # L_RP      (Eq. 33)
        j_tpp = np.outer(ir1, sel_axial)                  # J_TP'     (Eq. 31)
        itheta1 = shape.phi_theta_d1 @ p                  # I_theta'_P'
        j_rpp = np.outer(itheta1, sel_axial)              # J_RP'     (Eq. 33)
        zeta_tpp = 2 * v * (shape.phi_r_d1 @ p_dot) + v ** 2 * (shape.phi_r_d2 @ p)      # zeta_TP' (Eq. 31)
        zeta_rpp = 2 * v * (shape.phi_theta_d1 @ p_dot) + v ** 2 * (shape.phi_theta_d2 @ p)  # zeta_RP' (Eq. 33)
        omega_il = l_rp @ p_dot + j_rpp @ y               # I_omega_IL (Eq. 32)

        # --- Point O1 and body frame K_B (Eqs. 34-40) ---
        a_lb = euler_zyx(state.gamma, state.psi, state.phi)
        a_ib = a_il @ a_lb                                # A_IB (Eq. 34)

        l_rpo1 = np.array([0.0, state.y_l, state.z_l])    # L_r_P'O1
        l_rpo1_dot = np.array([0.0, y[1], y[2]])          # L_r_dot_P'O1 = [0, y_L_dot, z_L_dot]
        lr_skew = skew_sym_mat(l_rpo1)

        # Selector [0 | e_y | e_z | 0 | 0 | 0] picking (y_L_dot, z_L_dot) out of y (Eq. 37).
        sel_lateral = np.zeros((3, 6))
        sel_lateral[1, 1] = 1.0
        sel_lateral[2, 2] = 1.0
        # Selector [O3 | I3] picking L_omega_LB out of y (Eq. 40).
        sel_omega = np.zeros((3, 6))
        sel_omega[0, 3] = 1.0
        sel_omega[1, 4] = 1.0
        sel_omega[2, 5] = 1.0

        w_il_skew = skew_sym_mat(omega_il)

        l_to1 = l_tp - a_il @ lr_skew @ l_rp                                        # Eq. 36
        j_to1 = j_tpp + a_il @ sel_lateral - a_il @ lr_skew @ j_rpp                 # Eq. 37
        zeta_to1 = (
            zeta_tpp
            - a_il @ lr_skew @ zeta_rpp
            + 2 * w_il_skew @ a_il @ l_rpo1_dot
            + w_il_skew @ w_il_skew @ a_il @ l_rpo1
        )                                                                          # Eq. 38

        j_ro1 = j_rpp + a_il @ sel_omega                                            # Eq. 40
        zeta_ro1 = zeta_rpp + w_il_skew @ a_il @ y[3:6]                             # Eq. 40
        omega_ib = l_rp @ p_dot + j_ro1 @ y                                         # Eq. 39

        # Absolute position of O1: rigid axial term + rail deformation + lateral offset.
        r_o1 = state.x_r * e_x + l_tp @ p + a_il @ l_rpo1
        # Absolute velocity of O1 (Eq. 35): I_r_dot_O1 = L_TO1 p_dot + J_TO1 y.
        r_dot_o1 = l_to1 @ p_dot + j_to1 @ y

        return ProjectileKinematicState(
            a_il=a_il.astype(np.float64),
            a_ib=a_ib.astype(np.float64),
            omega_il=omega_il.astype(np.float64),
            omega_ib=omega_ib.astype(np.float64),
            r_o1=r_o1.astype(np.float64),
            r_dot_o1=r_dot_o1.astype(np.float64),
            l_to1=l_to1.astype(np.float64),
            j_to1=j_to1.astype(np.float64),
            zeta_to1=zeta_to1.astype(np.float64),
            l_rp=l_rp.astype(np.float64),
            j_ro1=j_ro1.astype(np.float64),
            zeta_ro1=zeta_ro1.astype(np.float64),
        )
