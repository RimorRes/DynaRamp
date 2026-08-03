from __future__ import annotations
import logging

import numpy as np
from scipy.spatial.transform import Rotation

from .types import VectorLike, Matrix


logger = logging.getLogger(__name__)


def skew_sym_mat(vec: VectorLike) -> Matrix:
    """
    Returns the skew symmetric matrix of a 3D vector.
    The skew symmetric matrix is defined such that for any vector v,
    the cross-product with another vector w can be expressed as a matrix multiplication: v x w = S(v) * w,
    where S(v) is the skew symmetric matrix of v.
    :param vec:
    :return:
    """

    vec = np.array(vec, dtype=np.float64)
    # Verify shape
    match vec.shape:
        # "Flat" vector
        case (3,):
            ssv = np.array([
                [0, -vec[2], vec[1]],
                [vec[2], 0, -vec[0]],
                [-vec[1], vec[0], 0]
            ])
        case (3, 1):
            ssv = np.array([
                [0, -vec[2, 0], vec[1, 0]],
                [vec[2, 0], 0, -vec[0, 0]],
                [-vec[1, 0], vec[0, 0], 0]
            ])
        case _:
            err_msg = f"Input vector must be of shape (3,) or (3,1). Got shape {vec.shape}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    return ssv.astype(np.float64)


def euler_zyx(gamma: float, psi: float, phi: float) -> Matrix:
    """
    Rotation matrix A_LB transforming coordinates from the missile body frame K_B
    to the launch frame K_L, built from the z-y-x Euler angles of Eq. (23):
    pitch gamma (about z), yaw psi (about y), roll phi (about x), applied intrinsically
    in the order pitch -> yaw -> roll.

    Implemented with scipy's intrinsic 'ZYX' sequence, which is exactly
    A_LB = Rz(gamma) @ Ry(psi) @ Rx(phi). This convention is the one for which the
    Euler-rate-to-angular-velocity map reproduces H_R exactly (see `h_rotation_matrix`).

    :param gamma: Pitch angle (rad), rotation about z.
    :param psi: Yaw angle (rad), rotation about y.
    :param phi: Roll angle (rad), rotation about x.
    :return: 3x3 rotation matrix A_LB (K_B -> K_L).
    """
    return Rotation.from_euler("ZYX", [gamma, psi, phi]).as_matrix().astype(np.float64)


def small_rot(theta: VectorLike) -> Matrix:
    """
    First-order (small-angle) rotation matrix I + skew(theta), used to approximate
    A_RP in Eq. (18): A_IP ~= A_IR (I + skew(theta_RP)).

    :param theta: 3D column matrix of small angular displacements [theta_x, theta_y, theta_z].
    :return: 3x3 matrix I3 + skew(theta).
    """
    return (np.identity(3) + skew_sym_mat(theta)).astype(np.float64)


def h_rotation_matrix(gamma: float, psi: float) -> Matrix:
    """
    The 3x3 block H_R of Eq. (27) mapping the Euler rates [gamma_dot, psi_dot, phi_dot]
    to the angular velocity of K_B relative to K_L, expressed in K_L:
        L_omega_LB = H_R @ [gamma_dot, psi_dot, phi_dot]^T

    :param gamma: Pitch angle (rad).
    :param psi: Yaw angle (rad).
    :return: 3x3 matrix H_R.
    """
    cg, sg = np.cos(gamma), np.sin(gamma)
    cp, sp = np.cos(psi), np.sin(psi)
    return np.array([
        [0, -sg, cg * cp],
        [0, cg, sg * cp],
        [1, 0, -sp],
    ], dtype=np.float64)


def h_rotation_matrix_dot(gamma: float, psi: float, gamma_dot: float, psi_dot: float) -> Matrix:
    """
    Time derivative H_R_dot of Eq. (28). Note H_R depends only on gamma and psi,
    so phi_dot does not appear.

    :param gamma: Pitch angle (rad).
    :param psi: Yaw angle (rad).
    :param gamma_dot: Pitch rate (rad/s).
    :param psi_dot: Yaw rate (rad/s).
    :return: 3x3 matrix H_R_dot.
    """
    cg, sg = np.cos(gamma), np.sin(gamma)
    cp, sp = np.cos(psi), np.sin(psi)
    return np.array([
        [0, -gamma_dot * cg, -gamma_dot * sg * cp - psi_dot * cg * sp],
        [0, -gamma_dot * sg, gamma_dot * cg * cp - psi_dot * sg * sp],
        [0, 0, -psi_dot * cp],
    ], dtype=np.float64)


def h_matrix(gamma: float, psi: float) -> Matrix:
    """
    The 6x6 kinematic map H of Eq. (26) relating the missile generalized velocity
    y = [s_B_dot; L_omega_LB] to the configuration rate x_dot via y = H @ x_dot.
    H = blkdiag(I3, H_R).

    :param gamma: Pitch angle (rad).
    :param psi: Yaw angle (rad).
    :return: 6x6 matrix H.
    """
    h = np.zeros((6, 6), dtype=np.float64)
    h[0:3, 0:3] = np.identity(3)
    h[3:6, 3:6] = h_rotation_matrix(gamma, psi)
    return h


def h_dot_matrix(gamma: float, psi: float, gamma_dot: float, psi_dot: float) -> Matrix:
    """
    The 6x6 time derivative H_dot of Eq. (26): H_dot = blkdiag(0, H_R_dot),
    used in y_dot = H @ x_ddot + H_dot @ x_dot.

    :param gamma: Pitch angle (rad).
    :param psi: Yaw angle (rad).
    :param gamma_dot: Pitch rate (rad/s).
    :param psi_dot: Yaw rate (rad/s).
    :return: 6x6 matrix H_dot.
    """
    h_dot = np.zeros((6, 6), dtype=np.float64)
    h_dot[3:6, 3:6] = h_rotation_matrix_dot(gamma, psi, gamma_dot, psi_dot)
    return h_dot
