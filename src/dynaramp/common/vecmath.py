from __future__ import annotations
import logging

import numpy as np
from scipy.spatial.transform import Rotation

from .types import VectorLike, Vector, Matrix


logger = logging.getLogger(__name__)

# Reused constants. Building these once keeps them out of the transfer-matrix hot path,
# where `np.identity` alone accounted for a measurable share of the runtime.
I3 = np.identity(3, dtype=np.float64)
I3.setflags(write=False)
Z3 = np.zeros((3, 3), dtype=np.float64)
Z3.setflags(write=False)


def skew_sym_mat(vec: VectorLike) -> Matrix:
    """
    Skew-symmetric matrix of a 3D vector.

    Defined such that for any vector ``w`` the cross product can be written as a
    matrix product: ``v x w = S(v) @ w``.

    Parameters
    ----------
    vec : VectorLike
        A 3-vector, of shape ``(3, )`` or ``(3, 1)``.

    Returns
    -------
    Matrix
        The 3x3 skew-symmetric matrix ``S(vec)``.

    Raises
    ------
    ValueError
        If ``vec`` is not of shape ``(3, )`` or ``(3, 1)``.
    """
    vec = np.asarray(vec, dtype=np.float64)
    # Verify shape
    match vec.shape:
        # "Flat" vector
        case (3,):
            a, b, c = vec[0], vec[1], vec[2]
        case (3, 1):
            a, b, c = vec[0, 0], vec[1, 0], vec[2, 0]
        case _:
            err_msg = f"Input vector must be of shape (3,) or (3,1). Got shape {vec.shape}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    return np.array([
        [0.0, -c, b],
        [c, 0.0, -a],
        [-b, a, 0.0],
    ], dtype=np.float64)


def moment_about(lever: VectorLike, force: VectorLike) -> Vector:
    """
    Moment produced by a force acting at the end of a lever arm: ``m = r x F``.

    Parameters
    ----------
    lever : VectorLike
        Position of the application point relative to the reference point, ``r``.
    force : VectorLike
        The applied force ``F``, in the same frame as ``lever``.

    Returns
    -------
    Vector
        The 3-component moment about the reference point.

    Notes
    -----
    Evaluated as ``skew(r) @ F`` rather than ``np.cross``. The two are algebraically
    identical, but keeping the matrix product preserves the exact floating-point
    operation order of the call sites this helper replaced.
    """
    return (skew_sym_mat(lever) @ np.asarray(force, dtype=np.float64)).astype(np.float64)


def parallel_axis(inertia: Matrix, mass: float, offset: VectorLike) -> Matrix:
    """
    Shift an inertia tensor to a new reference point (parallel-axis theorem).

    ``J = I + m (|r|^2 I3 - r.r^T)``, with ``r`` the vector from the new reference
    point to the point the tensor is currently expressed about. The expression is a
    quadratic form in ``r``, so the sign of ``offset`` does not matter.

    Parameters
    ----------
    inertia : Matrix
        3x3 inertia tensor about the original reference point (typically the center
        of mass), in the body frame.
    mass : float
        Mass of the body.
    offset : VectorLike
        Vector ``r`` between the two reference points, in the body frame.

    Returns
    -------
    Matrix
        The 3x3 inertia tensor about the new reference point.
    """
    r = np.asarray(offset, dtype=np.float64)
    shift = mass * (np.dot(r, r) * I3 - np.outer(r, r))
    return (np.asarray(inertia, dtype=np.float64) + shift).astype(np.float64)


def spatial_mass_matrix(mass: float, inertia: Matrix | None = None) -> Matrix:
    """
    The 6x6 spatial mass/inertia matrix ``blkdiag(m I3, J)``.

    Parameters
    ----------
    mass : float
        Mass ``m`` of the body (or mass per unit length, for a distributed matrix).
    inertia : Matrix | None
        3x3 rotational inertia ``J``. ``None`` gives a point mass, whose rotational
        block is zero.

    Returns
    -------
    Matrix
        The 6x6 spatial mass matrix.
    """
    m_mat = np.zeros((6, 6), dtype=np.float64)
    m_mat[0:3, 0:3] = mass * I3
    if inertia is not None:
        m_mat[3:6, 3:6] = np.asarray(inertia, dtype=np.float64)
    return m_mat


def lever_transform(lever: VectorLike, sign: float = 1.0) -> Matrix:
    """
    The 6x6 lever-arm transform ``[[I3, sign * skew(r)], [0, I3]]``.

    With ``sign = +1`` it is the static (force/moment) transform carrying a wrench
    from one point to another: a force at the far end contributes ``r x F`` to the
    moment. With ``sign = -1`` it is the kinematic (displacement/rotation) transform:
    a rotation at the reference point contributes ``-r x theta`` -- equivalently
    ``theta x r`` -- to the displacement at the far end. The two differ only by that
    sign, which is why they share this constructor.

    Parameters
    ----------
    lever : VectorLike
        Vector ``r`` from the reference point to the target point.
    sign : float
        ``+1`` for the static transform, ``-1`` for the kinematic one.

    Returns
    -------
    Matrix
        The 6x6 transform.
    """
    transform = np.zeros((6, 6), dtype=np.float64)
    transform[0:3, 0:3] = I3
    transform[3:6, 3:6] = I3
    transform[0:3, 3:6] = sign * skew_sym_mat(lever)
    return transform


def euler_zyx(gamma: float, psi: float, phi: float) -> Matrix:
    """
    Rotation matrix ``A_LB`` from the projectile body frame K_B to the launch frame K_L.

    Built from the intrinsic z-y-x Euler angles of Eq. (23): ``gamma`` (about z),
    ``psi`` (about y), ``phi`` (about x), applied in the order gamma -> psi -> phi.

    In the NED launch frame (+X forward, +Y right, +Z down) these are, physically, yaw
    (gamma, about +z/down), pitch (psi, about +y/right) and roll (phi, about +x) -- the
    standard aerospace 3-2-1 sequence. The paper's own frame instead labels gamma/psi as
    pitch/yaw; the matrix is identical regardless of the axis naming.

    Implemented with scipy's intrinsic 'ZYX' sequence, which is exactly
    ``A_LB = Rz(gamma) @ Ry(psi) @ Rx(phi)``. This convention is the one for which the
    Euler-rate-to-angular-velocity map reproduces H_R exactly (see `h_rotation_matrix`).

    Parameters
    ----------
    gamma : float
        First Euler angle (rad), rotation about z (yaw in NED).
    psi : float
        Second Euler angle (rad), rotation about y (pitch in NED).
    phi : float
        Third Euler angle (rad), rotation about x (roll).

    Returns
    -------
    Matrix
        The 3x3 rotation matrix ``A_LB`` (K_B -> K_L).
    """
    return Rotation.from_euler("ZYX", [gamma, psi, phi]).as_matrix().astype(np.float64)


def small_rot(theta: VectorLike) -> Matrix:
    """
    First-order (small-angle) rotation matrix ``I + skew(theta)``.

    Used to approximate ``A_RP`` in Eq. (18): ``A_IP ~= A_IR (I + skew(theta_RP))``.

    Parameters
    ----------
    theta : VectorLike
        Small angular displacements ``[theta_x, theta_y, theta_z]``.

    Returns
    -------
    Matrix
        The 3x3 matrix ``I3 + skew(theta)``.
    """
    return (I3 + skew_sym_mat(theta)).astype(np.float64)


def h_rotation_matrix(gamma: float, psi: float) -> Matrix:
    """
    The 3x3 block ``H_R`` of Eq. (27).

    Maps the Euler rates ``[gamma_dot, psi_dot, phi_dot]`` to the angular velocity of
    K_B relative to K_L, expressed in K_L:
    ``L_omega_LB = H_R @ [gamma_dot, psi_dot, phi_dot]^T``.

    Parameters
    ----------
    gamma : float
        Yaw angle (rad).
    psi : float
        Pitch angle (rad).

    Returns
    -------
    Matrix
        The 3x3 matrix ``H_R``.
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
    Time derivative ``H_R_dot`` of Eq. (28).

    ``H_R`` depends only on gamma and psi, so phi_dot does not appear.

    Parameters
    ----------
    gamma : float
        Yaw angle (rad).
    psi : float
        Pitch angle (rad).
    gamma_dot : float
        Yaw rate (rad/s).
    psi_dot : float
        Pitch rate (rad/s).

    Returns
    -------
    Matrix
        The 3x3 matrix ``H_R_dot``.
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
    The 6x6 kinematic map ``H`` of Eq. (26), ``H = blkdiag(I3, H_R)``.

    Relates the projectile generalized velocity ``y = [s_B_dot; L_omega_LB]`` to the
    configuration rate ``x_dot`` via ``y = H @ x_dot``.

    Parameters
    ----------
    gamma : float
        Yaw angle (rad).
    psi : float
        Pitch angle (rad).

    Returns
    -------
    Matrix
        The 6x6 matrix ``H``.
    """
    h = np.zeros((6, 6), dtype=np.float64)
    h[0:3, 0:3] = I3
    h[3:6, 3:6] = h_rotation_matrix(gamma, psi)
    return h


def h_dot_matrix(gamma: float, psi: float, gamma_dot: float, psi_dot: float) -> Matrix:
    """
    The 6x6 time derivative ``H_dot`` of Eq. (26), ``H_dot = blkdiag(0, H_R_dot)``.

    Used in ``y_dot = H @ x_ddot + H_dot @ x_dot``.

    Parameters
    ----------
    gamma : float
        Yaw angle (rad).
    psi : float
        Pitch angle (rad).
    gamma_dot : float
        Yaw rate (rad/s).
    psi_dot : float
        Pitch rate (rad/s).

    Returns
    -------
    Matrix
        The 6x6 matrix ``H_dot``.
    """
    h_dot = np.zeros((6, 6), dtype=np.float64)
    h_dot[3:6, 3:6] = h_rotation_matrix_dot(gamma, psi, gamma_dot, psi_dot)
    return h_dot


def block_rotation(dcm: Matrix) -> Matrix:
    """
    The 12x12 coordinate-transformation matrix ``H`` of the MSTMM.

    Rui, *Transfer Matrix Method for Multibody Systems*, Eq. 15.103. It re-expresses a
    spatial state vector ``Z = [r; theta; m; q]`` under a change of coordinate frame:
    block-diagonal, applying the same 3x3 direction-cosine matrix to each of the four
    3-vector sub-blocks (translation, rotation, moment, force), ``H = blkdiag(D, D, D, D)``.

    Parameters
    ----------
    dcm : Matrix
        The 3x3 direction-cosine matrix ``D`` (e.g. a scipy Rotation's ``as_matrix()``).

    Returns
    -------
    Matrix
        The 12x12 block-diagonal coordinate-transformation matrix.
    """
    d = np.asarray(dcm, dtype=np.float64)
    h = np.zeros((12, 12), dtype=np.float64)
    for i in range(4):
        h[3 * i:3 * i + 3, 3 * i:3 * i + 3] = d
    return h
