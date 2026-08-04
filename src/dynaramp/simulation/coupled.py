from __future__ import annotations
import logging

from typing import Sequence, Tuple

import numpy as np

from ..common.types import Vector
from ..projectile.modal_field import GuideModalField
from ..projectile.dynamics import ProjectileEOM
from ..contact.solver import GuideReaction

logger = logging.getLogger(__name__)


def modal_contact_force(
        field: GuideModalField,
        modal_masses: Vector,
        guide_reactions: Sequence[GuideReaction],
) -> Vector:
    """
    Project the guide-side contact reactions onto the launch-vehicle modes to build the
    modal force f_g (Eqs. 11, 68).

    Each reaction is an equivalent wrench (force I_q_Pi,i, moment I_m_Pi,i) at the contact
    station Pi(x_R,i). Its generalized force on mode s is the virtual work through the mode
    shape at that station,

        Q_s = Phi_r(x_R,i)[:,s] . I_q_Pi,i + Phi_theta(x_R,i)[:,s] . I_m_Pi,i,

    and the modal force is f_g[s] = (sum_i Q_s) / M_s.

    Parameters
    ----------
    field : GuideModalField
        The guide modal field (mode shapes at any station).
    modal_masses : Vector
        The n modal masses M_s (from System.calc_system_modal_masses).
    guide_reactions : Sequence[GuideReaction]
        Per-slider guide reactions from the contact solver.

    Returns
    -------
    Vector
        The n-vector modal force f_g.
    """
    n = field.n_modes
    gen = np.zeros(n, dtype=np.float64)
    for r in guide_reactions:
        phi_r, phi_theta = field.phi(r.x_r_i)                 # each 3 x n
        gen += phi_r.T @ r.i_q_pi + phi_theta.T @ r.i_m_pi    # Q_s per mode

    modal_masses = np.asarray(modal_masses, dtype=np.float64)
    return (gen / modal_masses).astype(np.float64)


def modal_damping_stiffness(
        frequencies: Vector,
        rayleigh: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[Vector, Vector]:
    """
    Diagonal generalized damping and stiffness of the launch vehicle (Eq. 11).

    With Rayleigh damping C = alpha*M + beta*K, the modes are C-orthogonal, so both are
    diagonal: K_g = diag(omega_s^2), C_g = diag(alpha + beta*omega_s^2). Returned as the
    n-vectors of their diagonals.

    :param frequencies: The n natural frequencies omega_s [rad/s].
    :param rayleigh: (alpha, beta) Rayleigh coefficients.
    :return: (c_g, k_g), the diagonals of C_g and K_g.
    """
    omega = np.asarray(frequencies, dtype=np.float64)
    k_g = omega ** 2
    alpha, beta = rayleigh
    c_g = alpha + beta * k_g
    return c_g.astype(np.float64), k_g.astype(np.float64)


def assemble_and_solve(
        eom: ProjectileEOM,
        c_g: Vector,
        k_g: Vector,
        p: Vector,
        p_dot: Vector,
        f_g: Vector,
        contact_q: Vector,
        contact_m: Vector,
        ext_q: Vector,
        ext_m: Vector,
) -> Tuple[Vector, Vector]:
    """
    Assemble and solve the coupled system of motion (Eq. 67) for the generalized
    accelerations [p_ddot; y_dot]:

        [ I_n    0   ] [p_ddot]   [b_g]
        [ M_Tp  M_Ty ] [y_dot ] = [b_T]
        [ M_Rp  M_Ry ]            [b_R]

    with b_g = f_g - C_g p_dot - K_g p (Eq. 68), b_T = h_T + contact + external forces
    (Eq. 69), b_R = h_R + contact + external moments (Eq. 70). C_g, K_g are passed as their
    diagonals.

    Parameters
    ----------
    eom : ProjectileEOM
        The section 3 blocks (M_Tp, M_Ty, M_Rp, M_Ry, h_T, h_R).
    c_g, k_g : Vector
        Diagonals of the vehicle generalized damping and stiffness (see
        `modal_damping_stiffness`).
    p, p_dot : Vector
        Vehicle modal coordinates and rates.
    f_g : Vector
        Vehicle modal force (guide-side contact projection; see `modal_contact_force`).
    contact_q, contact_m : Vector
        Total contact force and moment on the projectile at O1 (section 4 sums).
    ext_q, ext_m : Vector
        Total non-contact force and moment on the projectile at O1 (external loads).

    Returns
    -------
    Tuple[Vector, Vector]
        (p_ddot, y_dot): the n vehicle modal accelerations and the 6 projectile
        quasi-accelerations.
    """
    n = eom.m_tp.shape[1]
    c_g = np.asarray(c_g, dtype=np.float64)
    k_g = np.asarray(k_g, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p_dot = np.asarray(p_dot, dtype=np.float64)

    a = np.zeros((n + 6, n + 6), dtype=np.float64)
    a[:n, :n] = np.identity(n)
    a[n:n + 3, :n] = eom.m_tp
    a[n:n + 3, n:n + 6] = eom.m_ty
    a[n + 3:n + 6, :n] = eom.m_rp
    a[n + 3:n + 6, n:n + 6] = eom.m_ry

    b = np.empty(n + 6, dtype=np.float64)
    b[:n] = f_g - c_g * p_dot - k_g * p                         # b_g (Eq. 68)
    b[n:n + 3] = eom.h_t + contact_q + ext_q                    # b_T (Eq. 69)
    b[n + 3:n + 6] = eom.h_r + contact_m + ext_m                # b_R (Eq. 70)

    sol = np.linalg.solve(a, b)
    return sol[:n].astype(np.float64), sol[n:].astype(np.float64)
