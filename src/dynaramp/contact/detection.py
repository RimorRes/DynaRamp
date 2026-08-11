from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import brentq

from ..common.types import Vector, Matrix
from ..common.vecmath import skew_sym_mat, small_rot
from ..projectile.kinematics import ProjectileKinematicState
from ..projectile.modal_field import GuideModalField
from .profile import GuideProfile, SurfaceContact
from .contact_model import normal_force, friction_force

logger = logging.getLogger(__name__)

# Coordinate convention (NED): +x forward, +y right, +z down. The cross-section frame K_Pi
# shares these axes (up to the small bending rotation), so a slider's K_Pi (y, z) are its
# lateral (right) and vertical (down) offsets, and gravity (+z) seats a shoe on the +z floor.


@dataclass(frozen=True)
class SliderContact:
    """
    Result of the section 4 contact analysis for a single slider, projected in K_I.

    Attributes
    ----------
    x_r_i : float
        Axial station of the contact cross-section (Eq. 55).
    in_phase : bool
        Whether the slider is still within the guide (phase of action, Eq. 61).
    surfaces : List[SurfaceContact]
        Active guide faces (penetration > 0) from the profile.
    penetration_velocities : Dict[str, float]
        Per-face penetration velocity delta_dot (keyed by face label), for memory update.
    i_q_o1, i_m_o1 : Vector
        Resultant contact force and moment on the projectile at O1 (Eqs. B22-B23).
    i_q_pi, i_m_pi : Vector
        Reaction force and moment on the guide at Pi (Eqs. B24-B25).
    """
    x_r_i: float
    in_phase: bool
    surfaces: List[SurfaceContact]
    penetration_velocities: Dict[str, float]
    i_q_o1: Vector
    i_m_o1: Vector
    i_q_pi: Vector
    i_m_pi: Vector


def _cross_section(field: GuideModalField, a_ir: Matrix, x: float, p: Vector):
    """Return (A_IPi, I_r_Pi, phi_r, phi_theta) for the cross-section at axial station x."""
    phi_r, phi_theta = field.phi(x)
    theta_rp = a_ir.T @ (phi_theta @ p)
    a_ipi = a_ir @ small_rot(theta_rp)
    i_r_pi = x * a_ir[:, 0] + phi_r @ p
    return a_ipi, i_r_pi, phi_r, phi_theta


def contact_station(
        field: GuideModalField,
        a_ir: Matrix,
        i_r_vi: Vector,
        p: Vector,
        x0: float,
        bracket: float = 0.5,
) -> float:
    """
    Solve Eq. 55 for the contact station x_R,i: the axial station where the vector from the
    cross-section to the slider center V_i is perpendicular to the cross-section normal
    e^Pi_x, i.e. g(x) = (I_r_Vi - I_r_Pi(x)) . e^Pi_x(x) = 0. Uses a bracketed root find
    about the nominal station x0.
    """
    def g(x: float) -> float:
        a_ipi, i_r_pi, _, _ = _cross_section(field, a_ir, x, p)
        return float((i_r_vi - i_r_pi) @ a_ipi[:, 0])

    lo, hi = x0 - bracket, x0 + bracket
    for _ in range(6):
        if g(lo) * g(hi) <= 0.0:
            return float(brentq(g, lo, hi, xtol=1e-12))
        lo, hi = lo - bracket, hi + bracket
    logger.warning("contact_station: no sign change bracketing x0=%.4f; using nominal.", x0)
    return float(x0)


def evaluate_slider(
        kin: ProjectileKinematicState,
        field: GuideModalField,
        a_ir: Matrix,
        p: Vector,
        p_dot: Vector,
        slider,
        profile: GuideProfile,
        x_r: float,
        l_c: float,
        impact_velocities: Dict[str, float] | None = None,
        station_bracket: float = 0.5,
) -> SliderContact:
    """
    Full section 4 analysis for one slider: phase of action, contact station, penetration,
    impact velocity, and the resultant force/moment on the projectile and the guide.

    Parameters
    ----------
    kin : ProjectileKinematicState
        Section 3 kinematics at O1 (provides r_o1, r_dot_o1, a_ib, omega_ib).
    field : GuideModalField
        The guide modal field (evaluated at the per-slider station x_R,i).
    a_ir : Matrix
        Rotation A_IR (K_R -> K_I).
    p, p_dot : Vector
        Launch-vehicle modal coordinates and rates.
    slider : projectile.Slider
        The slider (body-frame offset B_r_O1Vi and radius).
    profile : GuideProfile
        The guide cross-section contact model.
    x_r : float
        Axial coordinate of O1 (needed for the phase-of-action test).
    l_c : float
        Distance from O1 to the guide front exit at t = 0 (Eq. 61).
    impact_velocities : Dict[str, float] | None
        Per-face impact-onset velocity delta_dot_minus (the section-5 memory). Faces not
        present are treated as making first contact (delta_dot_minus = current delta_dot).
    station_bracket : float
        Bracket size for the contact station root-find.
    """
    p = np.asarray(p, dtype=np.float64)
    p_dot = np.asarray(p_dot, dtype=np.float64)
    memory = impact_velocities or {}
    zeros = np.zeros(3, dtype=np.float64)

    br_o1vi = np.asarray(slider.position, dtype=np.float64)
    i_r_o1vi = kin.a_ib @ br_o1vi                      # I_r_O1Vi
    i_r_vi = kin.r_o1 + i_r_o1vi                        # I_r_Vi (Eq. 50)

    # --- Phase of action (Eq. 61) ---
    nominal_axial = x_r + float((a_ir.T @ i_r_o1vi)[0])
    if nominal_axial > l_c:
        return SliderContact(nominal_axial, False, [], {}, zeros, zeros, zeros, zeros)

    # --- Contact cross-section (Eq. 55) and its kinematics ---
    x_r_i = contact_station(field, a_ir, i_r_vi, p, x0=nominal_axial, bracket=station_bracket)
    a_ipi, i_r_pi, phi_r, phi_theta = _cross_section(field, a_ir, x_r_i, p)
    i_rdot_pi = phi_r @ p_dot                           # I_r_dot_Pi (Eq. 17, fixed section)
    omega_ipi = phi_theta @ p_dot                       # I_omega_IPi (Eq. 21)
    omega_pib = kin.omega_ib - omega_ipi

    pi_r_vi = a_ipi.T @ (i_r_vi - i_r_pi)               # slider center in K_Pi (Eq. 56 datum)

    # --- Impact velocity (Eqs. 62-64) ---
    i_delta_dot = (
        kin.r_dot_o1
        - skew_sym_mat(omega_ipi) @ kin.r_o1
        + skew_sym_mat(omega_pib) @ i_r_o1vi
        - i_rdot_pi
        + skew_sym_mat(omega_ipi) @ i_r_pi
    )
    pi_delta_dot = a_ipi.T @ i_delta_dot                # relative velocity of V_i in K_Pi

    # --- Per-face forces and resultant (Eqs. B16-B25) ---
    surfaces = profile.contacts(pi_r_vi, radius=slider.radius, station=x_r_i)
    force_kpi = np.zeros(3, dtype=np.float64)
    pen_vels: Dict[str, float] = {}
    for s in surfaces:
        # Penetration velocity: component of the relative velocity into the face (-normal).
        d_dot = float(-s.normal @ pi_delta_dot)
        pen_vels[s.label] = d_dot
        d_dot_minus = memory.get(s.label, d_dot)
        q_n = normal_force(s.penetration, d_dot, d_dot_minus, s.stiffness, s.restitution, s.exponent)

        # Coulomb friction opposes the tangential (sliding) relative velocity.
        v_t = pi_delta_dot - (pi_delta_dot @ s.normal) * s.normal
        vt_norm = float(np.linalg.norm(v_t))
        f_friction = np.zeros(3, dtype=np.float64)
        if vt_norm > 1e-12:
            f_friction = -friction_force(q_n, s.friction) * (v_t / vt_norm)

        force_kpi += q_n * s.normal + f_friction

    i_q_o1 = a_ipi @ force_kpi                           # Eq. B22
    i_m_o1 = skew_sym_mat(i_r_o1vi) @ i_q_o1             # Eq. B23
    i_q_pi = -i_q_o1                                     # Eq. B24
    i_m_pi = skew_sym_mat(i_r_vi - i_r_pi) @ i_q_pi      # Eq. B25 (I_r_Pi_Vi = I_r_Vi - I_r_Pi)

    return SliderContact(
        x_r_i=float(x_r_i),
        in_phase=True,
        surfaces=surfaces,
        penetration_velocities=pen_vels,
        i_q_o1=i_q_o1.astype(np.float64),
        i_m_o1=i_m_o1.astype(np.float64),
        i_q_pi=i_q_pi.astype(np.float64),
        i_m_pi=i_m_pi.astype(np.float64),
    )
