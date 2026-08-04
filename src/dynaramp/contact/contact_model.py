from __future__ import annotations
import logging

import numpy as np

from ..materials import Material

logger = logging.getLogger(__name__)


def hertz_stiffness(radius: float, material_a: Material, material_b: Material) -> float:
    """
    Generalized contact stiffness K for a sphere against a plane (Eqs. B13-B14):

        K = 4 / (3 (sigma_a + sigma_b)) * sqrt(radius),   sigma_k = (1 - nu_k^2) / E_k.

    :param radius: Ball-head radius R of the slider [m].
    :param material_a: Material of one contacting body (e.g. the slider).
    :param material_b: Material of the other (e.g. the guide).
    :return: Generalized contact stiffness K.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}.")
    sigma = material_a.contact_sigma + material_b.contact_sigma
    return 4.0 / (3.0 * sigma) * float(np.sqrt(radius))


def linear_contact_stiffness(
        area: float,
        material_a: Material,
        material_b: Material,
        compliance_length: float,
) -> float:
    """
    Contact stiffness for a flat (conformal) contact, modelled as the compression of a
    material layer of nominal area `area` and characteristic depth `compliance_length`:

        K = E* * area / compliance_length,   E* = 1 / (sigma_a + sigma_b).

    Intended for use with exponent n = 1 in `normal_force` (units of K are N/m). Unlike the
    Hertzian sphere-plane case there is no unique stiffness for conformal contact; the
    compliance length is a modelling choice (roughly the depth of material that deforms).
    Pick K large enough that the peak penetration stays a small fraction of the clearance.

    :param area: Nominal contact area of the flat face [m^2].
    :param material_a: Material of one contacting body.
    :param material_b: Material of the other.
    :param compliance_length: Characteristic elastic depth L [m].
    :return: Linear contact stiffness K [N/m].
    """
    if area <= 0.0:
        raise ValueError(f"area must be positive, got {area}.")
    if compliance_length <= 0.0:
        raise ValueError(f"compliance_length must be positive, got {compliance_length}.")
    e_star = 1.0 / (material_a.contact_sigma + material_b.contact_sigma)
    return e_star * area / compliance_length


def normal_force(
        penetration: float,
        penetration_velocity: float,
        impact_velocity: float,
        stiffness: float,
        restitution: float,
        exponent: float = 1.5,
) -> float:
    """
    Continuous normal contact force with hysteresis damping (Lankarani-Nikravesh, Eq. B12):

        q_N = K delta^n [ 1 + (3 (1 - e^2) / 4) (delta_dot / delta_dot_minus) ].

    Returns 0 when there is no penetration, and never returns a negative (adhesive) force.
    When the impact velocity is zero (quasi-static contact) the damping term is dropped.

    :param penetration: delta, relative penetration depth (> 0 in contact).
    :param penetration_velocity: delta_dot, relative penetration velocity (approach > 0).
    :param impact_velocity: delta_dot_minus, penetration velocity at the onset of contact.
    :param stiffness: K, generalized contact stiffness (see `hertz_stiffness`).
    :param restitution: e, coefficient of restitution in [0, 1].
    :param exponent: n, force-penetration exponent (1.5 for a Hertzian sphere-plane).
    :return: Normal contact force magnitude q_N (>= 0).
    """
    if not 0.0 <= restitution <= 1.0:
        raise ValueError(f"restitution must be in [0, 1], got {restitution}.")
    if penetration <= 0.0:
        return 0.0

    hysteresis = 1.0
    if impact_velocity > 0.0:
        hysteresis += (3.0 * (1.0 - restitution ** 2) / 4.0) * (penetration_velocity / impact_velocity)

    q_n = stiffness * penetration ** exponent * hysteresis
    return max(q_n, 0.0)


def friction_force(normal_force_magnitude: float, friction_coefficient: float) -> float:
    """
    Coulomb friction magnitude (Eq. B15): q_T = mu * q_N.

    :param normal_force_magnitude: q_N, the normal contact force magnitude.
    :param friction_coefficient: mu, the Coulomb friction coefficient.
    :return: Friction force magnitude q_T.
    """
    return friction_coefficient * normal_force_magnitude
