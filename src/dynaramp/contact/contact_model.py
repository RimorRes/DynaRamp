from __future__ import annotations
import logging

import numpy as np

from ..materials import Material

logger = logging.getLogger(__name__)


def hertz_stiffness(radius: float, material_a: Material, material_b: Material) -> float:
    """
    Generalized contact stiffness for a sphere against a plane (Eqs. B13-B14).

    ``K = 4 / (3 (sigma_a + sigma_b)) * sqrt(radius)``, with
    ``sigma_k = (1 - nu_k^2) / E_k``.

    Parameters
    ----------
    radius : float
        Ball-head radius ``R`` of the slider [m].
    material_a : Material
        Material of one contacting body, e.g. the slider.
    material_b : Material
        Material of the other, e.g. the guide.

    Returns
    -------
    float
        The generalized contact stiffness ``K``. Intended for use with exponent
        ``n = 1.5`` in :func:`normal_force`.

    Raises
    ------
    ValueError
        If ``radius`` is not positive.
    """
    if radius <= 0.0:
        err_msg = f"radius must be positive, got {radius}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    sigma = material_a.contact_sigma + material_b.contact_sigma
    return 4.0 / (3.0 * sigma) * float(np.sqrt(radius))


def linear_contact_stiffness(
        area: float,
        material_a: Material,
        material_b: Material,
        compliance_length: float,
) -> float:
    """
    Contact stiffness for a flat (conformal) contact.

    Modeled as the compression of a material layer of nominal ``area`` and
    characteristic depth ``compliance_length``:

    ``K = E* * area / compliance_length``, with ``E* = 1 / (sigma_a + sigma_b)``.

    Parameters
    ----------
    area : float
        Nominal contact area of the flat face [m^2].
    material_a : Material
        Material of one contacting body.
    material_b : Material
        Material of the other.
    compliance_length : float
        Characteristic elastic depth ``L`` [m].

    Returns
    -------
    float
        The linear contact stiffness ``K`` [N/m]. Intended for use with exponent
        ``n = 1`` in :func:`normal_force`.

    Raises
    ------
    ValueError
        If ``area`` or ``compliance_length`` is not positive.

    Notes
    -----
    Unlike the Hertzian sphere-plane case there is no unique stiffness for conformal
    contact; the compliance length is a modeling choice, roughly the depth of material
    that deforms. Pick ``K`` large enough that the peak penetration stays a small
    fraction of the clearance, but no larger -- an over-stiff penalty makes an explicit
    integrator's step size collapse.
    """
    if area <= 0.0:
        err_msg = f"area must be positive, got {area}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    if compliance_length <= 0.0:
        err_msg = f"compliance_length must be positive, got {compliance_length}."
        logger.error(err_msg)
        raise ValueError(err_msg)
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
    Continuous normal contact force with hysteresis damping (Lankarani-Nikravesh, Eq. B12).

    ``q_N = K delta^n [1 + (3 (1 - e^2) / 4) (delta_dot / delta_dot_minus)]``.

    Parameters
    ----------
    penetration : float
        ``delta``, relative penetration depth; positive in contact.
    penetration_velocity : float
        ``delta_dot``, relative penetration velocity; positive on approach.
    impact_velocity : float
        ``delta_dot_minus``, the penetration velocity at the onset of contact.
    stiffness : float
        ``K``, the generalized contact stiffness. See :func:`hertz_stiffness`.
    restitution : float
        ``e``, the coefficient of restitution, in [0, 1].
    exponent : float
        ``n``, the force-penetration exponent; 1.5 for a Hertzian sphere-plane contact.

    Returns
    -------
    float
        The normal contact force magnitude ``q_N``, never negative.

    Raises
    ------
    ValueError
        If ``restitution`` is outside [0, 1].

    Notes
    -----
    Returns zero when there is no penetration, and never returns a negative (adhesive)
    force. When the impact velocity is zero -- quasi-static contact -- the damping term
    is dropped.
    """
    if not 0.0 <= restitution <= 1.0:
        err_msg = f"restitution must be in [0, 1], got {restitution}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    if penetration <= 0.0:
        return 0.0

    hysteresis = 1.0
    if impact_velocity > 0.0:
        hysteresis += (3.0 * (1.0 - restitution ** 2) / 4.0) * (penetration_velocity / impact_velocity)

    q_n = stiffness * penetration ** exponent * hysteresis
    return max(q_n, 0.0)


def friction_force(normal_force_magnitude: float, friction_coefficient: float) -> float:
    """
    Coulomb friction magnitude (Eq. B15): ``q_T = mu * q_N``.

    Parameters
    ----------
    normal_force_magnitude : float
        ``q_N``, the normal contact force magnitude.
    friction_coefficient : float
        ``mu``, the Coulomb friction coefficient.

    Returns
    -------
    float
        The friction force magnitude ``q_T``.
    """
    return friction_coefficient * normal_force_magnitude
