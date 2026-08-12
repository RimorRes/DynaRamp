"""
Modal-mechanics helpers shared by the structural, contact, and simulation layers.

Both quantities defined here were previously spelled out several times across the
package, in mutually inconsistent parameterizations. Keeping one definition of each
means a change of convention is a change in one place.
"""

from __future__ import annotations

import logging

import numpy as np

from .types import Vector, VectorLike, Matrix

logger = logging.getLogger(__name__)

__all__ = ["project_wrench", "rayleigh_modal_damping", "rayleigh_damping_ratios"]


def project_wrench(shapes: Matrix, wrench: VectorLike) -> Vector:
    """
    Project a generalized load onto a set of mode shapes: ``f_p = <f, V^p>``.

    The inner product of a point load with a mode is its virtual work on that mode
    shape, ``F . X^p + T . Theta^p``. Written with the shapes stacked as rows, this is
    a single matrix-vector product.

    Parameters
    ----------
    shapes : Matrix
        Mode shapes at the load's application point, ``(n_modes, 6)``, each row being
        ``[x, y, z, theta_x, theta_y, theta_z]``.
    wrench : VectorLike
        The 6-component generalized load ``[F; T]`` at that point, in the same frame
        as the shapes.

    Returns
    -------
    Vector
        The ``n_modes`` generalized forces.

    Notes
    -----
    The ``(n_modes, 6)`` row convention is the one used throughout the package. A
    mode-shape field expressed as the pair of ``(3, n_modes)`` blocks
    ``(Phi_r, Phi_theta)`` is the transpose of this layout: ``np.vstack([phi_r,
    phi_theta]).T`` converts it.
    """
    return (np.asarray(shapes, dtype=np.float64)
            @ np.asarray(wrench, dtype=np.float64)).astype(np.float64)


def rayleigh_modal_damping(
        frequencies: VectorLike,
        alpha: float = 0.0,
        beta: float = 0.0,
) -> Vector:
    """
    Per-mode modal damping coefficients implied by ``C = alpha M + beta K``.

    Because the modes are both M- and K-orthogonal, ``C`` is diagonal in modal
    coordinates. Using ``K_p = omega_p^2 M_p``, each mode's damping coefficient
    *per unit modal mass* is

        c_p = alpha + beta omega_p^2 = 2 zeta_p omega_p.

    This is the single source of truth for Rayleigh damping in the package. The damping
    *ratios* and the assembled ``C`` matrix are both derived from it.

    Parameters
    ----------
    frequencies : VectorLike
        The natural frequencies ``omega_p`` [rad/s].
    alpha : float
        Mass-proportional Rayleigh coefficient.
    beta : float
        Stiffness-proportional Rayleigh coefficient.

    Returns
    -------
    Vector
        The ``n_modes`` modal damping coefficients.

    Notes
    -----
    Finite at every frequency, including ``omega = 0``: a rigid-body mode is genuinely
    damped by the mass-proportional term, at rate ``alpha``. This is the form to use
    when a rigid-body freedom is present; the damping *ratio* is not defined there.
    """
    omega = np.asarray(frequencies, dtype=np.float64)
    return (alpha + beta * omega ** 2).astype(np.float64)


def rayleigh_damping_ratios(
        frequencies: VectorLike,
        alpha: float = 0.0,
        beta: float = 0.0,
) -> Vector:
    """
    Per-mode damping ratios implied by Rayleigh damping ``C = alpha M + beta K``.

    Derived from :func:`rayleigh_modal_damping` by ``zeta_p = c_p / (2 omega_p)``, i.e.

        zeta_p = (alpha / omega_p + beta omega_p) / 2.

    Parameters
    ----------
    frequencies : VectorLike
        The natural frequencies ``omega_p`` [rad/s].
    alpha : float
        Mass-proportional Rayleigh coefficient.
    beta : float
        Stiffness-proportional Rayleigh coefficient.

    Returns
    -------
    Vector
        The ``n_modes`` damping ratios.

    Notes
    -----
    The mass-proportional term diverges as ``omega -> 0``: a rigid-body mode has no
    restoring force, so a damping *ratio* is not defined for it. Rather than return an
    infinity that would silently poison an integration, the ratio of a zero-frequency
    mode is reported as ``0.0``. Such a mode is still damped -- see
    :func:`rayleigh_modal_damping`, which expresses that damping in the absolute terms
    where it remains meaningful.
    """
    omega = np.asarray(frequencies, dtype=np.float64)
    c_p = rayleigh_modal_damping(omega, alpha, beta)
    zeta = np.zeros_like(omega)

    rigid = omega <= 0.0
    if np.any(rigid) and alpha != 0.0:
        logger.debug(
            "Rayleigh damping ratio undefined for %d rigid-body mode(s); reported as zero. "
            "Their damping is carried by the modal damping coefficient instead.",
            int(np.count_nonzero(rigid)),
        )

    elastic = ~rigid
    zeta[elastic] = 0.5 * c_p[elastic] / omega[elastic]
    return zeta.astype(np.float64)
