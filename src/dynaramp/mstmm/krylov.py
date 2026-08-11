"""
Krylov-Duncan functions for the Euler-Bernoulli beam transfer matrix.

The four functions

    S(z) = (cosh z + cos z) / 2        U(z) = (cosh z - cos z) / 2
    T(z) = (sinh z + sin z) / 2        V(z) = (sinh z - sin z) / 2

are the standard basis for the bending solution of a uniform beam. In a transfer
matrix they never appear alone: every entry that carries a flexibility divides one
of them by a power of the wave number ``lam``, as ``T/lam``, ``U/(EI lam^2)``,
``V/(EI lam^3)``. Since the argument is itself ``z = lam x``, those quotients are

    T(z)/lam = x (T(z)/z),   U(z)/lam^2 = x^2 (U(z)/z^2),   V(z)/lam^3 = x^3 (V(z)/z^3)

so the wave number cancels exactly. This module therefore returns the *normalized*
group ``(S, T/z, U/z^2, V/z^3)`` and never divides by ``lam`` at all. Two problems
disappear as a result.

Zero frequency
--------------
At ``omega = 0`` every wave number vanishes and the un-normalized quotients are all
``0/0``. Their limits are finite -- they are precisely the static flexibilities of a
cantilever, ``x``, ``x^2/2EI``, ``x^3/6EI`` -- so the beam transfer matrix has a
perfectly well-defined static limit and the NaN is an artifact of the factorization,
not physics. With the normalized group the limit is reached exactly, from
``T/z -> 1``, ``U/z^2 -> 1/2``, ``V/z^3 -> 1/6``, with no special case anywhere.

Cancellation at small argument
------------------------------
``U`` and ``V`` are differences of nearly equal quantities: for small ``z``,
``cosh z - cos z ~ z^2`` is obtained by subtracting two numbers both close to 1, and
``sinh z - sin z ~ z^3/3`` likewise. At ``z ~ 1e-4`` -- routinely reached by a
finite-difference stencil near the start of a segment -- roughly eight significant
digits are lost. Below ``Z_SWITCH`` the functions are therefore evaluated from their
Taylor series instead, which is not merely accurate but *more* accurate than the
closed form there.

The series are unusually tidy. Writing ``w = z^4``, all four normalized functions are
the same sum with a shifted factorial:

    S    = sum_j w^j / (4j)!        T/z   = sum_j w^j / (4j+1)!
    U/z^2 = sum_j w^j / (4j+2)!     V/z^3 = sum_j w^j / (4j+3)!

Every one is even in ``z``, so negative spans need no special handling.
"""

from __future__ import annotations

import logging
from math import factorial
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Z_SWITCH", "krylov_normalized", "sinc"]

# Argument below which the Taylor series replaces the closed form. At z = 1 the series
# needs 5 terms to reach machine precision (w = 1, and the 5th term is 1/17! ~ 3e-15),
# while the closed form has already begun to lose digits to cancellation in U and V.
Z_SWITCH = 1.0

# Number of series terms retained. Chosen so that the truncation error at Z_SWITCH is
# below machine epsilon: the last term contributes 1/25! ~ 6e-26 at w = 1.
_N_TERMS = 7

# Reciprocal factorials 1/(4j+m)! for m = 0..3, in ascending powers of w = z^4, laid out
# in descending order for Horner evaluation.
_COEFFS = tuple(
    np.array([1.0 / factorial(4 * j + m) for j in range(_N_TERMS)][::-1], dtype=np.float64)
    for m in range(4)
)


def _horner(coeffs: np.ndarray, w: float) -> float:
    """Evaluate a polynomial given in descending-power order at ``w``."""
    acc = coeffs[0]
    for c in coeffs[1:]:
        acc = acc * w + c
    return float(acc)


def krylov_normalized(z: float) -> Tuple[float, float, float, float]:
    """
    The normalised Krylov-Duncan group ``(S, T/z, U/z^2, V/z^3)``.

    Returning the group rather than four separate functions is deliberate: a transfer
    matrix needs all four at the same argument, and computing them together costs one
    set of four transcendental evaluations instead of up to eighteen.

    Parameters
    ----------
    z : float
        The argument ``lam * x``. May be negative; all four returned functions are
        even in ``z``.

    Returns
    -------
    tuple of float
        ``(s, t1, u2, v3)`` where ``s = S(z)``, ``t1 = T(z)/z``, ``u2 = U(z)/z^2``
        and ``v3 = V(z)/z^3``. At ``z = 0`` these are exactly ``(1, 1, 1/2, 1/6)``.

    Notes
    -----
    For ``|z| > Z_SWITCH`` the closed form is used. For very large ``|z|`` (beyond
    about 350) ``cosh`` overflows; such an argument means the retained frequency is
    far above the range where a slender-beam model is meaningful.
    """
    if abs(z) <= Z_SWITCH:
        w = z ** 4
        return (
            _horner(_COEFFS[0], w),
            _horner(_COEFFS[1], w),
            _horner(_COEFFS[2], w),
            _horner(_COEFFS[3], w),
        )

    cz, sz = np.cos(z), np.sin(z)
    chz, shz = np.cosh(z), np.sinh(z)
    z2 = z * z
    return (
        float(0.5 * (chz + cz)),
        float(0.5 * (shz + sz) / z),
        float(0.5 * (chz - cz) / z2),
        float(0.5 * (shz - sz) / (z2 * z)),
    )


def sinc(z: float) -> float:
    """
    The unnormalized cardinal sine ``sin(z) / z``, with the removable singularity filled.

    Plays the same role for the axial and torsional entries of the beam transfer matrix
    that :func:`krylov_normalized` plays for the bending ones: it lets
    ``sin(beta x) / beta`` be written as ``x * sinc(beta x)``, cancelling the wave
    number so that ``omega = 0`` yields the static limit ``x`` exactly.

    Parameters
    ----------
    z : float
        The argument ``beta * x``.

    Returns
    -------
    float
        ``sin(z)/z``, and exactly ``1.0`` at ``z = 0``.
    """
    if z == 0.0:
        return 1.0
    return float(np.sin(z) / z)
