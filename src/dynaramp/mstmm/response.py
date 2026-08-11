"""
Transient forced response of a linear multibody system by the augmented-eigenvector
method.

This is section 3 of Rui's *Transfer Matrix Method for Multibody Systems* -- the
companion to the eigenvalue machinery already in :mod:`dynaramp.mstmm.system`.
Where :meth:`System.natural_modes` answers "at what frequencies does this system
like to vibrate, and in what shape", this module answers "given a set of forces
applied to it, how does it actually move".

Theory
------
The body dynamics equation of an undamped multi-rigid-flexible system is

    M v_tt + K v = f                                            (Rui 3.90)

where ``v`` collects the physical coordinates of every body and beam in the
system and ``M``, ``K`` are the augmented mass and stiffness *operators*. The
physical response is expanded on the augmented eigenvectors ``V^k``:

    v = sum_k V^k q_k(t)                                        (Rui 3.91)

The augmented eigenvectors are orthogonal with respect to both operators,

    <M V^k, V^p> = delta_kp M_p ,  <K V^k, V^p> = delta_kp K_p  (Rui 3.93)

so substituting the expansion and taking the inner product with ``V^p``
decouples the system into one scalar oscillator per mode:

    q_p_tt + omega_p^2 q_p = <f, V^p> / M_p                     (Rui 3.94)

The modal force ``<f, V^p>`` of a point load is just its virtual work on the
mode shape: ``F . X^p(point) + T . Theta^p(point)``. Each oscillator is then
integrated independently and the physical motion recovered from (3.91).

The analogy worth holding onto: the augmented eigenvectors are a set of
"natural postures" the structure can hold. Any motion is a weighted blend of
those postures, and orthogonality guarantees that pushing on one posture never
spills energy into another -- so a system with hundreds of degrees of freedom
becomes a handful of independent mass-spring oscillators.

Degenerate (repeated) eigenfrequencies
--------------------------------------
Symmetric systems routinely produce repeated roots -- a body on springs of equal
stiffness in y and z has a two-dimensional eigenspace at a single frequency.
:meth:`System.natural_modes` already returns every eigenvector at such a frequency,
so the basis is complete before this module sees it.

What it does not do, because nothing else needs it, is make the vectors *within* a
repeated cluster orthogonal to each other in the augmented inner product. The SVD
hands back a basis that is orthonormal in R^n, which is not the same thing and does
not satisfy (3.93). :func:`augmented_modes` supplies that step, and it is what makes
the modal equations actually decouple.

Damping
-------
Rui section 3.7 treats damping by keeping the undamped augmented eigenvectors and
adding a modal damping term, which is what :func:`transient_response` does:

    q_p_tt + 2 zeta_p omega_p q_p_t + omega_p^2 q_p = <f, V^p> / M_p

``zeta_p`` may be given directly, or derived from Rayleigh coefficients
``(alpha, beta)`` as ``zeta_p = (alpha / omega_p + beta omega_p) / 2``. That is the
same convention as :func:`dynaramp.simulation.coupled.modal_damping_stiffness`, which
expresses it as the diagonal ``c_g = alpha + beta omega_p^2 = 2 zeta_p omega_p``.

What lives where
----------------
The pieces this module builds on are all plain :class:`System` operations, and stay
there so that an eigenvalue analysis can use them without pulling in the response
machinery:

    ``System.mode_shape``              a mode's shape at any material point
    ``System.modal_product``           the augmented inner product <M V^k, V^p>
    ``System.calc_system_modal_masses``  its diagonal, the modal masses
    ``System.eigenvectors_at``         every eigenvector at a known frequency

This module adds the orthogonalisation, the load projection and the time integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy import integrate

from ..common.types import EntityID, Vector, VectorLike, Matrix
from .system import Mode, System

logger = logging.getLogger(__name__)

__all__ = [
    "PointLoad",
    "ModalBasis",
    "TransientResponse",
    "augmented_modes",
    "transient_response",
]


# --------------------------------------------------------------------------------------
# Augmented eigenvectors
# --------------------------------------------------------------------------------------

def _combine(modes: Sequence[Mode], weights: Sequence[float]) -> Mode:
    """Form the linear combination ``sum_i w_i * mode_i`` of co-frequent modes."""
    keys = modes[0].internal_states.keys()
    states = {
        k: sum(w * m.internal_states[k] for w, m in zip(weights, modes))
        for k in keys
    }
    return Mode(frequency=modes[0].frequency, internal_states=states)


def _m_orthogonalise(system: System, modes: List[Mode], tol: float = 1e-10) -> List[Mode]:
    """
    Gram-Schmidt a set of co-frequent modes with respect to the augmented inner
    product, so that (3.93) holds within the repeated cluster.

    Any vector that collapses to (numerically) zero norm was linearly dependent on
    its predecessors -- or is a spurious null direction carrying no inertia -- and
    is dropped.
    """
    if len(modes) <= 1:
        return modes

    orthogonal: List[Mode] = []
    norms: List[float] = []
    for candidate in modes:
        current = candidate
        for basis_mode, basis_norm in zip(orthogonal, norms):
            coeff = system.modal_product(current, basis_mode) / basis_norm
            if abs(coeff) > 0.0:
                current = _combine([current, basis_mode], [1.0, -coeff])
        norm = system.modal_product(current, current)
        reference = system.modal_product(candidate, candidate)
        if norm <= tol * max(reference, 1.0):
            logger.debug(
                f"Dropping a linearly dependent / inertia-free direction at "
                f"{candidate.frequency:.6e} rad/s (modal mass {norm:.3e})."
            )
            continue
        orthogonal.append(current)
        norms.append(norm)
    return orthogonal


def augmented_modes(
        system: System,
        n_modes: int,
        omega_min: float = 0.0,
        omega_max: float = 1000.0,
        search_res: int = 10000,
        rtol: float = 1e-5,
        mtol: float = 1e-5,
        max_multiplicity: int = 6,
) -> List[Mode]:
    """
    The augmented eigenvectors of the system, ready to expand a response on.

    ``System.natural_modes`` already returns every eigenvector at a repeated
    eigenfrequency; what it does not do -- because it has no reason to -- is make the
    vectors within a repeated cluster orthogonal to each other in the augmented inner
    product. The SVD hands back a basis that is orthonormal in R^n, which is not the same
    thing and does not satisfy (3.93). This function supplies that last step, which is
    what makes the modal equations actually decouple.

    :param system: The system to analyse.
    :param n_modes: Number of *distinct eigenfrequencies* to retain from the search.
        The returned list may be longer if some of them are repeated.
    :param omega_min: Lower bound of the frequency sweep (rad/s).
    :param omega_max: Upper bound of the frequency sweep (rad/s).
    :param search_res: Number of sweep points.
    :param rtol: Reciprocal-condition-number tolerance for accepting a mode.
    :param mtol: Tolerance for merging near-identical frequencies.
    :param max_multiplicity: Cap on the multiplicity detected at any one frequency.
    :return: The augmented eigenvectors, sorted by frequency.
    """
    found = system.natural_modes(
        n_modes=n_modes, omega_min=omega_min, omega_max=omega_max,
        search_res=search_res, rtol=rtol, mtol=mtol,
        max_multiplicity=max_multiplicity,
    )

    # Group co-frequent modes, then orthogonalise each group in the augmented product.
    modes: List[Mode] = []
    clusters: List[List[Mode]] = []
    for mode in sorted(found, key=lambda m: m.frequency):
        if clusters and np.isclose(clusters[-1][0].frequency, mode.frequency, rtol=mtol):
            clusters[-1].append(mode)
        else:
            clusters.append([mode])
    for cluster in clusters:
        modes.extend(_m_orthogonalise(system, cluster))

    modes.sort(key=lambda m: m.frequency)
    logger.info(
        f"Assembled {len(modes)} augmented eigenvectors "
        f"across {len(clusters)} distinct eigenfrequencies."
    )
    return modes


# --------------------------------------------------------------------------------------
# Loads
# --------------------------------------------------------------------------------------

ForceLike = VectorLike | Callable[[float], VectorLike]


@dataclass
class PointLoad:
    """
    A force and/or moment applied at a fixed material point of the system.

    The point is identified by the element it belongs to plus its coordinates in
    that element's local frame -- the same coordinates used when declaring ports,
    so a load at a beam tip of length ``L`` is ``position=(L, 0, 0)``.

    ``force`` and ``moment`` are given in the global frame. Either may be a constant
    3-vector or a callable of time; a callable makes the load transient (a ramp, a
    pulse, a measured history), a constant makes it a step applied at ``t = 0``.

    :param element: ID of the element carrying the load.
    :param position: Application point, in the element's local frame.
    :param force: Force vector [N], constant or ``f(t)``.
    :param moment: Moment vector [N.m], constant or ``m(t)``.
    """
    element: EntityID
    position: VectorLike
    force: ForceLike = (0.0, 0.0, 0.0)
    moment: ForceLike = (0.0, 0.0, 0.0)

    @property
    def is_constant(self) -> bool:
        return not callable(self.force) and not callable(self.moment)

    def wrench(self, t: float) -> Vector:
        """The 6-component generalized load ``[F; T]`` at time ``t``."""
        f = self.force(t) if callable(self.force) else self.force
        m = self.moment(t) if callable(self.moment) else self.moment
        return np.concatenate([
            np.asarray(f, dtype=np.float64),
            np.asarray(m, dtype=np.float64),
        ])


# --------------------------------------------------------------------------------------
# Modal basis
# --------------------------------------------------------------------------------------

@dataclass
class ModalBasis:
    """
    A set of augmented eigenvectors together with everything needed to project loads
    onto them and to rebuild physical motion from them.

    :param system: The system the modes belong to.
    :param modes: The augmented eigenvectors.
    """
    system: System
    modes: List[Mode]
    modal_masses: Vector = field(init=False)
    frequencies: Vector = field(init=False)

    def __post_init__(self) -> None:
        self.frequencies = np.array([m.frequency for m in self.modes], dtype=np.float64)
        self.modal_masses = self.system.calc_system_modal_masses(self.modes)
        if np.any(self.modal_masses <= 0.0):
            bad = np.flatnonzero(self.modal_masses <= 0.0)
            err_msg = (
                f"Non-positive modal mass for mode(s) {bad.tolist()}; the augmented "
                f"eigenvectors are not a valid basis."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    @classmethod
    def build(cls, system: System, n_modes: int, **kwargs) -> "ModalBasis":
        """Convenience constructor: run :func:`augmented_modes` and wrap the result."""
        return cls(system=system, modes=augmented_modes(system, n_modes, **kwargs))

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def shape_at(self, e_id: EntityID, position: VectorLike) -> Matrix:
        """
        The mode shapes of every mode at one material point, as an ``(n_modes, 6)``
        array of ``[x, y, z, theta_x, theta_y, theta_z]``.
        """
        return np.array(
            [self.system.mode_shape(m, e_id, position) for m in self.modes],
            dtype=np.float64,
        )

    def modal_forces(self, loads: Sequence[PointLoad], t: float) -> Vector:
        """
        Project the applied loads onto every mode: ``f_p = <f, V^p>``.

        The inner product of a point load with a mode is its virtual work on that
        mode shape, ``F . X^p + T . Theta^p``.

        :param loads: The loads acting on the system.
        :param t: Evaluation time, passed to any time-dependent load.
        :return: The ``n_modes`` modal forces.
        """
        f = np.zeros(self.n_modes, dtype=np.float64)
        for load in loads:
            shapes = self.shape_at(load.element, load.position)  # (n_modes, 6)
            f += shapes @ load.wrench(t)
        return f

    def damping_ratios(
            self,
            zeta: float | Sequence[float] | None = None,
            rayleigh: Tuple[float, float] | None = None,
    ) -> Vector:
        """
        Per-mode damping ratios, from either a direct specification or Rayleigh
        coefficients ``(alpha, beta)`` with ``zeta_p = (alpha / omega_p + beta omega_p) / 2``.
        """
        if zeta is not None and rayleigh is not None:
            err_msg = "Specify either `zeta` or `rayleigh`, not both."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if rayleigh is not None:
            alpha, beta = rayleigh
            return 0.5 * (alpha / self.frequencies + beta * self.frequencies)
        if zeta is None:
            return np.zeros(self.n_modes, dtype=np.float64)
        return np.broadcast_to(
            np.asarray(zeta, dtype=np.float64), (self.n_modes,)
        ).astype(np.float64).copy()


# --------------------------------------------------------------------------------------
# Transient response
# --------------------------------------------------------------------------------------

@dataclass
class TransientResponse:
    """
    The result of a transient analysis: the generalized coordinates over time plus
    the means to turn them back into physical motion anywhere in the system.

    :param basis: The modal basis used.
    :param t: Time samples [s], shape ``(n_t,)``.
    :param q: Generalized coordinates, shape ``(n_t, n_modes)``.
    :param q_dot: Generalized velocities, shape ``(n_t, n_modes)``.
    :param q_ddot: Generalized accelerations, shape ``(n_t, n_modes)``.
    """
    basis: ModalBasis
    t: Vector
    q: Matrix
    q_dot: Matrix
    q_ddot: Matrix

    def displacement(self, e_id: EntityID, position: VectorLike) -> Matrix:
        """
        Physical motion at a material point over time: ``v = sum_p V^p q_p(t)``.

        :return: ``(n_t, 6)`` array of ``[x, y, z, theta_x, theta_y, theta_z]``.
        """
        return self.q @ self.basis.shape_at(e_id, position)

    def velocity(self, e_id: EntityID, position: VectorLike) -> Matrix:
        """Time derivative of :meth:`displacement`, same shape."""
        return self.q_dot @ self.basis.shape_at(e_id, position)

    def acceleration(self, e_id: EntityID, position: VectorLike) -> Matrix:
        """Second time derivative of :meth:`displacement`, same shape."""
        return self.q_ddot @ self.basis.shape_at(e_id, position)


def _constant_load_response(
        omega: Vector, zeta: Vector, q_ss: Vector, q0: Vector, qd0: Vector, t: Vector,
) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Closed-form step response of the decoupled oscillators (Rui 3.99 with a constant
    forcing term), used whenever every load is time-invariant.

    Exact at every sample, so a step-loaded system costs nothing to evaluate at high
    time resolution and carries no integration error to confuse a validation study.
    """
    t_col = t[:, None]
    w = omega[None, :]
    z = zeta[None, :]
    wd = w * np.sqrt(np.maximum(1.0 - z ** 2, 0.0))

    a = (q0 - q_ss)[None, :]
    # Undamped limit: wd -> w, and the sine coefficient reduces to qd0 / w.
    wd_safe = np.where(wd > 0.0, wd, 1.0)
    b = (qd0[None, :] + z * w * a) / wd_safe

    env = np.exp(-z * w * t_col)
    c, s = np.cos(wd * t_col), np.sin(wd * t_col)

    q = q_ss[None, :] + env * (a * c + b * s)
    q_dot = env * (
        (-z * w) * (a * c + b * s) + wd * (-a * s + b * c)
    )
    q_ddot = -2.0 * z * w * q_dot - w ** 2 * (q - q_ss[None, :])
    return q, q_dot, q_ddot


def transient_response(
        basis: ModalBasis,
        loads: Sequence[PointLoad],
        t_eval: VectorLike,
        zeta: float | Sequence[float] | None = None,
        rayleigh: Tuple[float, float] | None = None,
        q0: VectorLike | None = None,
        q_dot0: VectorLike | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-10,
) -> TransientResponse:
    """
    Integrate the decoupled modal equations (Rui 3.94) and return the system's
    transient response.

    Constant loads are solved in closed form; time-varying loads are integrated
    with an adaptive Runge-Kutta scheme.

    :param basis: The modal basis to project onto.
    :param loads: Point loads acting on the system.
    :param t_eval: Times at which to report the response [s].
    :param zeta: Modal damping ratio, scalar or one per mode.
    :param rayleigh: Rayleigh coefficients ``(alpha, beta)``, alternative to ``zeta``.
    :param q0: Initial generalized coordinates. Defaults to rest.
    :param q_dot0: Initial generalized velocities. Defaults to rest.
    :param rtol: Relative tolerance of the ODE solver (time-varying loads only).
    :param atol: Absolute tolerance of the ODE solver (time-varying loads only).
    :return: The :class:`TransientResponse`.
    """
    t = np.asarray(t_eval, dtype=np.float64)
    n = basis.n_modes
    omega = basis.frequencies
    m_p = basis.modal_masses
    zeta_p = basis.damping_ratios(zeta=zeta, rayleigh=rayleigh)

    q_init = np.zeros(n) if q0 is None else np.asarray(q0, dtype=np.float64)
    qd_init = np.zeros(n) if q_dot0 is None else np.asarray(q_dot0, dtype=np.float64)

    if all(load.is_constant for load in loads):
        f_p = basis.modal_forces(loads, 0.0)
        q_ss = f_p / (m_p * omega ** 2)
        q, q_dot, q_ddot = _constant_load_response(omega, zeta_p, q_ss, q_init, qd_init, t)
        return TransientResponse(basis=basis, t=t, q=q, q_dot=q_dot, q_ddot=q_ddot)

    def rhs(time: float, y: Vector) -> Vector:
        qq, qd = y[:n], y[n:]
        f_p = basis.modal_forces(loads, time)
        acc = f_p / m_p - 2.0 * zeta_p * omega * qd - omega ** 2 * qq
        return np.concatenate([qd, acc])

    sol = integrate.solve_ivp(
        rhs,
        t_span=(float(t[0]), float(t[-1])),
        y0=np.concatenate([q_init, qd_init]),
        t_eval=t,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:  # pragma: no cover - solver failure is exceptional
        err_msg = f"Modal integration failed: {sol.message}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    q = sol.y[:n].T
    q_dot = sol.y[n:].T
    q_ddot = np.array([rhs(ti, sol.y[:, i])[n:] for i, ti in enumerate(sol.t)])
    return TransientResponse(basis=basis, t=sol.t, q=q, q_dot=q_dot, q_ddot=q_ddot)
