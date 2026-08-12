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

    M v_tt + K v = f (Rui 3.90)

Where ``v`` collects the physical coordinates of every body and beam in the
system and ``M``, ``K`` are the augmented mass and stiffness *operators*. The
physical response is expanded on the augmented eigenvectors ``V^k``:

    v = sum_k V^k q_k(t) (Rui 3.91)

The augmented eigenvectors are orthogonal with respect to both operators.

    <M V^k, V^p> = delta_kp M_p, <K V^k, V^p> = delta_kp K_p (Rui 3.93)

So substituting the expansion and taking the inner product with ``V^p``
decouples the system into one scalar oscillator per mode:

    q_p_tt + omega_p^2 q_p = <f, V^p> / M_p (Rui 3.94)

The modal force ``<f, V^p>`` of a point load is just its virtual work on the
mode shape: ``F . X^p(point) + T . Theta^p(point)``. Each oscillator is then
integrated independently and the physical motion recovered from (3.91).

The augmented eigenvectors are a set of "natural shapes" the structure can hold.
Any motion is a weighted blend of those shapes, and orthogonality guarantees
that "pushing" on one shape never spills energy into another -- so a system with
hundreds of degrees of freedom becomes a handful of independent mass-spring oscillators.

Degenerate (repeated) eigenfrequencies
--------------------------------------
Symmetric systems routinely produce repeated roots -- a body on springs of equal
stiffness in y and z has a two-dimensional eigenspace at a single frequency.
:meth:`System.natural_modes` already returns every eigenvector at such a frequency,
so the basis is complete before this module sees it.

Because nothing else requires it, it does not make the vectors *within* a
repeated cluster orthogonal to each other in the augmented inner product. The SVD
hands back a basis that is orthonormal in R^n, which is a different thing and does
not satisfy (3.93). :func:`augmented_modes` supplies that step, and it is what makes
the modal equations actually decouple.

Damping
-------
Rui section 3.7 treats damping by keeping the undamped augmented eigenvectors and
adding a modal damping term, which is what :func:`transient_response` does:

    q_p_tt + c_p q_p_t + omega_p^2 q_p = <f, V^p> / M_p

The modal damping coefficient ``c_p = 2 zeta_p omega_p`` may be given through a
damping ratio ``zeta_p`` or through Rayleigh coefficients ``(alpha, beta)``; both
routes go through :mod:`dynaramp.common.modal`, which is the single definition of
Rayleigh damping in the package and is shared with
:func:`dynaramp.simulation.coupled.modal_damping_stiffness`.

Rigid-body modes
----------------
A mode at ``omega = 0`` is a rigid-body freedom: it has no restoring force, so it
does not oscillate about a static offset. The oscillator solution
degenerates accordingly, and :func:`transient_response` integrates that case in its
own closed form rather than dividing by a zero stiffness.

What lives where
----------------
The pieces this module builds on are all plain :class:`System` operations, and stay
there so that an eigenvalue analysis can use them without pulling in the response
machinery:

    ``System.mode_shape`` a mode's shape at any material point
    ``System.modal_product`` the augmented inner product <M V^k, V^p>
    ``System.calc_system_modal_masses`` its diagonal, the modal masses
    ``System.eigenvectors_at`` every eigenvector at a known frequency

This module adds the orthogonalization, the load projection, and the time integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple, cast

import numpy as np
from scipy import integrate

from ..common.types import EntityID, Vector, VectorLike, Matrix
from ..common.modal import project_wrench, rayleigh_damping_ratios, rayleigh_modal_damping
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
    """
    Form the linear combination ``sum_i w_i * mode_i`` of co-frequent modes.

    Parameters
    ----------
    modes : Sequence[Mode]
        Modes at the same eigenfrequency.
    weights : Sequence[float]
        One weight per mode.

    Returns
    -------
    Mode
        The combined mode.
    """
    keys = modes[0].internal_states.keys()
    states = {
        k: sum(w * m.internal_states[k] for w, m in zip(weights, modes))
        for k in keys
    }
    states = cast(Dict[str, Vector], states)
    return Mode(frequency=modes[0].frequency, internal_states=states)


def _m_orthogonalize(system: System, modes: List[Mode], tol: float = 1e-10) -> List[Mode]:
    """
    Gram-Schmidt a set of co-frequent modes in the augmented inner product.

    Makes (3.93) hold within a repeated cluster. Any vector that collapses to a
    numerically zero norm was linearly dependent on its predecessors -- or is a
    spurious null direction carrying no inertia -- and is dropped.

    Parameters
    ----------
    system : System
        The system providing the inner product.
    modes : List[Mode]
        Modes sharing one eigenfrequency.
    tol : float
        Relative tolerance below which a vector is treated as dependent.

    Returns
    -------
    List[Mode]
        The surviving, mutually orthogonal modes.
    """
    if len(modes) <= 1:
        return modes

    orthogonal: List[Mode] = []
    norms: List[float] = []
    for candidate in modes:
        # The candidate's own norm is a fixed reference for the dependence test and does
        # not change as the candidate is deflated. Computing it once avoids a second full
        # integration over every element per candidate.
        reference = system.modal_product(candidate, candidate)
        current = candidate
        for basis_mode, basis_norm in zip(orthogonal, norms):
            coeff = system.modal_product(current, basis_mode) / basis_norm
            if abs(coeff) > 0.0:
                current = _combine([current, basis_mode], [1.0, -coeff])
        norm = system.modal_product(current, current)
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
    product. The SVD hands back a basis that is orthonormal in R^n, which is a different
    thing and does not satisfy (3.93). This function supplies that last step, which is
    what makes the modal equations actually decouple.

    Parameters
    ----------
    system : System
        The system to analyze.
    n_modes : int
        Number of *distinct eigenfrequencies* to retain from the search. The returned
        list may be longer if some of them are repeated.
    omega_min : float
        Lower bound of the frequency sweep [rad/s].
    omega_max : float
        Upper bound of the frequency sweep [rad/s].
    search_res : int
        Number of sweep points.
    rtol : float
        Reciprocal-condition-number tolerance for accepting a mode.
    mtol : float
        Tolerance for merging near-identical frequencies.
    max_multiplicity : int
        Cap on the multiplicity detected at any one frequency.

    Returns
    -------
    List[Mode]
        The augmented eigenvectors, sorted by frequency.
    """
    found = system.natural_modes(
        n_modes=n_modes, omega_min=omega_min, omega_max=omega_max,
        search_res=search_res, rtol=rtol, mtol=mtol,
        max_multiplicity=max_multiplicity,
    )

    # Group co-frequent modes, then orthogonalize each group in the augmented product.
    modes: List[Mode] = []
    clusters: List[List[Mode]] = []
    for mode in sorted(found, key=lambda m: m.frequency):
        if clusters and np.isclose(clusters[-1][0].frequency, mode.frequency, rtol=mtol):
            clusters[-1].append(mode)
        else:
            clusters.append([mode])
    for cluster in clusters:
        modes.extend(_m_orthogonalize(system, cluster))

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

    Attributes
    ----------
    element : EntityID
        ID of the element carrying the load.
    position : VectorLike
        Application point, in the element's local frame.
    force : ForceLike
        Force vector [N], in the global frame. Either a constant 3-vector or a callable
        of time; a callable makes the load transient (a ramp, a pulse, a measured
        history), a constant makes it a step applied at ``t = 0``.
    moment : ForceLike
        Moment vector [N.m], in the global frame, with the same constant-or-callable
        convention as ``force``.
    """
    element: EntityID
    position: VectorLike
    force: ForceLike = (0.0, 0.0, 0.0)
    moment: ForceLike = (0.0, 0.0, 0.0)

    @property
    def is_constant(self) -> bool:
        """Whether neither the force nor the moment varies with time."""
        return not callable(self.force) and not callable(self.moment)

    def wrench(self, t: float) -> Vector:
        """
        The 6-component generalized load ``[F; T]`` at one instant.

        Parameters
        ----------
        t : float
            Evaluation time [s].

        Returns
        -------
        Vector
            The generalized load.
        """
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
    A set of augmented eigenvectors plus everything needed to use them.

    Projects load onto the modes and rebuild physical motion from them.

    Attributes
    ----------
    system : System
        The system the modes belong to.
    modes : List[Mode]
        The augmented eigenvectors.
    modal_masses : Vector
        Their modal masses, computed on construction.
    frequencies : Vector
        Their eigenfrequencies, in the same order.
    """
    system: System
    modes: List[Mode]
    modal_masses: Vector = field(init=False)
    frequencies: Vector = field(init=False)

    def __post_init__(self) -> None:
        self.frequencies = np.array([m.frequency for m in self.modes], dtype=np.float64)
        self.modal_masses = self.system.calc_system_modal_masses(self.modes)
        # Mode shapes at a fixed material point do not change with time, but a load
        # projection asks for them at every step of an integration. They are memoised
        # per (element, position) rather than recomputed.
        self._shape_cache: Dict[Tuple[EntityID, bytes], Matrix] = {}
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
        """
        Convenience constructor: run :func:`augmented_modes` and wrap the result.

        Parameters
        ----------
        system : System
            The system to analyze.
        n_modes : int
            Number of distinct eigenfrequencies to retain.
        **kwargs
            Forwarded to :func:`augmented_modes`.

        Returns
        -------
        ModalBasis
            The assembled basis.
        """
        return cls(system=system, modes=augmented_modes(system, n_modes, **kwargs))

    @property
    def n_modes(self) -> int:
        """Number of modes in the basis."""
        return len(self.modes)

    def shape_at(self, e_id: EntityID, position: VectorLike) -> Matrix:
        """
        The mode shapes of every mode at one material point.

        Parameters
        ----------
        e_id : EntityID
            Element the point belongs to.
        position : VectorLike
            Point coordinates in the element's local frame.

        Returns
        -------
        Matrix
            An ``(n_modes, 6)`` array of ``[x, y, z, theta_x, theta_y, theta_z]``.
            Read-only, as it is shared from a cache.
        """
        pos = np.asarray(position, dtype=np.float64)
        key = (e_id, pos.tobytes())
        cached = self._shape_cache.get(key)
        if cached is not None:
            return cached

        shapes = self.system.mode_shapes_at(self.modes, e_id, pos)
        shapes.setflags(write=False)
        self._shape_cache[key] = shapes
        return shapes

    def modal_forces(self, loads: Sequence[PointLoad], t: float) -> Vector:
        """
        Project the applied loads onto every mode: ``f_p = <f, V^p>``.

        Parameters
        ----------
        loads : Sequence[PointLoad]
            The loads acting on the system.
        t : float
            Evaluation time, passed to any time-dependent load.

        Returns
        -------
        Vector
            The ``n_modes`` modal forces.
        """
        f = np.zeros(self.n_modes, dtype=np.float64)
        for load in loads:
            f += project_wrench(self.shape_at(load.element, load.position), load.wrench(t))
        return f

    def damping_ratios(
            self,
            zeta: float | Sequence[float] | None = None,
            rayleigh: Tuple[float, float] | None = None,
    ) -> Vector:
        """
        Per-mode damping ratios, from a direct specification or Rayleigh coefficients.

        Parameters
        ----------
        zeta : float | Sequence[float] | None
            Damping ratio, scalar or one per mode.
        rayleigh : tuple of float | None
            Rayleigh coefficients ``(alpha, beta)``, an alternative to ``zeta``.

        Returns
        -------
        Vector
            The ``n_modes`` damping ratios; all zero if neither argument is given.

        Raises
        ------
        ValueError
            If both arguments are supplied.
        """
        if zeta is not None and rayleigh is not None:
            err_msg = "Specify either `zeta` or `rayleigh`, not both."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if rayleigh is not None:
            return rayleigh_damping_ratios(self.frequencies, *rayleigh)
        if zeta is None:
            return np.zeros(self.n_modes, dtype=np.float64)
        return np.broadcast_to(
            np.asarray(zeta, dtype=np.float64), (self.n_modes,)
        ).astype(np.float64).copy()

    def modal_damping(
            self,
            zeta: float | Sequence[float] | None = None,
            rayleigh: Tuple[float, float] | None = None,
    ) -> Vector:
        """
        Per-mode modal damping coefficients ``c_p = 2 zeta_p omega_p``.

        The form the oscillator equations are actually integrated in, and the only one
        that stays meaningful for a rigid-body mode.

        Parameters
        ----------
        zeta : float | Sequence[float] | None
            Damping ratio, scalar or one per mode.
        rayleigh : tuple of float | None
            Rayleigh coefficients ``(alpha, beta)``, an alternative to ``zeta``.

        Returns
        -------
        Vector
            The ``n_modes`` modal damping coefficients.

        Raises
        ------
        ValueError
            If both arguments are supplied.
        """
        if zeta is not None and rayleigh is not None:
            err_msg = "Specify either `zeta` or `rayleigh`, not both."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if rayleigh is not None:
            # Taken directly, so a rigid-body mode keeps its mass-proportional damping.
            return rayleigh_modal_damping(self.frequencies, *rayleigh)
        return 2.0 * self.damping_ratios(zeta=zeta) * self.frequencies


# --------------------------------------------------------------------------------------
# Transient response
# --------------------------------------------------------------------------------------

@dataclass
class TransientResponse:
    """
    The result of a transient analysis.

    The generalized coordinates over time, plus the means to turn them back into
    physical motion anywhere in the system.

    Attributes
    ----------
    basis : ModalBasis
        The modal basis used.
    t : Vector
        Time samples [s], shape ``(n_t,)``.
    q : Matrix
        Generalized coordinates, shape ``(n_t, n_modes)``.
    q_dot : Matrix
        Generalized velocities, shape ``(n_t, n_modes)``.
    q_ddot : Matrix
        Generalized accelerations, shape ``(n_t, n_modes)``.
    """
    basis: ModalBasis
    t: Vector
    q: Matrix
    q_dot: Matrix
    q_ddot: Matrix

    def displacement(self, e_id: EntityID, position: VectorLike) -> Matrix:
        """
        Physical motion at a material point over time: ``v = sum_p V^p q_p(t)``.

        Parameters
        ----------
        e_id : EntityID
            Element the point belongs to.
        position : VectorLike
            Point coordinates in the element's local frame.

        Returns
        -------
        Matrix
            ``(n_t, 6)`` array of ``[x, y, z, theta_x, theta_y, theta_z]``.
        """
        return self.q @ self.basis.shape_at(e_id, position)

    def velocity(self, e_id: EntityID, position: VectorLike) -> Matrix:
        """
        Time derivative of :meth:`displacement`.

        Parameters
        ----------
        e_id : EntityID
            Element the point belongs to.
        position : VectorLike
            Point coordinates in the element's local frame.

        Returns
        -------
        Matrix
            ``(n_t, 6)`` array.
        """
        return self.q_dot @ self.basis.shape_at(e_id, position)

    def acceleration(self, e_id: EntityID, position: VectorLike) -> Matrix:
        """
        Second time derivative of :meth:`displacement`.

        Parameters
        ----------
        e_id : EntityID
            Element the point belongs to.
        position : VectorLike
            Point coordinates in the element's local frame.

        Returns
        -------
        Matrix
            ``(n_t, 6)`` array.
        """
        return self.q_ddot @ self.basis.shape_at(e_id, position)


def _constant_load_response(
        omega: Vector, c_p: Vector, accel: Vector, q0: Vector, qd0: Vector, t: Vector,
) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Closed-form step response of the decoupled oscillators.

    Rui 3.99 with a constant forcing term, used whenever every load is time-invariant.
    Exact at every sample, so a step-loaded system costs nothing to evaluate at high
    time resolution and carries no integration error to confuse a validation study.

    Parameters
    ----------
    omega : Vector
        Modal frequencies [rad/s].
    c_p : Vector
        Modal damping coefficients ``2 zeta_p omega_p``.
    accel : Vector
        Constant modal forcing per unit modal mass, ``f_p / M_p``.
    q0, qd0 : Vector
        Initial generalized coordinates and velocities.
    t : Vector
        Time samples [s].

    Returns
    -------
    tuple of Matrix
        ``(q, q_dot, q_ddot)``, each ``(n_t, n_modes)``.

    Notes
    -----
    Elastic and rigid-body modes are integrated by different closed forms. An elastic
    mode relaxes towards the static offset ``a / omega^2``; a rigid-body mode has no
    such offset and instead accelerates freely, decaying towards a terminal velocity
    ``a / c_p`` when mass-proportional damping is present.
    """
    t_col = t[:, None]
    n_t, n = t.size, omega.size

    q = np.zeros((n_t, n), dtype=np.float64)
    q_dot = np.zeros((n_t, n), dtype=np.float64)
    q_ddot = np.zeros((n_t, n), dtype=np.float64)

    elastic = omega > 0.0

    # --- elastic modes: damped oscillation about the static offset ---
    if np.any(elastic):
        w = omega[elastic][None, :]
        c = c_p[elastic][None, :]
        z = 0.5 * c / w
        if np.any(z >= 1.0):
            logger.warning(
                "Critically or over-damped mode(s) present (zeta >= 1); the closed-form "
                "step response assumes an underdamped oscillator and will be inaccurate "
                "for them."
            )
        wd = w * np.sqrt(np.maximum(1.0 - z ** 2, 0.0))

        q_ss = accel[elastic][None, :] / w ** 2
        a = q0[elastic][None, :] - q_ss
        # Undamped limit: wd -> w, and the sine coefficient reduces to qd0 / w.
        wd_safe = np.where(wd > 0.0, wd, 1.0)
        b = (qd0[elastic][None, :] + z * w * a) / wd_safe

        env = np.exp(-z * w * t_col)
        cs, sn = np.cos(wd * t_col), np.sin(wd * t_col)

        q_e = q_ss + env * (a * cs + b * sn)
        qd_e = env * ((-z * w) * (a * cs + b * sn) + wd * (-a * sn + b * cs))
        q[:, elastic] = q_e
        q_dot[:, elastic] = qd_e
        q_ddot[:, elastic] = -c * qd_e - w ** 2 * (q_e - q_ss)

    # --- rigid-body modes: free acceleration, with optional mass-proportional drag ---
    rigid = ~elastic
    if np.any(rigid):
        a_r = accel[rigid][None, :]
        c_r = c_p[rigid][None, :]
        q0_r = q0[rigid][None, :]
        qd0_r = qd0[rigid][None, :]

        drag = c_r > 0.0
        # Undamped: q = q0 + qd0 t + a t^2 / 2.
        q_r = q0_r + qd0_r * t_col + 0.5 * a_r * t_col ** 2
        qd_r = qd0_r + a_r * t_col
        qdd_r = np.broadcast_to(a_r, q_r.shape).copy()

        if np.any(drag):
            c_safe = np.where(drag, c_r, 1.0)
            terminal = a_r / c_safe
            decay = np.exp(-c_safe * t_col)
            q_damped = q0_r + terminal * t_col + (qd0_r - terminal) * (1.0 - decay) / c_safe
            qd_damped = terminal + (qd0_r - terminal) * decay
            qdd_damped = -c_safe * (qd0_r - terminal) * decay
            mask = np.broadcast_to(drag, q_r.shape)
            q_r = np.where(mask, q_damped, q_r)
            qd_r = np.where(mask, qd_damped, qd_r)
            qdd_r = np.where(mask, qdd_damped, qdd_r)

        q[:, rigid] = q_r
        q_dot[:, rigid] = qd_r
        q_ddot[:, rigid] = qdd_r

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
    Integrate the decoupled modal equations (Rui 3.94) and return the transient response.

    Constant loads are solved in closed form; time-varying loads are integrated with an
    adaptive Runge-Kutta scheme.

    Parameters
    ----------
    basis : ModalBasis
        The modal basis to project onto.
    loads : Sequence[PointLoad]
        Point loads acting on the system.
    t_eval : VectorLike
        Times at which to report the response [s].
    zeta : float | Sequence[float] | None
        Modal damping ratio, scalar or one per mode.
    rayleigh : tuple of float | None
        Rayleigh coefficients ``(alpha, beta)``, an alternative to ``zeta``.
    q0 : VectorLike | None
        Initial generalized coordinates. Defaults to rest.
    q_dot0 : VectorLike | None
        Initial generalized velocities. Defaults to rest.
    rtol : float
        Relative tolerance of the ODE solver (time-varying loads only).
    atol : float
        Absolute tolerance of the ODE solver (time-varying loads only).

    Returns
    -------
    TransientResponse
        The response.

    Raises
    ------
    RuntimeError
        If the adaptive integration fails.
    """
    t = np.asarray(t_eval, dtype=np.float64)
    n = basis.n_modes
    omega = basis.frequencies
    m_p = basis.modal_masses
    c_p = basis.modal_damping(zeta=zeta, rayleigh=rayleigh)

    q_init = np.zeros(n) if q0 is None else np.asarray(q0, dtype=np.float64)
    qd_init = np.zeros(n) if q_dot0 is None else np.asarray(q_dot0, dtype=np.float64)

    if all(load.is_constant for load in loads):
        accel = basis.modal_forces(loads, 0.0) / m_p
        q, q_dot, q_ddot = _constant_load_response(omega, c_p, accel, q_init, qd_init, t)
        return TransientResponse(basis=basis, t=t, q=q, q_dot=q_dot, q_ddot=q_ddot)

    def rhs(time: float, y: Vector) -> Vector:
        qq, qd = y[:n], y[n:]
        f_p = basis.modal_forces(loads, time)
        acc = f_p / m_p - c_p * qd - omega ** 2 * qq
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
