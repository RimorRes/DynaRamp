from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from ..common.types import Vector, Matrix
from ..mstmm.response import ModalBasis
from ..projectile.state import ProjectileState
from ..projectile.kinematics import ProjectileKinematics
from ..projectile.dynamics import ProjectileEOM
from ..projectile.projectile import Projectile
from ..projectile.modal_field import GuideModalField
from ..contact.solver import ContactSolver, ContactMemory, ContactResult
from .external_forces import ExternalForce, sum_external_forces
from .coupled import modal_contact_force, modal_damping_stiffness, assemble_and_solve

logger = logging.getLogger(__name__)

# Number of progress reports emitted over a run, regardless of its length.
_PROGRESS_REPORTS = 20

# Weight of the Tikhonov term that pins rank-deficient directions of the equilibrium
# solve to their seed. Small enough not to bias a constrained direction, large enough
# that an unconstrained one cannot drift.
_NULL_PENALTY = 1e-6

# Largest transverse acceleration [m/s^2] still accepted as an equilibrium.
_EQUILIBRIUM_TOL = 1e-6


@dataclass
class LaunchResult:
    """
    Time histories of a launch simulation. Each array is indexed by time step.

    Attributes
    ----------
    t : Vector
        Time [s], shape ``(steps,)``.
    x : Matrix
        Projectile configuration ``[x_R, y_L, z_L, gamma, psi, phi]``, shape
        ``(steps, 6)``.
    y : Matrix
        Projectile quasi-velocity ``[s_B_dot; L_omega_LB]``, shape ``(steps, 6)``.
    p : Matrix
        Vehicle modal coordinates, shape ``(steps, n)``.
    contact_force : Matrix
        Total contact force on the projectile at O1, shape ``(steps, 3)``.
    contact_moment : Matrix
        Total contact moment on the projectile at O1, shape ``(steps, 3)``.
    exited : bool
        Whether the run terminated because all sliders left the guide, as opposed to
        reaching ``t_max``.
    """
    t: Vector
    x: Matrix
    y: Matrix
    p: Matrix
    contact_force: Matrix
    contact_moment: Matrix
    exited: bool

    @property
    def attitude(self) -> Matrix:
        """Attitude history ``[gamma, psi, phi]``, shape ``(steps, 3)``."""
        return self.x[:, 3:6]

    @property
    def angular_velocity(self) -> Matrix:
        """Body angular velocity history ``L_omega_LB``, shape ``(steps, 3)``."""
        return self.y[:, 3:6]


class LaunchSimulator:
    """
    Time-integrates the coupled launch dynamics (section 5, Fig. 6).

    Uses a fixed-step RK4 scheme. The stepping is isolated in :meth:`_rk4_step` so a
    stiff or implicit integrator can be swapped in later without touching the physics in
    :meth:`_rhs`.

    Parameters
    ----------
    field : GuideModalField
        The guide modal field, which also carries the solved MSTMM system and the
        retained modes.
    projectile : Projectile
        The projectile's inertial properties.
    solver : ContactSolver
        Configured with the sliders, guide profile and exit stations.
    forces : Sequence[ExternalForce]
        Non-contact loads: gravity, thrust, and so on.
    rayleigh : tuple of float
        Rayleigh damping coefficients ``(alpha, beta)`` for the vehicle modes.
    """

    def __init__(
            self,
            field: GuideModalField,
            projectile: Projectile,
            solver: ContactSolver,
            forces: Sequence[ExternalForce] = (),
            rayleigh: Tuple[float, float] = (0.0, 0.0),
    ):
        self.field = field
        self.projectile = projectile
        self.solver = solver
        self.forces = list(forces)
        self.n = field.n_modes
        # ModalBasis carries the frequencies and modal masses together and rejects a
        # basis with a non-positive modal mass, which would otherwise surface much later
        # as a silently wrong response.
        self.basis = ModalBasis(system=field.system, modes=field.modes)
        self.modal_masses = self.basis.modal_masses
        self.c_g, self.k_g = modal_damping_stiffness(self.basis.frequencies, rayleigh)
        # Stateless and reusable, so it is built once rather than on every one of the
        # four right-hand-side evaluations per step.
        self.kinematics = ProjectileKinematics(field)

    # --- state packing: z = [p (n), p_dot (n), x (6), y (6)] ---
    def _split(self, z: Vector) -> Tuple[Vector, Vector, Vector, Vector]:
        """
        Unpack the integration state vector.

        Parameters
        ----------
        z : Vector
            The packed state.

        Returns
        -------
        tuple of Vector
            ``(p, p_dot, x, y)``.
        """
        n = self.n
        return z[:n], z[n:2 * n], z[2 * n:2 * n + 6], z[2 * n + 6:2 * n + 12]

    def _rhs(self, t: float, z: Vector, memory: ContactMemory) -> Tuple[Vector, ContactResult]:
        """
        The state derivative, plus the contact result at this state.

        Parameters
        ----------
        t : float
            Current time [s].
        z : Vector
            The packed state.
        memory : ContactMemory
            Impact-onset velocity memory carried from the step start.

        Returns
        -------
        tuple
            ``(z_dot, contact)``. The contact result is returned alongside because the
            caller needs it for the memory update, the termination test and the history.
        """
        p, p_dot, x, y = self._split(z)
        state = ProjectileState(x, y)

        kin = self.kinematics.evaluate(state, p, p_dot)
        eom = ProjectileEOM.assemble(kin, self.projectile)
        contact = self.solver.evaluate(kin, state.x_r, p, p_dot, memory)
        f_g = modal_contact_force(self.field, self.modal_masses, contact.guide_reactions)
        ext_q, ext_m = sum_external_forces(self.forces, kin, self.projectile, t)

        p_ddot, y_dot = assemble_and_solve(
            eom, self.c_g, self.k_g, p, p_dot, f_g,
            contact.sum_q_o1, contact.sum_m_o1, ext_q, ext_m,
        )
        x_dot = state.config_rates()                       # x_dot = H^{-1} y
        z_dot = np.concatenate([p_dot, p_ddot, x_dot, y_dot])
        return z_dot, contact

    def _rk4_step(
            self,
            t: float,
            z: Vector,
            dt: float,
            memory: ContactMemory,
    ) -> Tuple[Vector, ContactMemory, ContactResult]:
        """
        Advance the state by one RK4 step.

        Parameters
        ----------
        t : float
            Time at the start of the step [s].
        z : Vector
            State at the start of the step.
        dt : float
            Step size [s].
        memory : ContactMemory
            Impact-onset velocity memory at the start of the step.

        Returns
        -------
        tuple
            ``(z_next, memory_next, contact)``. The contact memory is held fixed across
            the four stages and refreshed once per step, from the step-start evaluation;
            the returned contact result is likewise the step-start one.
        """
        k1, contact = self._rhs(t, z, memory)
        memory_next = contact.memory
        k2, _ = self._rhs(t + 0.5 * dt, z + 0.5 * dt * k1, memory)
        k3, _ = self._rhs(t + 0.5 * dt, z + 0.5 * dt * k2, memory)
        k4, _ = self._rhs(t + dt, z + dt * k3, memory)
        z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z_next, memory_next, contact

    def _seed_equilibrium(self, u: Vector, raw_residual, n: int, steps: int = 2000) -> Vector:
        """
        Build a starting point for the equilibrium solve that is already in contact.

        Two cheap stages. First the transverse coordinates are marched along their own
        acceleration -- the direction the projectile actually wants to move -- with the
        modal coordinates held at zero, until it settles against the guide. Then the
        modal block is solved in one shot: with the velocities zero the modal equation is
        just ``p_ddot = f_g - omega^2 p``, so ``p + p_ddot / omega^2`` lands exactly on
        the modal equilibrium for the contact forces now acting.

        Parameters
        ----------
        u : Vector
            Starting guess, ``[y_L, z_L, gamma, psi, phi, p...]``.
        raw_residual : callable
            Returns ``[p_ddot, y_dot[1:6]]`` at a state.
        n : int
            Number of vehicle modes.
        steps : int
            Cap on marching iterations.

        Returns
        -------
        Vector
            A seed in contact, with the guide already carrying its load.
        """
        u = u.copy()
        # Small enough not to overshoot the stiffest retained mode within one step.
        omega_max = float(np.max(self.basis.frequencies)) if n else 1.0
        step = 0.5 * (0.5 / max(omega_max, 1.0)) ** 2

        free_flight = raw_residual(u)[n:]
        reference = max(float(np.max(np.abs(free_flight))), 1e-12)

        # Phase A -- cross the gap. While the projectile is clear of the guide the
        # residual does not merely fail to improve, it does not change at all: nothing
        # depends on position yet. A step chosen for the stiffest mode would need tens of
        # thousands of iterations to cross a millimetre of clearance, so the step grows
        # geometrically until the residual finally moves, which is contact.
        engaged = False
        for _ in range(steps):
            r = raw_residual(u)[n:]
            if float(np.max(np.abs(r - free_flight))) > 1e-3 * reference:
                engaged = True
                break
            u[:5] += step * r
            step *= 1.5
        if not engaged:
            logger.warning(
                "Relaxation never reached contact; the equilibrium solve is starting "
                "from a floating state and will probably fail. Pass an `x_guess` that "
                "is already resting against the guide."
            )

        # Phase B -- settle. Contact now gives the residual a gradient, so the ordinary
        # rule applies: accept a step that reduces it, shrink on overshoot.
        prev = float(np.max(np.abs(raw_residual(u)[n:])))
        step *= 0.05
        for _ in range(steps):
            if prev < 1.0:
                break
            r = raw_residual(u)[n:]
            trial = u.copy()
            trial[:5] += step * r
            nxt = float(np.max(np.abs(raw_residual(trial)[n:])))
            if nxt < prev:
                u, prev = trial, nxt
                step *= 1.3
            else:
                step *= 0.4
                if step < 1e-30:
                    break

        omega2 = self.basis.frequencies ** 2
        elastic = omega2 > 0.0
        modal = raw_residual(u)[:n]
        u[5:][elastic] += modal[elastic] / omega2[elastic]
        logger.debug("Equilibrium seed: transverse residual %.3e m/s^2", prev)
        return u

    def equilibrium_state(
            self,
            x_r: float = 0.0,
            t: float = 0.0,
            x_guess: Vector | None = None,
            p_guess: Vector | None = None,
            tol: float = 1e-10,
            diff_step: float = 1e-6,
    ) -> Tuple[Vector, Vector]:
        """
        Solve for the state the system rests in before the launch begins.

        A launch simulation started from a straight, unloaded guide is not starting from
        rest: the guide is already carrying its share of the projectile's weight, and
        releasing it at ``t = 0`` rings it at its own natural frequencies. That transient
        is an artifact of the initial condition, not physics, and it contaminates exactly
        the early part of the record a launch study cares about.

        This solves it away by finding the configuration and vehicle modal coordinates at
        which every acceleration vanishes, which is the initialization of Eq. 12.

        Parameters
        ----------
        x_r : float
            Axial station to place the projectile at. Held fixed.
        t : float
            Time at which the loads are evaluated -- relevant when the thrust curve is
            non-zero at the start.
        x_guess : Vector | None
            Starting guess for the full configuration. Defaults to the projectile
            centered in the guide, which is adequate whenever a clearance is small.
        p_guess : Vector | None
            Starting guess for the vehicle modal coordinates. Defaults to zero.
        tol : float
            Residual tolerance passed to the root finder.
        diff_step : float
            Relative step for the finite-difference Jacobian. The default is far larger
            than the solver's own, deliberately -- see the notes.

        Returns
        -------
        tuple of Vector
            ``(x0, p0)``: the equilibrium configuration and the vehicle modal
            coordinates. The matching velocities are zero, so ``y0`` and ``p_dot0`` are
            simply zeros.

        Raises
        ------
        RuntimeError
            If the equilibrium solve does not converge.

        Notes
        -----
        The axial direction is deliberately excluded. A projectile under thrust has no
        axial equilibrium -- that is the entire point of the exercise -- so the residual
        covers the five transverse and angular quasi-accelerations and the ``n`` modal
        accelerations, against the five transverse configuration coordinates and the
        ``n`` modal coordinates. The system is square.

        Two numerical hazards make this harder than it looks, and both are handled here.

        Contact forces vanish identically until a face is penetrated, so wherever the
        projectile floats free of the guide the residual is not merely large but *flat*:
        its derivative with respect to position is exactly zero and a Newton step has
        nothing to work with. :meth:`_seed_equilibrium` walks into contact first.

        The problem is also routinely rank-deficient. A projectile whose sliders all lie
        on its own axis has nothing resisting roll, so roll is a genuine null direction
        and an unregularized solve will happily send it to infinity while reporting a
        perfect residual. A weak Tikhonov term pulls the solution toward the seed, which
        pins the null directions at their physically sensible starting values and leaves
        the determinate ones alone.

        The Jacobian step needs care for a third reason. Evaluating the residual runs the
        whole contact chain, including a bracketed root find for each slider's contact
        station, so it is only smooth down to about ``1e-12``. A solver's default relative
        step of ``sqrt(eps)`` puts the perturbation of a transverse coordinate of order
        ``1e-4 m`` at roughly ``1e-12 m`` -- right on that noise floor, where the
        difference quotient measures round-off rather than a derivative. The solve then
        wanders until it exhausts its evaluation budget. ``diff_step`` defaults to ``1e-6``
        instead, which is coarse enough to clear the noise and still local; it turns a
        1200-evaluation failure into convergence in about fifteen.
        """
        n = self.n
        x_guess = np.zeros(6) if x_guess is None else np.asarray(x_guess, dtype=np.float64).copy()
        x_guess[0] = x_r
        p_guess = np.zeros(n) if p_guess is None else np.asarray(p_guess, dtype=np.float64)
        omega2 = self.basis.frequencies ** 2

        def raw_residual(u: Vector) -> Vector:
            x = np.concatenate([[x_r], u[:5]])
            p = u[5:]
            z = np.concatenate([p, np.zeros(n), x, np.zeros(6)])
            z_dot, _ = self._rhs(t, z, {})
            p_ddot = z_dot[n:2 * n]
            y_dot = z_dot[2 * n + 6:2 * n + 12]
            return np.concatenate([p_ddot, y_dot[1:6]])

        seed = self._seed_equilibrium(
            np.concatenate([x_guess[1:6], p_guess]), raw_residual, n)

        # Per-block scales. The modal coordinates carry the scale of the eigenvectors,
        # which the SVD leaves arbitrary and which is typically many orders away from the
        # transverse coordinates in meters; comparing the two unscaled is meaningless.
        scale = np.empty_like(seed)
        scale[:5] = np.maximum(np.abs(seed[:5]), 1e-4)
        scale[5:] = np.maximum(np.abs(seed[5:]), 1e-3 * max(np.max(np.abs(seed[5:])), 1.0))

        def residual(u: Vector) -> Vector:
            r = raw_residual(u)
            # Modal rows divided by omega^2 so they read as a displacement error, which
            # puts them on the same footing as the transverse accelerations.
            physical = np.concatenate([
                np.where(omega2 > 0.0, r[:n] / np.where(omega2 > 0.0, omega2, 1.0), 0.0),
                r[n:],
            ])
            return np.concatenate([physical, _NULL_PENALTY * (u - seed) / scale])

        sol = least_squares(residual, seed, method="trf", x_scale="jac",
                            diff_step=diff_step, xtol=tol, ftol=tol, gtol=tol)
        r_final = raw_residual(sol.x)
        res = float(np.max(np.abs(r_final[n:])))
        modal_res = float(np.max(np.abs(r_final[:n] / np.where(omega2 > 0.0, omega2, 1.0))))

        if res > _EQUILIBRIUM_TOL:
            err_msg = (
                f"Equilibrium solve did not converge: max transverse acceleration "
                f"{res:.3e} m/s^2 remains. Try an `x_guess` already in contact with the "
                f"guide, or check that the projectile is actually constrained."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        x0 = np.concatenate([[x_r], sol.x[:5]])
        p0 = sol.x[5:]
        logger.info(
            "Equilibrium at x_R = %.4f m: y_L = %.3e m, z_L = %.3e m, attitude %s deg; "
            "residual %.2e m/s^2 (transverse), %.2e (modal)",
            x_r, x0[1], x0[2], np.array2string(np.degrees(x0[3:6]), precision=5),
            res, modal_res,
        )
        return x0, p0

    def run(
            self,
            x0: Vector,
            y0: Vector,
            dt: float,
            t_max: float,
            p0: Vector | None = None,
            p_dot0: Vector | None = None,
    ) -> LaunchResult:
        """
        Integrate the launch from an initial state.

        Stops early once every slider has left the guide (Eq. 61).

        Parameters
        ----------
        x0 : Vector
            Initial projectile configuration ``[x_R, y_L, z_L, gamma, psi, phi]``.
        y0 : Vector
            Initial projectile quasi-velocity ``[s_B_dot; L_omega_LB]``.
        dt : float
            Fixed step size [s].
        t_max : float
            Maximum simulated time [s].
        p0 : Vector | None
            Initial vehicle modal coordinates. Defaults to rest.
        p_dot0 : Vector | None
            Initial vehicle modal rates. Defaults to rest.

        Returns
        -------
        LaunchResult
            The time histories.

        Raises
        ------
        ValueError
            If ``dt`` or ``t_max`` is not positive.
        """
        if dt <= 0.0:
            err_msg = f"dt must be positive, got {dt}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if t_max <= 0.0:
            err_msg = f"t_max must be positive, got {t_max}."
            logger.error(err_msg)
            raise ValueError(err_msg)

        n = self.n
        p0 = np.zeros(n) if p0 is None else np.asarray(p0, dtype=np.float64)
        p_dot0 = np.zeros(n) if p_dot0 is None else np.asarray(p_dot0, dtype=np.float64)
        z = np.concatenate([p0, p_dot0, np.asarray(x0, dtype=np.float64), np.asarray(y0, dtype=np.float64)])
        memory: ContactMemory = {}

        ts: List[float] = []
        xs: List[Vector] = []
        ys: List[Vector] = []
        ps: List[Vector] = []
        fqs: List[Vector] = []
        fms: List[Vector] = []

        t = 0.0
        exited = False
        n_steps = int(np.ceil(t_max / dt))
        # Report a fixed number of times whatever the run length. The previous form,
        # `step_i % (n_steps // 1000)`, divided by zero for any run shorter than 1000
        # steps -- which is every short diagnostic run.
        report_every = max(1, n_steps // _PROGRESS_REPORTS)
        logger.info("Starting launch integration: %d steps of %.3e s (t_max = %.4f s).",
                    n_steps, dt, t_max)

        for step_i in range(n_steps):
            z_next, memory_next, contact = self._rk4_step(t, z, dt, memory)

            # record the state at t (before advancing)
            p, p_dot, x, y = self._split(z)
            ts.append(t)
            xs.append(x.copy())
            ys.append(y.copy())
            ps.append(p.copy())
            fqs.append(contact.sum_q_o1.copy())
            fms.append(contact.sum_m_o1.copy())

            # termination: all sliders have exited the guide
            if contact.slider_contacts and all(not sc.in_phase for sc in contact.slider_contacts):
                exited = True
                logger.info("All sliders exited the guide at t = %.6f s (step %d).", t, step_i)
                break

            if step_i % report_every == 0:
                logger.info("(%6.1f%%) step %d / %d, t = %.4f s, x_R = %.4f m",
                            100.0 * step_i / n_steps, step_i, n_steps, t, x[0])

            z, memory, t = z_next, memory_next, t + dt

        logger.info("Integration finished after %d recorded steps (exited = %s).", len(ts), exited)

        return LaunchResult(
            t=np.array(ts, dtype=np.float64),
            x=np.array(xs, dtype=np.float64),
            y=np.array(ys, dtype=np.float64),
            p=np.array(ps, dtype=np.float64),
            contact_force=np.array(fqs, dtype=np.float64),
            contact_moment=np.array(fms, dtype=np.float64),
            exited=exited,
        )
