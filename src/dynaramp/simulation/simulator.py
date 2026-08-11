from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

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


@dataclass
class LaunchResult:
    """
    Time histories of a launch simulation. Each array is indexed by time step.

    Attributes
    ----------
    t : Vector
        Time [s], shape ``(steps, )``.
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
