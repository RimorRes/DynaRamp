from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import List, Sequence, Tuple, cast

import numpy as np

from ..common.types import Vector, Matrix
from ..mstmm.response import ModalBasis
from ..projectile.state import ProjectileState
from ..projectile.kinematics import ProjectileKinematics
from ..projectile.dynamics import ProjectileEOM
from ..projectile.projectile import Projectile
from ..projectile.modal_field import GuideModalField
from ..contact.solver import ContactSolver, ContactMemory
from .external_forces import ExternalForce, sum_external_forces
from .coupled import modal_contact_force, modal_damping_stiffness, assemble_and_solve

logger = logging.getLogger(__name__)


@dataclass
class LaunchResult:
    """
    Time histories of a launch simulation (each array is indexed by time step).

    Attributes
    ----------
    t : Vector                       (steps, )      time [s]
    x : Matrix                       (steps, 6)    projectile config [x_R, y_L, z_L, gamma, psi, phi]
    y : Matrix                       (steps, 6)    projectile quasi-velocity [s_B_dot; L_omega_LB]
    p : Matrix                       (steps, n)    vehicle modal coordinates
    contact_force, contact_moment : Matrix (steps, 3)  total contact resultants on the projectile at O1
    exited : bool                    terminated because all sliders left the guide (vs t_max)
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
        """Pitch/yaw/roll history [gamma, psi, phi], (steps, 3)."""
        return self.x[:, 3:6]

    @property
    def angular_velocity(self) -> Matrix:
        """Body angular velocity history L_omega_LB, (steps, 3)."""
        return self.y[:, 3:6]


class LaunchSimulator:
    """
    Time-integrates the coupled launch dynamics (section 5, Fig. 6) with a fixed-step RK4
    scheme. The stepping is isolated in `_rk4_step` so a stiff/implicit integrator can be
    swapped in later without touching the physics (`_rhs`).

    Parameters
    ----------
    field : GuideModalField
        The guide modal field (also carries the solved MSTMM system and retained modes).
    projectile : Projectile
        The projectile's inertial properties.
    solver : ContactSolver
        Configured with the sliders, guide profile and exit stations.
    forces : Sequence[ExternalForce]
        Non-contact loads (gravity, thrust, ...).
    rayleigh : (float, float)
        Rayleigh damping coefficients (alpha, beta) for the vehicle modes.
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

    # --- state packing: z = [p (n), p_dot (n), x (6), y (6)] ---
    def _split(self, z: Vector):
        n = self.n
        return z[:n], z[n:2 * n], z[2 * n:2 * n + 6], z[2 * n + 6:2 * n + 12]

    def _rhs(self, t: float, z: Vector, memory: ContactMemory):
        """State derivative z_dot and the contact result (for memory, termination, history)."""
        p, p_dot, x, y = self._split(z)
        state = ProjectileState(x, y)

        kin = ProjectileKinematics(self.field).evaluate(state, p, p_dot)
        eom = ProjectileEOM.assemble(kin, self.projectile)
        contact = self.solver.evaluate(kin, state.x_r, p, p_dot, memory)
        f_g = modal_contact_force(self.field, self.modal_masses, contact.guide_reactions)
        f_g = cast(Vector, cast(object, f_g))  # mypy can't see that modal_contact_force returns a Vector
        ext_q, ext_m = sum_external_forces(self.forces, kin, self.projectile, t)

        p_ddot, y_dot = assemble_and_solve(
            eom, self.c_g, self.k_g, p, p_dot, f_g,
            contact.sum_q_o1, contact.sum_m_o1, ext_q, ext_m,
        )
        x_dot = state.config_rates()                       # x_dot = H^{-1} y
        z_dot = np.concatenate([p_dot, p_ddot, x_dot, y_dot])
        return z_dot, contact

    def _rk4_step(self, t: float, z: Vector, dt: float, memory: ContactMemory):
        """One RK4 step. The contact memory is held across the four stages and refreshed
        once per step (from the step-start evaluation); returns (z_next, memory_next, k1_contact)."""
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
        Integrate from the initial projectile config x0 and quasi-velocity y0 (and optional
        initial vehicle modal state p0, p_dot0 -- default rest) for up to t_max, stopping
        early once every slider has left the guide (Eq. 61).
        """
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
        print(f"Max number of simulation steps: {n_steps}")
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
                break

            if step_i % (n_steps // 1000) < 1:
                print(f"({step_i/n_steps:6.1%}) {step_i:>{len(str(n_steps))}d} / {n_steps} steps")

            z, memory, t = z_next, memory_next, t + dt

        return LaunchResult(
            t=np.array(ts, dtype=np.float64),
            x=np.array(xs, dtype=np.float64),
            y=np.array(ys, dtype=np.float64),
            p=np.array(ps, dtype=np.float64),
            contact_force=np.array(fqs, dtype=np.float64),
            contact_moment=np.array(fms, dtype=np.float64),
            exited=exited,
        )
