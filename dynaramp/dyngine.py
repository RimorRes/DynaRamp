import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import root, least_squares

from .launchrail import LaunchRail
from .vehicle import RigidRocket2D


class System:

    def __init__(self, ramp: LaunchRail, vehicle: RigidRocket2D):
        self.ramp = ramp
        self.vehicle = vehicle

        n = self.ramp.n_modes + 3  # [s, y, theta, eta_1, ..., eta_n]
        self.M = np.zeros((n, n))
        self.M[0, 0] = self.vehicle.mass
        self.M[1, 1] = self.vehicle.mass
        self.M[2, 2] = self.vehicle.J
        self.M[3:, 3:] = self.ramp.M

        self.C = np.zeros_like(self.M)
        self.C[3:, 3:] = self.ramp.C

        self._f_int_jax = self._build_f_int_jax()

    def _build_f_int_jax(self):
        """
        Build a JAX-traceable (and JIT-compiled) internal force function.
        Internal forces = elastic restoring forces = +dV/dq (Lagrangian convention).

        Generalized coordinates:  q = [s, y, theta, eta_1, ..., eta_n]

        For shoe spring k, let the beam deflection at the contact point be
            w_k = phi(x_k)·eta
        and define the spring deformation using the implemented sign convention:
            delta_k = w_k - y - rk - dk·theta - l0
            N_k     = spring_const * delta_k
        Then the generalized internal-force contribution is
            f_int[y]     += -N_k
            f_int[theta] += -dk * N_k
            f_int[eta_i] += phi_i(x_k) * N_k
        Plus beam modal stiffness (outside shoe loop):
            f_int[eta_i] += K_ii * eta_i
        """
        shoes = self.vehicle.shoes
        jax_shapes = self.ramp.jax_modal_shapes
        k_beam = jnp.array(self.ramp.K)
        n_modes = self.ramp.n_modes
        dphi = [jax.grad(phi) for phi in jax_shapes]

        def f_int(q):
            s = q[0]
            y = q[1]
            theta = q[2]
            eta = q[3:]

            f = jnp.zeros(3 + n_modes)

            for shoe in shoes:
                # Contact location along rail: s + dk - rk*theta (small-angle)
                x_k = s + shoe.dk - shoe.rk * theta
                # Only apply force if shoe has not been released
                is_active = x_k <= shoe.x_release
                # Beam transverse deflection at contact (modal expansion)
                w_k = sum(eta[i] * jax_shapes[i](x_k) for i in range(n_modes))
                dw_k = sum(eta[i] * dphi[i](x_k) for i in range(n_modes))
                # Spring deformation: extension = rail_y − shoe_tip_y − l0
                # shoe_tip_y ≈ y + rk + dk·θ  (rocket hangs below rail, y up)
                delta_k = w_k - y - shoe.rk - shoe.dk * theta - shoe.l0
                n_k = jnp.where(is_active, shoe.spring_const * delta_k, 0.0)

                f = f.at[0].add(n_k * dw_k)
                f = f.at[1].add(-n_k)
                f = f.at[2].add(-n_k * (shoe.dk + shoe.rk * dw_k))
                for i in range(n_modes):
                    f = f.at[3 + i].add(jax_shapes[i](x_k) * n_k)

            # Beam modal stiffness restoring force (must be outside shoe loop)
            f = f.at[3:].add(k_beam @ eta)

            return f

        return jax.jit(f_int)

    def compute_static_equilibrium(self, s: float, q_free0: np.ndarray | None = None, tol: float = 1e-4) -> np.ndarray:
        n_free = 2 + self.ramp.n_modes  # [y, theta, eta_1, ..., eta_n]
        if q_free0 is None:
            q_free0 = np.zeros(n_free)
            if self.vehicle.shoes:
                # Rough sag guess: align shoe preload with rail at zero deflection.
                q_free0[0] = -min(shoe.rk + shoe.l0 for shoe in self.vehicle.shoes)

        j_func = jax.jacobian(self._f_int_jax)

        def _assemble_q(q_free):
            q = np.zeros(3 + self.ramp.n_modes)
            q[0] = s
            q[1:] = q_free
            return q

        def residual(q_free):
            q = _assemble_q(q_free)
            q_dot = np.zeros_like(q)
            f_int = np.asarray(self.internal_forces(q))
            f_ext = np.asarray(self.external_forces(q, q_dot, 0.0))
            return (f_ext - f_int)[1:]

        def jac(q_free):
            q = _assemble_q(q_free)
            jac_full = np.asarray(j_func(jnp.array(q, dtype=float)))
            return -jac_full[1:, 1:]

        sol = root(residual, q_free0, jac=jac, tol=tol, method="hybr")
        if sol.success:
            q_free = sol.x
        else:
            print(f"Static equilibrium solve failed: {sol.message}; ")
            ls = least_squares(residual, q_free0, jac=jac, xtol=tol, ftol=tol, gtol=tol)
            if not ls.success:
                raise RuntimeError(
                    f"Static equilibrium solve failed: {sol.message}; "
                    f"least_squares: {ls.message}"
                )
            q_free = ls.x

        q = _assemble_q(q_free)
        return q

    def internal_forces(self, q):
        # Returns a JAX array — keeps autodiff chain intact for jax.jacobian in the solver.
        # Downstream numpy code accepts JAX arrays via __array__ protocol.
        return self._f_int_jax(jnp.array(q, dtype=float))

    def external_forces(self, q: np.ndarray, q_dot: np.ndarray, t: float) -> np.ndarray:
        """
        External (non-conservative) generalized forces.

        Rail frame: x along rail, y transverse (away from rail), z down.
        Gravity resolved along rail: g·sin(angle); transverse: g·cos(angle).

        Rocket DOF:
          f_ext[s]     = T·cos(θ) - m·g·sin(α) - Σ μ_k|N_k|·sign(ṡ)   [thrust + gravity + friction]
          f_ext[y]     = T·sin(θ) - m·g·cos(α)                          [thrust + gravity transverse]
          f_ext[theta] = 0                                               [no pitching moment]
        Rail DOF (beam modes):
          f_ext[η_i]   = -μ·g·cos(α)·∫φ_i dx                           [gravity on beam]
        Note: shoe forces on the beam are already captured in f_int via Lagrangian coupling.
        """
        s, y, theta = q[0], q[1], q[2]
        eta = q[3:]

        f_ext = np.zeros(3 + self.ramp.n_modes)

        thrust = self.vehicle.motor.thrust if t > 0.0 else 0.0
        m = self.vehicle.mass
        g = 9.81
        alpha = self.ramp.angle

        # Along rail: thrust + gravity - friction (velocity-dependent, non-smooth)
        f_ext[0] = thrust * np.cos(theta) - m * g * np.sin(alpha)
        for shoe in self.vehicle.shoes:
            x_k = s + shoe.dk - shoe.rk * theta
            is_active = x_k <= shoe.x_release
            w_k = self.ramp.displacement(x_k, eta)
            delta_k = w_k - y - shoe.dk * theta - shoe.rk - shoe.l0
            n_k = jnp.where(is_active, shoe.spring_const * delta_k, 0.0)
            f_ext[0] -= shoe.f_coef * abs(n_k) * np.sign(q_dot[0])

        # Transverse: thrust component + gravity
        f_ext[1] = thrust * np.sin(theta) - m * g * np.cos(alpha)

        # Pitch: no moment (thrust through CG)
        f_ext[2] = 0.0

        # Beam modes: distributed gravity loading projected onto mode shapes
        f_ext[3:] = (-self.ramp.beam.mu * g * np.cos(alpha)
                     * np.array(self.ramp.shape_ints))

        return f_ext
