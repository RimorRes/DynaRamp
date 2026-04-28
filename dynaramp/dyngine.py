import numpy as np
import jax
import jax.numpy as jnp

from beam import EulerBernoulliBeam
from physics import nnr_solver
from launchrail import LaunchRail
from vehicle import RigidRocket2D, SimpleMotor

import matplotlib.pyplot as plt


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

        From the Lagrangian, shoe spring k with deformation
            delta_k = y + phi(x_k)·eta - dk·theta - rk - l0
            N_k     = spring_const * delta_k
        contributes:
            f_int[y]     += N_k
            f_int[theta] += -dk * N_k          (moment arm, Lagrangian sign)
            f_int[eta_i] += phi_i(x_k) * N_k  (mode-shape coupling)
        Plus beam modal stiffness (outside shoe loop):
            f_int[eta_i] += K_ii * eta_i
        """
        shoes = self.vehicle.shoes
        jax_shapes = self.ramp.jax_modal_shapes
        K_beam = jnp.array(self.ramp.K)
        n_modes = self.ramp.n_modes

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
                # Spring deformation: gap between shoe and natural length
                delta_k = y + w_k - shoe.dk * theta - shoe.rk - shoe.l0
                n_k = jnp.where(is_active, shoe.spring_const * delta_k, 0.0)

                f = f.at[1].add(n_k)
                f = f.at[2].add(-shoe.dk * n_k)
                for i in range(n_modes):
                    f = f.at[3 + i].add(jax_shapes[i](x_k) * n_k)

            # Beam modal stiffness restoring force (must be outside shoe loop)
            f = f.at[3:].add(K_beam @ eta)

            return f

        return jax.jit(f_int)

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

        T = self.vehicle.motor.thrust if self.vehicle.motor is not None else 0.0
        m = self.vehicle.mass
        g = 9.81
        alpha = self.ramp.angle

        # Along rail: thrust + gravity - friction (velocity-dependent, non-smooth)
        f_ext[0] = T * np.cos(theta) - m * g * np.sin(alpha)
        for shoe in self.vehicle.shoes:
            x_k = s + shoe.dk - shoe.rk * theta
            is_active = x_k <= shoe.x_release
            w_k = self.ramp.displacement(x_k, eta)
            delta_k = y + w_k - shoe.dk * theta - shoe.rk - shoe.l0
            n_k = jnp.where(is_active, shoe.spring_const * delta_k, 0.0)
            f_ext[0] -= shoe.f_coef * abs(n_k) * np.sign(q_dot[0])

        # Transverse: thrust component + gravity
        f_ext[1] = T * np.sin(theta) - m * g * np.cos(alpha)

        # Pitch: no moment (thrust through CG)
        f_ext[2] = 0.0

        # Beam modes: distributed gravity loading projected onto mode shapes
        f_ext[3:] = (-self.ramp.beam.mu * g * np.cos(alpha)
                     * np.array(self.ramp.shape_ints))

        return f_ext


if __name__ == "__main__":

    def make_beam():
        b = 0.5/2
        h = 1.0/2
        A = b * h
        rho = 7850
        mu = rho * A    # linear mass density, kg/m
        E = 210e9       # elastic modulus, Pa
        I = b * h ** 3 / 12
        L = 20.0        # length, m
        return EulerBernoulliBeam(mu, E, I, L)

    Beam = make_beam()
    Rail = LaunchRail(Beam, n_modes=4, angle=np.pi / 4)

    # Black Brandt X approximate parameters
    Motor = SimpleMotor(thrust=257e3, isp=280)
    mass = 2600.0
    r = 0.44 / 2
    h_rocket = 14.50
    inertia = 1 / 12 * mass * (3 * r ** 2 + h_rocket ** 2)

    Rocket = RigidRocket2D(mass=mass, inertia=inertia, motor=Motor)
    Rocket.add_shoe(rel_pos=np.array([5.0, r]), friction_coef=0.5, release_point=Rail.beam.L)
    Rocket.add_shoe(rel_pos=np.array([-5.0, r]), friction_coef=0.5, release_point=Rail.beam.L)

    Sys = System(Rail, Rocket)

    # Initial state: CG at s=5, y at shoe equilibrium (delta=0), theta=0
    shoe_l0 = Rocket.shoes[0].l0
    q0 = np.zeros(3 + Rail.n_modes)
    q0[:3] = [5.0, r + shoe_l0, 0.0]   # y = rk + l0 → zero spring deformation
    q_dot0 = np.zeros_like(q0)

    for t_val, vals in enumerate(nnr_solver(
        m=Sys.M,
        c=Sys.C,
        f_int_func=Sys.internal_forces,
        f_ext_func=Sys.external_forces,
        init_state=(q0, q_dot0),
        t_stop=1,
        dt=0.005,
    )):
        q, q_dot, q_ddot = vals
        print(f"t={t_val * 0.005:.3f}  s={q[0]:.4f}  y={q[1]:.6f}  theta={q[2]:.6f}")

    print(f"Final state: theta={np.degrees(q[2]):.2f}°  theta_dot={np.degrees(q_dot[2]):.3f}°")

    # Plotting final state
    xs = np.linspace(0, Beam.L, 100)
    ws = np.array([Rail.displacement(x, q[3:]) for x in xs])
    plt.plot(xs, ws)
    plt.xlabel("x [m]")
    plt.ylabel("w [m]")
    plt.title("Final state")
    plt.show()
