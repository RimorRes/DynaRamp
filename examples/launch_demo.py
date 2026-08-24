"""
DynaRamp end-to-end demo: a sounding rocket launching off a flexible rail.

This wires together all four sections of the framework:
  * section 2 (mstmm)       -- the flexible rail's modal model,
  * section 3 (projectile)  -- the rocket's kinematics and equations of motion in the rail,
  * section 4 (contact)     -- the T-shoe / rail-groove contact with clearance,
  * section 5 (simulation)  -- the coupled solve (Eq. 67) and time integration.

It runs a powered launch off a ramp elevated 45 degrees until both slider pairs have left
the rail, then reports the initial-disturbance quantities (exit attitude and rates) and
plots the histories.

The run starts from the system's static equilibrium rather than from a straight rail,
so the record is free of the release transient a straight initial condition injects.

Run:  python examples/launch_demo.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import dynaramp.mstmm as dyn
from dynaramp.materials import STEEL
from dynaramp.projectile import GuideModalField, Projectile, Slider
from dynaramp.contact import RailProfile, ContactSolver, linear_contact_stiffness
from dynaramp.simulation import LaunchSimulator, Gravity, Thrust, G0

import logging

# INFO gives the mode summary and run progress. Drop to DEBUG for per-frequency
# transfer-matrix detail, which is verbose enough to dominate the runtime.
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-30s %(message)s")

RAIL_LENGTH = 518 * 25.4e-3        # rail length [m]
RAIL_WIDTH, RAIL_HEIGHT = 0.5, 1.0  # cross-section [m]
RAIL_IY = RAIL_WIDTH * RAIL_HEIGHT ** 3 / 12
RAIL_IZ = RAIL_WIDTH ** 3 * RAIL_HEIGHT / 12

# Elevation of the ramp above the horizon [deg].
RAMP_ANGLE_DEG = 45.0


def ramp_attitude() -> np.ndarray:
    """
    A_IR for the ramp elevated to :data:`RAMP_ANGLE_DEG`.

    Returns
    -------
    np.ndarray
        The 3x3 rotation from the rail frame K_R to the inertial frame K_I.

    Notes
    -----
    In NED (+x forward, +y right, +z down) an elevation above the horizon points the
    rail axis partly *upward*, i.e. toward negative z, and a positive rotation about +y
    does exactly that. The check that it is the right sign and not its mirror: gravity
    must resolve to a *negative* axial component, decelerating the rocket as it climbs.
    """
    return Rotation.from_euler("y", RAMP_ANGLE_DEG, degrees=True).as_matrix()


def build_rail() -> dyn.System:
    """
    A slender steel launch rail: an Euler-Bernoulli beam clamped at its base (x = 0)
    and free at the muzzle end (x = L).

    Returns
    -------
    dyn.System
        The assembled system.
    """
    topo = dyn.TopologyHandler()
    beam = dyn.EulerBernoulliBeam(
        e_id="rail",
        length=RAIL_LENGTH,
        density=STEEL.density,
        youngs_mod=STEEL.youngs_modulus,
        shear_mod=STEEL.shear_modulus,
        area=RAIL_WIDTH * RAIL_HEIGHT,
        i_y=RAIL_IY,
        i_z=RAIL_IZ,
    )
    topo.add_elements(beam)
    # clamp the base (x=0): all displacements zero; free muzzle (x=L): all forces zero.
    clamped = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    free = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_tip(beam, clamped, input_pos=(0, 0, 0))
    topo.add_root(beam, free, output_pos=(RAIL_LENGTH, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


def main() -> None:
    # --- section 2: rail modal model ---
    system = build_rail()
    modes = system.natural_modes(15, omega_max=2000)
    a_ir = ramp_attitude()
    field = GuideModalField.from_elements(system, ["rail"], modes, a_ir=a_ir)
    g_axial, _, g_normal = a_ir.T @ np.array([0.0, 0.0, G0])
    print(f"Ramp elevation: {RAMP_ANGLE_DEG:.1f} deg  "
          f"(gravity in rail frame: axial {g_axial:+.3f}, normal {g_normal:+.3f} m/s^2)")
    print(f"Rail modes retained: {len(modes)} "
          f"(frequencies: {', '.join(f'{m.frequency:.1f}' for m in modes)} rad/s)")

    # --- the projectile: a ~Pathfinder sounding rocket, rear + front slider pairs ---
    mass = 2468.1
    r = 22 * 25.4e-3
    h_rocket = 439 * 25.4e-3
    inertia_xx = 1 / 2 * mass * r ** 2
    inertia_yy = inertia_zz = 1 / 12 * mass * (3 * r ** 2 + h_rocket ** 2)
    lug1 = (679.40 - 660.44) * 25.4e-3
    lug2 = (679.40 - 533.94) * 25.4e-3
    lug3 = (679.40 - 478.42) * 25.4e-3
    lug4 = (679.40 - 358.16) * 25.4e-3
    rocket = Projectile(
        mass=mass,
        inertia_com=np.diag([inertia_xx, inertia_yy, inertia_zz]),   # roll light, pitch/yaw heavy
        com_o1=(h_rocket/2, 0.0, 0.0),                    # COM roughly halfway up wrt the rear slider (O1)
        sliders=[
            Slider((lug1, 0.0, 0.0), radius=0.0),  # rear-most shoe (at O1)
            Slider((lug2, 0.0, 0.0), radius=0.0),
            Slider((lug3, 0.0, 0.0), radius=0.0),
            Slider((lug4, 0.0, 0.0), radius=0.0)  # front-most shoe
        ],
    )

    # --- section 4: the rail groove contact ---
    # A tuned linear (flat-shoe) contact. NOTE: a real steel-on-steel stiffness is far
    # higher and would make the explicit RK4 step stiff; this softer penalty keeps the demo
    # fast, and is exactly the case for the stiff-integrator upgrade noted in the spec.
    k = linear_contact_stiffness(area=2e-4, material_a=STEEL, material_b=STEEL, compliance_length=0.4)
    profile = RailProfile(
        lateral_clearance=5e-4, bottom_clearance=5e-4, top_clearance=5e-4,
        stiffness=k, restitution=0.4, friction=0.1,
    )
    # single rail exit at 2.5 m: the front shoe (station x_R+1.5) leaves first, then the rear
    # -- the sequential detachment that produces the launch disturbance.
    solver = ContactSolver(field, profile, rocket.sliders, l_c=518 * 25.4e-3)

    # --- section 5: loads, simulator, run ---
    forces = [
        # NED down = +z. On an elevated ramp gravity splits: the component along the rail
        # decelerates the rocket, the one across it seats the shoes on the groove floor.
        Gravity(g=(0.0, 0.0, G0)),
        Thrust(curve=lambda t: 285e3 * min(1.0, t / 0.1)),  # 250 kN motor, 100 ms ramp
    ]
    sim = LaunchSimulator(field, rocket, solver, forces=forces, rayleigh=(2.0, 1e-5))

    # Start from static equilibrium. Left to itself the rail would begin straight while
    # already carrying the rocket, and releasing it at t = 0 would ring it at its own
    # natural frequencies -- a transient that is an artifact of the initial condition and
    # that lands squarely on the early record. Solving for the resting state first (the
    # initialization of Eq. 12) removes it: the rail starts already sagged under the
    # rocket, and the shoes start already seated in the groove.
    x0, p0 = sim.equilibrium_state(x_r=0.2)
    print(f"Equilibrium start: z_L = {x0[2] * 1e3:+.4f} mm, "
          f"pitch = {np.degrees(x0[4]):+.5f} deg, "
          f"rail tip sag = {np.linalg.norm(field.shape_at(RAIL_LENGTH).T @ p0) * 1e3:.4f} mm")
    result = sim.run(x0, np.zeros(6), dt=1e-4, t_max=0.7, p0=p0)

    # --- report the initial disturbance ---
    print(f"\nSteps: {len(result.t)}   exited rail: {result.exited}")
    if result.exited:
        print(f"Exit time: {float(result.t[-1] * 1e3):.1f} ms   "
              f"axial exit speed: {result.y[-1, 0]:.1f} m/s")
    g, psi, phi = np.degrees(result.attitude[-1])
    wx, wy, wz = np.degrees(result.angular_velocity[-1])
    # NED z-y-x angles: gamma about +z = yaw, psi about +y = pitch, phi about +x = roll.
    print(f"Exit attitude  [yaw, pitch, roll]: [{g:+.3f}, {psi:+.3f}, {phi:+.3f}] deg")
    print(f"Exit body rates              : [{wx:+.2f}, {wy:+.2f}, {wz:+.2f}] deg/s")
    print(f"Peak contact force on rocket : {float(np.max(np.linalg.norm(result.contact_force, axis=1))):.0f} N")

    _plot(result)


def _plot(result) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping plots)")
        return

    t_ms = result.t * 1e3
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    ax[0, 0].plot(t_ms, result.x[:, 0])
    ax[0, 0].set(title="Axial position along rail", xlabel="t [ms]", ylabel="x_R [m]")

    ax[0, 1].plot(t_ms, np.degrees(result.attitude))
    ax[0, 1].legend(["yaw γ", "pitch ψ", "roll φ"])
    ax[0, 1].set(title="Attitude", xlabel="t [ms]", ylabel="[deg]")

    ax[1, 0].plot(t_ms, np.degrees(result.angular_velocity))
    ax[1, 0].legend(["ω_x", "ω_y", "ω_z"])
    ax[1, 0].set(title="Body angular velocity", xlabel="t [ms]", ylabel="[deg/s]")

    ax[1, 1].plot(t_ms, np.linalg.norm(result.contact_force, axis=1))
    ax[1, 1].set(title="Contact force on rocket", xlabel="t [ms]", ylabel="|q| [N]")

    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.suptitle("DynaRamp — sounding rocket rail launch")
    fig.tight_layout()
    out = "launch_demo_output.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved plots to {out}")


if __name__ == "__main__":
    main()
