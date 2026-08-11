"""
DynaRamp end-to-end demo: a sounding rocket launching off a flexible rail.

This wires together all four sections of the framework:
  * section 2 (mstmm)       -- the flexible rail's modal model,
  * section 3 (projectile)  -- the rocket's kinematics and equations of motion in the rail,
  * section 4 (contact)     -- the T-shoe / rail-groove contact with clearance,
  * section 5 (simulation)  -- the coupled solve (Eq. 67) and time integration.

It runs a powered launch until both slider pairs have left the rail, then reports the
initial-disturbance quantities (exit attitude and rates) and plots the histories.

Run:  python examples/launch_demo.py
"""

from __future__ import annotations

import numpy as np

import dynaramp.mstmm as dyn
from dynaramp.materials import STEEL
from dynaramp.projectile import GuideModalField, Projectile, Slider
from dynaramp.contact import RailProfile, ContactSolver, linear_contact_stiffness
from dynaramp.simulation import LaunchSimulator, Gravity, Thrust, G0

import logging

logging.basicConfig(level=logging.DEBUG)

def build_rail() -> dyn.System:
    """A slender steel launch rail, modeled as an Euler-Bernoulli beam clamped at its base
    (x = 0) and free at the muzzle end (x = L)."""
    topo = dyn.TopologyHandler()
    length = 518 * 25.4e-3 # rail length [m]
    width, height = 0.5, 1.0  # cross-section [m]
    beam = dyn.EulerBernoulliBeam(
        e_id="rail",
        length=length,
        density=STEEL.density,
        youngs_mod=STEEL.youngs_modulus,
        shear_mod=STEEL.shear_modulus,
        area=width * height,
        i_y=width * height ** 3 / 12,
        i_z=width ** 3 * height / 12,
    )
    topo.add_elements(beam)
    # clamp the base (x=0): all displacements zero; free muzzle (x=L): all forces zero.
    clamped = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    free = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_tip(beam, clamped, input_pos=(0, 0, 0))
    topo.add_root(beam, free, output_pos=(length, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


def main() -> None:
    # --- section 2: rail modal model ---
    system = build_rail()
    modes = system.natural_modes(7, omega_max=1500)
    field = GuideModalField.from_elements(system, ["rail"], modes)
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
        Gravity(g=(0.0, 0.0, G0)),                        # NED down = +z; gravity seats the shoes on the groove floor
        Thrust(curve=lambda t: 285e3 * min(1.0, t / 0.1)),  # 250 kN motor, 100 ms ramp
    ]
    sim = LaunchSimulator(field, rocket, solver, forces=forces, rayleigh=(2.0, 1e-5))

    # start seated on the groove floor (+z, "down") at the base, at rest.
    x0 = np.array([0.2, 0.0, 5.5e-4, 0.0, 0.0, 0.0])      # [x_R, y_L, z_L, gamma, psi, phi]
    result = sim.run(x0, np.zeros(6), dt=1e-4, t_max=0.7)

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
