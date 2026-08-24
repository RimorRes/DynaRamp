"""
Reproduction of the section 6.3 validation case of Liu et al. (2024).

Reference
---------
Z. Liu, G. Wang, X. Rui, G. Wu, J. Tang, L. Gu, "Modeling and simulation framework for
missile launch dynamics in a rigid-flexible multibody system with slider-guide
clearance", Nonlinear Dynamics 112:21701-21728 (2024). Section 6.3, Fig. 10 (geometry),
Table 3 (parameters) and Fig. 11 (results).

A missile is driven out of a flexible launch canister held at 60 degrees by a constant
engine thrust, with clearance between its sliders and the two guides machined into the
canister walls. The paper cross-checks its own method against MSC.ADAMS; this script runs
the same scenario through DynaRamp and plots the result on the axes of Fig. 11.

Run:  python examples/liu_6_3_validation.py
"""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

import dynaramp.mstmm as dyn
from dynaramp.contact import CanisterProfile, ContactSolver, OffsetProfile
from dynaramp.mstmm import augmented_modes
from dynaramp.projectile import GuideModalField, Projectile, Slider
from dynaramp.simulation import Gravity, LaunchSimulator, Thrust, G0

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-30s %(message)s")
logging.getLogger("dynaramp.mstmm").setLevel(logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
DIGITIZED = os.path.join(HERE, "data", "liu2024_fig11_digitized.csv")

# --------------------------------------------------------------------------------------
# Geometry and parameters
#
# Table 3 gives the materials, thrust, clearances, contact stiffness and friction
# directly. Fig. 10 gives the lengths and the two cross-sections as drawings; the derived
# quantities below follow from them, and each one is cross-checked against a second,
# independent statement in the paper -- noted where that is the case.
# --------------------------------------------------------------------------------------

LAUNCH_ANGLE_DEG = 60.0          # Table 3
E_MOD = 207e9                    # Table 3, Pa
DENSITY = 7801.0                 # Table 3, kg/m^3
POISSON = 0.29                   # Table 3
THRUST = 8000.0                  # Table 3, N (constant)
CONTACT_STIFFNESS = 1.0e6        # Table 3, N/m -- linear, see note below
FRICTION = 0.2                   # Table 3
RESTITUTION = 0.75               # NOT in Table 3; carried over from Table 1. See notes.

SHEAR_MOD = E_MOD / (2.0 * (1.0 + POISSON))

# --- launch canister (Fig. 10a, 10b) ---
CANISTER_LENGTH = 3.0            # Fig. 10a
CANISTER_OUTER = 0.16            # Fig. 10b
CANISTER_BORE = 0.10             # Fig. 10b
CANISTER_AREA = CANISTER_OUTER ** 2 - CANISTER_BORE ** 2
CANISTER_I = (CANISTER_OUTER ** 4 - CANISTER_BORE ** 4) / 12.0   # square: I_y == I_z

# --- missile (Fig. 10a, 10c) ---
MISSILE_LENGTH = 2.5             # Fig. 10a
MISSILE_SIDE = 0.08              # Fig. 10c
MISSILE_MASS = DENSITY * MISSILE_SIDE ** 2 * MISSILE_LENGTH      # 124.8 kg

# Slider stations along the missile. Fig. 10a dimensions the pair separation as 1.45 m
# from the tail, where the engine thrust is applied; the rear pair therefore sits at the
# tail, which is also O1, the body-frame origin.
SLIDER_STATIONS = (0.0, 1.45)

# Lateral geometry of a slider, from Fig. 10c: the slider projects 0.02 m beyond the
# 0.08 m missile flat, so its tip reaches 0.04 + 0.02 = 0.06 m from the axis, and the
# 0.01 m dimension is the ball diameter. The ball center is one radius inboard of the tip.
SLIDER_BALL_RADIUS = 0.005
SLIDER_LATERAL = MISSILE_SIDE / 2.0 + 0.02 - SLIDER_BALL_RADIUS   # 0.055 m

# Clearances, Table 3. The paper's frame is (+x forward, +y up, +z right), the mirror of
# the NED convention used here, so its c_y is the VERTICAL gap and its c_z the LATERAL
# one. Both are reproduced independently by the Fig. 10b/10c geometry:
#   vertical : half the 0.012 m slot height, less the ball radius  = 0.006 - 0.005 = 0.001
#   lateral  : slot depth 0.05 + 0.012 = 0.062, less center 0.055 and radius 0.005 = 0.002
CLEARANCE_VERTICAL = 0.001       # paper's c_y
CLEARANCE_LATERAL = 0.002        # paper's c_z

# Distance from O1 to the canister muzzle at t = 0. The missile starts fully seated with
# its tail at the fixed end, so the rear pair travels the whole canister. This makes the
# front pair release at x_R = 3.0 - 1.45 = 1.55 m, which the paper's own Fig. 11c
# corroborates: its pitch rate turns over at about 0.24 s, exactly when x_R reaches 1.55.
EXIT_STATION = CANISTER_LENGTH

DT = 5e-5
T_MAX = 0.36
N_MODES = 8
OMEGA_MAX = 2500.0

# Initial condition: everything at zero, which is the paper's own setup.
#
# The paper starts the missile and canister "in a relatively static state" and obtains the
# initial generalized coordinates from Eq. 12. That equation is only the projection of an
# initial *physical* state v(0) onto the modes, p_s(0) = <v(0), M V^s> / M_s -- it does not
# assert that v(0) is the static equilibrium. With the canister undeformed and everything
# at rest, v(0) = 0 and therefore p(0) = 0.
#
# So the canister starts straight while already carrying the missile, and is released at
# t = 0. The resulting transient is not an artifact to be removed here: it is part of what
# Fig. 11 shows, and reproducing the paper means reproducing it. `LaunchSimulator.
# equilibrium_state` would suppress it, which is right for a launch study of one's own --
# see `launch_demo.py` -- and wrong for this comparison.
#
# This also fixes the missile on the guide axis rather than seated on a flank, which is
# the reading that matches Fig. 11b's amplitude.


def build_canister() -> dyn.System:
    """
    The launch canister: a hollow square steel tube, clamped at its base, free at the muzzle.

    Returns
    -------
    dyn.System
        The assembled MSTMM system.
    """
    topo = dyn.TopologyHandler()
    beam = dyn.EulerBernoulliBeam(
        e_id="canister",
        length=CANISTER_LENGTH,
        density=DENSITY,
        youngs_mod=E_MOD,
        shear_mod=SHEAR_MOD,
        area=CANISTER_AREA,
        i_y=CANISTER_I,
        i_z=CANISTER_I,
    )
    topo.add_elements(beam)
    clamped = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    free = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_tip(beam, clamped, input_pos=(0, 0, 0))
    topo.add_root(beam, free, output_pos=(CANISTER_LENGTH, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


def launcher_attitude() -> np.ndarray:
    """
    A_IR for a canister elevated to the predefined launch angle.

    Returns
    -------
    np.ndarray
        The 3x3 rotation from the canister frame K_R to the inertial frame K_I.

    Notes
    -----
    In NED (+x forward, +y right, +z down) an elevation above the horizon points the
    canister axis partly *upward*, i.e. toward negative z. A positive rotation about +y
    produces exactly that, so gravity resolves to a negative axial component and
    decelerates the missile, as it must.
    """
    return Rotation.from_euler("y", LAUNCH_ANGLE_DEG, degrees=True).as_matrix()


def build_missile() -> Projectile:
    """
    The missile: a solid square bar with two slider pairs.

    Returns
    -------
    Projectile
        Inertia and sliders, with the body origin O1 at the rear slider pair.
    """
    m = MISSILE_MASS
    a = MISSILE_SIDE
    i_xx = m * (a ** 2 + a ** 2) / 12.0
    i_yy = i_zz = m * (MISSILE_LENGTH ** 2 + a ** 2) / 12.0

    sliders = [
        Slider((station, sign * SLIDER_LATERAL, 0.0), radius=SLIDER_BALL_RADIUS)
        for station in SLIDER_STATIONS
        for sign in (+1.0, -1.0)
    ]
    return Projectile(
        mass=m,
        inertia_com=np.diag([i_xx, i_yy, i_zz]),
        com_o1=(MISSILE_LENGTH / 2.0, 0.0, 0.0),
        sliders=sliders,
    )


def build_contact(field: GuideModalField, missile: Projectile) -> ContactSolver:
    """
    The two-guide contact model.

    Parameters
    ----------
    field : GuideModalField
        The canister's modal field.
    missile : Projectile
        Supplies the sliders, whose lateral sign selects which guide each one rides.

    Returns
    -------
    ContactSolver
        Configured with one profile per slider.

    Notes
    -----
    Table 3 quotes the contact stiffness as ``1e6 N/m``. Those units are those of a
    *linear* penalty, not of the Hertzian parameter of Eq. B13, which for these materials
    and a 5 mm ball would be about ``1.5e10 N/m^1.5`` -- four orders of magnitude away.
    The exponent is therefore taken as 1, which is the reading consistent with the value
    actually published.

    Table 3 omits the restitution coefficient; the value from Table 1 is used, and it is
    the one genuinely free parameter in this reproduction.
    """
    groove = CanisterProfile(
        clearance_y=CLEARANCE_LATERAL,
        clearance_z=CLEARANCE_VERTICAL,
        stiffness=CONTACT_STIFFNESS,
        restitution=RESTITUTION,
        friction=FRICTION,
        exponent=1.0,
    )
    # One guide per side of the canister. Each slider's own groove datum sits at its
    # nominal lateral station, and the left-hand groove is the mirror of the right.
    profiles = [
        OffsetProfile(
            groove,
            offset=(0.0, float(s.position[1]), 0.0),
            mirror_y=(s.position[1] < 0.0),
        )
        for s in missile.sliders
    ]
    return ContactSolver(
        field, profiles, missile.sliders,
        l_c=EXIT_STATION, station_bracket=0.5,
    )


def load_digitized() -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Read the digitized Fig. 11 curves, if they have been extracted.

    Returns
    -------
    dict
        ``(panel, series) -> (t, value)``, sorted by time. Empty if the file is absent.
    """
    if not os.path.exists(DIGITIZED):
        return {}
    buckets: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    with open(DIGITIZED) as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith("#"))
        for row in rows:
            buckets[(row["panel"], row["series"])].append(
                (float(row["t_s"]), float(row["value"])))
    out = {}
    for key, pts in buckets.items():
        pts.sort()
        arr = np.array(pts, dtype=np.float64)
        out[key] = (arr[:, 0], arr[:, 1])
    return out


def main() -> None:
    """Build the model, run the launch, and plot against the digitized reference."""
    system = build_canister()
    a_ir = launcher_attitude()

    # Repeated eigenfrequencies are guaranteed here: the canister section is square, so
    # every bending mode comes as a y/z pair at one frequency. `augmented_modes` supplies
    # the orthogonalization those pairs need before the modal equations decouple --
    # `natural_modes` alone would return a basis that does not satisfy (3.93).
    modes = augmented_modes(system, N_MODES, omega_max=OMEGA_MAX)
    field = GuideModalField.from_elements(system, ["canister"], modes, a_ir=a_ir)
    print(f"Canister modes retained: {len(modes)}")
    print("  frequencies [rad/s]: "
          + ", ".join(f"{m.frequency:.1f}" for m in modes))
    print(f"  first bending: {modes[0].frequency:.2f} rad/s "
          f"= {modes[0].frequency / (2 * np.pi):.2f} Hz")

    missile = build_missile()
    solver = build_contact(field, missile)
    print(f"\nMissile mass {MISSILE_MASS:.2f} kg, "
          f"sliders at x = {SLIDER_STATIONS} m, y = +/-{SLIDER_LATERAL:.3f} m")

    forces = [Gravity(g=(0.0, 0.0, G0)), Thrust(curve=lambda t: THRUST)]
    sim = LaunchSimulator(field, missile, solver, forces=forces)

    # Seat the missile on the lower flank of its slots, where gravity holds it: the
    # clearance plus the static penetration the contact stiffness allows. Starting it
    # centered would inject a settling transient that is not in the paper's scenario.
    g_axial, _, g_normal = a_ir.T @ np.array([0.0, 0.0, G0])
    print(f"  gravity in canister frame: axial {g_axial:+.3f}, normal {g_normal:+.3f} m/s^2")
    print(f"  expected net axial acceleration: "
          f"{THRUST / MISSILE_MASS + g_axial:.2f} m/s^2")

    # Undeformed canister, missile on the guide axis, everything at rest: v(0) = 0, so
    # Eq. 12 gives p(0) = 0. See the note at the top of the file.
    x0 = np.zeros(6)
    print("  initial state: undeformed canister, missile on the guide axis, at rest "
          "(p(0) = 0 by Eq. 12)")

    result = sim.run(x0, np.zeros(6), dt=DT, t_max=T_MAX)

    print(f"\nSteps: {len(result.t)}   exited: {result.exited}")
    print(f"Exit time: {result.t[-1] * 1e3:.1f} ms   "
          f"axial exit speed: {result.y[-1, 0]:.2f} m/s")
    g, psi, phi = np.degrees(result.attitude[-1])
    wx, wy, wz = np.degrees(result.angular_velocity[-1])
    print(f"Exit attitude [yaw, pitch, roll]: [{g:+.4f}, {psi:+.4f}, {phi:+.4f}] deg")
    print(f"Exit body rates                 : [{wx:+.3f}, {wy:+.3f}, {wz:+.3f}] deg/s")

    _plot(result, sim)


def _pitch_rate_deg(result) -> np.ndarray:
    """
    Pitch rate in the paper's sense, in deg/s.

    Parameters
    ----------
    result : LaunchResult
        The completed run.

    Returns
    -------
    np.ndarray
        ``gamma_dot`` over time.

    Notes
    -----
    The paper's ``gamma`` is the angle its own frame calls pitch. In the NED convention
    used here that is the rotation about +y, which is component 1 of the body rate.
    """
    return np.degrees(result.y[:, 4])


def _paper_y_ddot_l(result, sim) -> np.ndarray:
    """
    The generalized acceleration ``y_ddot_L`` of Fig. 11b, in the paper's own frame.

    Parameters
    ----------
    result : LaunchResult
        The completed run.
    sim : LaunchSimulator
        Used to re-evaluate the right-hand side, which is where the accelerations live.

    Returns
    -------
    np.ndarray
        The paper's vertical generalized acceleration over time [m/s^2].

    Notes
    -----
    This is the one quantity in Fig. 11 where the frame difference actually bites. The
    paper's axes are (+x forward, +y up, +z right), so its ``y_L`` is the *vertical*
    offset of the missile within the canister. DynaRamp is NED (+y right, +z down), so
    the same physical quantity is ``-z_L`` here. Plotting DynaRamp's own ``y_L`` against
    Fig. 11b would compare the lateral channel with the vertical one -- and in this
    perfectly symmetric scenario the lateral channel is identically zero, so the mistake
    produces a flat line rather than an obviously wrong one.

    ``LaunchResult`` stores coordinates and velocities, not accelerations, so the
    quasi-acceleration is recovered by replaying the right-hand side at each recorded
    state. The contact memory is not replayed, which affects only the hysteresis term
    of a face already in contact.
    """
    n = sim.n
    out = np.empty(len(result.t), dtype=np.float64)
    for i, t in enumerate(result.t):
        z = np.concatenate([result.p[i], np.zeros(n), result.x[i], result.y[i]])
        z_dot, _ = sim._rhs(float(t), z, {})
        out[i] = -z_dot[2 * n + 6 + 2]       # -d(z_L_dot)/dt: NED down -> paper's up
    return out


def _plot(result, sim) -> None:
    """
    Reproduce the three panels of Fig. 11, overlaying the digitized reference.

    Parameters
    ----------
    result : LaunchResult
        The completed run.
    sim : LaunchSimulator
        Needed to recover the accelerations of panel (b).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping plots)")
        return

    ref = load_digitized()
    if not ref:
        print(f"\n(no digitized reference at {DIGITIZED} -- plotting DynaRamp only)")

    t = result.t
    # Panels (a) and (c) keep Fig. 11's own limits so they can be laid side by side with
    # it. Panel (b) is left to autoscale: the contact spikes overshoot the published
    # range, and clipping them would quietly understate the disagreement.
    series = [
        ("a", "$x_R$ [m]", result.x[:, 0], (0, 3.2)),
        ("b", r"$\ddot{y}_L$ [m/s$^2$]", _paper_y_ddot_l(result, sim), None),
        ("c", r"$\dot{\gamma}$ [deg/s]", _pitch_rate_deg(result), (-16, 7)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (panel, ylabel, y, ylim) in zip(axes, series):
        for key, color, label, style in (
            (("%s" % panel, "adams"), "#3b6ea5", "MSC.ADAMS (digitized)", "-"),
            (("%s" % panel, "liu"), "#c66a2e", "Liu et al. (digitized)", "--"),
        ):
            if key in ref:
                rt, rv = ref[key]
                ax.plot(rt, rv, style, color=color, lw=1.0, alpha=0.75, label=label)
        ax.plot(t, y, "-", color="#111111", lw=1.4, label="DynaRamp")
        ax.set(xlabel="t [s]", ylabel=ylabel, xlim=(0, 0.36))
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"({panel})", loc="left", fontsize=10)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Liu et al. (2024) section 6.3 — DynaRamp vs the published figure\n"
        "reference curves digitized from Fig. 11; approximate, see data file header",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "liu_6_3_validation.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved comparison to {out}")


if __name__ == "__main__":
    main()
