# Getting started with DynaRamp

DynaRamp simulates what happens to a projectile in the few tenths of a second between
motor ignition and the moment it leaves a launcher — while it is still touching a
structure that is itself bending under the load. The output is the *initial
disturbance*: the attitude and body rates the vehicle carries away from the rail, which
set the dispersion everything downstream has to correct for.

The package is organized around the four sections of the underlying method, and a
simulation is built by walking through them in order:

| Stage          | Package               | Question it answers                                     |
|----------------|-----------------------|---------------------------------------------------------|
| 1. Structure   | `dynaramp.mstmm`      | How does the launcher vibrate?                          |
| 2. Modal field | `dynaramp.projectile` | Where is the rail at station *x*, and how is it tilted? |
| 3. Projectile  | `dynaramp.projectile` | How does the vehicle move inside the moving rail?       |
| 4. Contact     | `dynaramp.contact`    | What force passes through the slider–groove clearance?  |
| 5. Coupling    | `dynaramp.simulation` | Solve both together, and march in time.                 |

---

## The shortest complete simulation

Everything below is elaboration. This is the whole pipeline:

```python
import numpy as np
import dynaramp.mstmm as dyn
from dynaramp.materials import STEEL
from dynaramp.projectile import GuideModalField, Projectile, Slider
from dynaramp.contact import RailProfile, ContactSolver, linear_contact_stiffness
from dynaramp.simulation import LaunchSimulator, Gravity, Thrust, G0

# 1. structure -------------------------------------------------------------
length = 13.157
topo = dyn.TopologyHandler()
beam = dyn.EulerBernoulliBeam(
    e_id="rail", length=length, density=STEEL.density,
    youngs_mod=STEEL.youngs_modulus, shear_mod=STEEL.shear_modulus,
    area=0.5, i_y=1.0 / 24, i_z=1.0 / 96,
)
topo.add_elements(beam)
topo.add_tip(beam, [0, 0, 0, 0, 0, 0, None, None, None, None, None, None], input_pos=(0, 0, 0))
topo.add_root(beam, [None, None, None, None, None, None, 0, 0, 0, 0, 0, 0], output_pos=(length, 0, 0))
topo.make_tree()
system = dyn.System(topo)

# 2. modal analysis + modal field -----------------------------------------
modes = system.natural_modes(7, omega_max=1500)
field = GuideModalField.from_elements(system, ["rail"], modes)

# 3. projectile ------------------------------------------------------------
rocket = Projectile(
    mass=2468.1,
    inertia_com=np.diag([304.0, 8100.0, 8100.0]),
    com_o1=(5.58, 0.0, 0.0),
    sliders=[Slider((x, 0.0, 0.0), radius=0.0) for x in (0.48, 3.70, 5.11, 8.16)],
)

# 4. contact ---------------------------------------------------------------
k = linear_contact_stiffness(area=2e-4, material_a=STEEL, material_b=STEEL,
                             compliance_length=0.4)
profile = RailProfile(lateral_clearance=5e-4, bottom_clearance=5e-4,
                      top_clearance=5e-4, stiffness=k, restitution=0.4, friction=0.1)
solver = ContactSolver(field, profile, rocket.sliders, l_c=length)

# 5. loads, integration ----------------------------------------------------
forces = [Gravity(g=(0.0, 0.0, G0)),
          Thrust(curve=lambda t: 285e3 * min(1.0, t / 0.1))]
sim = LaunchSimulator(field, rocket, solver, forces=forces, rayleigh=(2.0, 1e-5))

x0 = np.array([0.2, 0.0, 5.5e-4, 0.0, 0.0, 0.0])
result = sim.run(x0, np.zeros(6), dt=1e-4, t_max=0.7)

print(np.degrees(result.attitude[-1]), np.degrees(result.angular_velocity[-1]))
```

A runnable version with plotting is in `examples/launch_demo.py`.

---

## Frames and sign conventions

Get this wrong, and the simulation still runs... but will return plausible nonsense.

DynaRamp uses **NED**: `+x` forward (along the rail, toward the muzzle), `+y` right,
`+z` **down**. Gravity is therefore `(0, 0, +9.81)`, and it seats a shoe on the groove
*floor*, which is the `+z` face.

| Frame  | Meaning                                                                           |
|--------|-----------------------------------------------------------------------------------|
| `K_I`  | Inertial. Everything the solver reports is projected here.                        |
| `K_R`  | Guide reference. Related to `K_I` by `A_IR`, the launcher's attitude.             |
| `K_Pi` | A guide cross-section, tilted with the local bending. Contact lives here.         |
| `K_B`  | Projectile body. Origin at **O1**, the rear slider position; `x` toward the nose. |

Attitude uses intrinsic z-y-x Euler angles `(gamma, psi, phi)` = (yaw, pitch, roll).

> **Caution:** If you are transcribing a test case using another coordinate system,
> please note that you will have to convert to the right-handed NED frame
> described above, which is the same as the body frame of a conventional aircraft.
> Notable traps will be pitch/yaw/roll angles, geometry, direction of forces etc...

---

## Stage 1 — The structural model

Three objects, always in this order: elements, then a topology wiring them, then a
`System` that solves it.

```python
topo = dyn.TopologyHandler()

topo.add_elements(beam)              # register elements
...                                  # recreate the topology by connecting them
topo.connect_elements(...)
topo.cut_connection(...)             # optional hinge-cutting, for branches or loops
...
topo.add_tip(beam, bc, input_pos=...)    # upstream boundary
topo.add_root(beam, bc, output_pos=...)  # downstream boundary

topo.make_tree()                     # reduce and validate <-- mandatory
system = dyn.System(topo)
```

**Elements** available in `dynaramp.mstmm`:

| Element               | Use                                                        |
|-----------------------|------------------------------------------------------------|
| `EulerBernoulliBeam`  | Slender flexible member. The rail itself.                  |
| `RigidBody`           | Massive stiff component; any number of ports.              |
| `LumpedMass`          | Point mass, no rotational inertia.                         |
| `SpatialElasticHinge` | Six-DOF spring connection (three linear, three torsional). |
| `JunctionNode`        | Massless branch point.                                     |

Every element takes an optional `orientation`, a 3×3 direction-cosine matrix mapping its
local frame to `K_R`. Leave it out for anything axis-aligned — the code detects the
identity once and skips the change of frame entirely.

**Connecting** elements is `topo.connect_elements(src, dst, src_pos, dst_pos)`, where
each position is a port location *in that element's own local frame*. The call creates ports
implicitly. The first input port an element receives becomes its **main
input**, the port its state vector propagates from — worth knowing because mode shapes
and modal masses are all anchored to it.

**Boundary conditions** are 12-component vectors, one entry per state component
`[r(3); theta(3); m(3); q(3)]`, using `None` for "unknown, solve for it" and a number for
"prescribed":

```python
clamped = [0, 0, 0, 0, 0, 0, None, None, None, None, None, None]  # no motion
free    = [None, None, None, None, None, None, 0, 0, 0, 0, 0, 0]  # no load
```

The count has to work out: `make_tree()` will build, but `overall_transfer_mat` raises
`ValueError` if the knowns and unknowns leave a non-square system — that error means
you have over- or under-constrained the structure, not that the code failed.

**`make_tree()` is not optional.** The transfer matrix method needs a tree; a topology
with loops or branches is reduced to one by cutting the extra connections and recording
what was cut. Everything downstream raises `RuntimeError` until it has been called.

---

## Stage 2 — Modal analysis

```python
modes = system.natural_modes(n_modes=7, omega_max=1500)
```

This sweeps `sigma_min(U_all(omega))` and refines each dip. Two parameters actually
change the answer:

- **`omega_max`** — the ceiling of the search. Set it from the physics: modes well above
  the excitation bandwidth contribute nothing but cost.
- **`search_res`** (default 10000) — sweep resolution. Too coarse and a narrow dip will be
  stepped over; the associated mode will be silently missing from your basis. If a mode count comes
  back lower than expected, raise this before anything else.

`n_modes` counts **distinct eigenfrequencies**, not returned modes. A symmetric structure
has repeated roots — equal stiffness in `y` and `z` gives a two-dimensional eigenspace at
one frequency — and every basis vector of it is a physically distinct mode, so
`len(modes)` can exceed `n_modes`. That is correct behavior, not a bug.

Zero frequency is excluded from the sweep by design: a root there is a rigid-body
freedom, not a vibration mode. The element transfer matrices themselves are perfectly
well-defined at `omega = 0`, where they reduce to their static form, so a structure with
a genuine rigid-body mode can ask for it directly with `system.eigenvectors_at(0.0)`.

For a *forced-response* study rather than a launch simulation, use
`augmented_modes(system, n)` instead, which adds the orthogonalization step that
repeated eigenfrequencies need before the modal equations decouple.

---

## Stage 3 — The modal field

The seam between the structure and the projectile. It relies on modal superposition
to determine where the **rail**, and how it is tilted, at station (i.e., position along the rail) `x`.

```python
field = GuideModalField.from_elements(system, ["rail"], modes, a_ir=None)
```

- The element list is **ordered from the reference end**, and the elements must be
  collinear. Use the `GuideSegment` form when a segment's length is not readable from
  the element.
- **`a_ir`** is the launcher's attitude — elevation and azimuth — as a 3×3 matrix.
  Default identity means upright and aligned with `K_I`.
- **`step`** is the finite-difference step for the spatial derivatives. Leave it `None`.
  The default selects `optimal_step()`, which balances truncation against round-off; a
  hand-picked value may be *less* accurate, as the second-derivative stencil divides by `h²`.

`field.n_modes`, `field.total_length` and `field.evaluate(x)` are the things you will
interact with. Note the field also carries the `system` and `modes`, which is why later stages
only need the field.

---

## Stage 4 — Projectile and contact

The **projectile** is inertia plus a list of sliders:

```python
rocket = Projectile(mass=..., inertia_com=..., com_o1=..., sliders=[...])
```

`inertia_com` is about the center of mass; `com_o1` locates that center relative to
**O1**, the rear slider, which is the body-frame origin. Shifting to O1 happens
internally.

A `Slider` is a position in `K_B` and a radius. Use `radius=0.0` for a flat shoe — its
dimensions belong in the profile's clearances instead.

The **profile** is the cross-section the sliders ride in:

| Profile                 | Geometry                                                         | Exponent         |
|-------------------------|------------------------------------------------------------------|------------------|
| `RailProfile`           | Rectangular groove, flat T-shoe: two side walls, floor, top lip  | `1.0`, conformal |
| `CanisterProfile`       | Slot in a canister wall, spherical slider: two flanks + deep end | `1.5`, Hertzian  |
| `StationVaryingProfile` | Delegates to a different profile per station                     | —                |

Match the **stiffness helper to the exponent**: `linear_contact_stiffness` with `n = 1`,
`hertz_stiffness` with `n = 1.5`.

> **Note: the stiffness is the parameter that will cost you time.** A true steel-on-steel
> contact stiffness is enormous, and with an explicit fixed-step integrator it forces a
> step small enough to make the run impractical. Choose the softest penalty for which
> peak penetration stays a small fraction of the clearance and check that it did.

The **solver** ties them together:

```python
solver = ContactSolver(field, profile, rocket.sliders, l_c=length)
```

`l_c` is the exit station — where a slider leaves the guide. A **scalar** gives every
slider the same exit, so the front shoe releases first and then the rear: sequential
detachment, and the dominant source of the launch disturbance. A **sequence**, one entry
per slider, lets them release **independently** (simultaneously, for example), 
as a groove that widens toward the muzzle would do.
> **Note:** Setting the exit station `l_c` changes the character of the result more than almost
> anything else in the model.

---

## Stage 5 — Loads and integration

```python
forces = [Gravity(g=(0.0, 0.0, G0)),
          Thrust(curve=lambda t: 285e3 * min(1.0, t / 0.1),
                 application_point=(0, 0, 0), axis=(1, 0, 0))]
sim = LaunchSimulator(field, rocket, solver, forces=forces, rayleigh=(2.0, 1e-5))
result = sim.run(x0, y0, dt=1e-4, t_max=0.7)
```

`rayleigh` offers Rayleigh damping using `(alpha, beta)` in `C = alpha M + beta K`, 
giving each mode `c_p = alpha + beta omega_p²`.

The initial state is two 6-vectors:

- `x0 = [x_R, y_L, z_L, gamma, psi, phi]` — axial station, lateral and vertical offset,
  then yaw/pitch/roll.
- `y0 = [x_R_dot, y_L_dot, z_L_dot, w1, w2, w3]` — the quasi-velocity, **not** `x0`'s
  derivative. They are related by `y = H(gamma, psi) x_dot`; if you have rates, use
  `ProjectileState.from_config_rates(x, x_dot).y`.

Start the projectile *seated*, not floating: set `z_L` just past `bottom_clearance` so
the shoes rest on the groove floor as they would under gravity. Starting it centred in
the clearance produces a spurious settling transient at `t = 0`.

`dt` must resolve the contact stiffness, which is the stiffest thing in the system. If
the contact force history shows growing oscillation, the step is too large — halve it
and confirm the answer stops moving.

`run` stops early once every slider has left the guide; `result.exited` tells you whether
it did, or whether it simply hit `t_max`. **A run that ends with `exited=False` has not
finished the launch**, and its final attitude is not the exit attitude.

### Reading the result

```python
result.t                  # (steps,) time
result.x                  # (steps, 6) configuration
result.y                  # (steps, 6) quasi-velocity
result.p                  # (steps, n) rail modal coordinates
result.attitude           # (steps, 3) == x[:, 3:6], degrees via np.degrees
result.angular_velocity   # (steps, 3) == y[:, 3:6]
result.contact_force      # (steps, 3) resultant on the projectile at O1
result.contact_moment     # (steps, 3)
```

The two numbers the whole exercise exists to produce are the last rows of `attitude` and
`angular_velocity`.

---

## Adding your own pieces

The extension points all follow the same shape: subclass, implement one method.

**A new element** — subclass `DiscreteElement` (lumped mass) or `ContinuousElement`
(distributed mass), implement `_u_local(input_pos, output_pos, omega)` returning the
12×12 transfer matrix *in the element's local frame*, plus the `_m_param_mat` /
`_m_bar_param_mat` property. The base class handles orientation and the modal inner
product. Krylov-Duncan functions and other helpers are available in
`dynaramp.mstmm.krylov` for the exact solutions of continuous elastic systems.

**A new guide cross-section** — subclass `UniformProfile` and implement
`contacts(r_vi, radius, station)`, returning one `SurfaceContact` per penetrated face via
`self._surface(...)`. Only the geometry is yours; the contact-law parameters come from
the base.

**A new load** — any callable `(kin, projectile, t) -> (force, moment)` satisfies the
`ExternalForce` protocol. Aerodynamic drag, a lanyard, a blast overpressure:

```python
@dataclass
class Drag:
    cd_a: float
    rho: float = 1.225
    def __call__(self, kin, projectile, t):
        v = kin.r_dot_o1
        speed = np.linalg.norm(v)
        if speed < 1e-9:
            return np.zeros(3), np.zeros(3)
        force = -0.5 * self.rho * self.cd_a * speed * v
        return force, np.zeros(3)
```

---

## Troubleshooting

| Symptom                                              | Cause                                                                               |
|------------------------------------------------------|-------------------------------------------------------------------------------------|
| `RuntimeError: tree must be generated`               | `make_tree()` not called.                                                           |
| `ValueError: under-constrained or over-constrained`  | Boundary `None` count does not match the unknowns.                                  |
| Fewer modes than expected                            | `search_res` too coarse, or `omega_max` too low.                                    |
| `Propagated state doesn't match root boundary state` | Usually a genuinely inconsistent topology or boundary set.                          |
| `Non-positive modal mass`                            | The mode set is not a valid basis — often a spurious mode from too loose an `rtol`. |
| Contact force oscillates and grows                   | `dt` too large for the contact stiffness, or the stiffness is unphysically high.    |
| `exited=False`                                       | The projectile never left the rail: `t_max` too short, or thrust too low.           |
| Result changes when `step` changes                   | Finite-difference step off its optimum; leave `step=None`.                          |

Logging is per-module under the `dynaramp` namespace, so
`logging.getLogger("dynaramp.simulation").setLevel(logging.INFO)` gives run progress
without the modal-analysis detail.
