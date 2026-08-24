# DynaRamp

Dynamic response solver for rail-launched vehicles.

DynaRamp simulates the coupled dynamics of a projectile leaving a flexible launcher: the
launcher bends under the load, the bending throws the projectile around, and the
projectile rattles inside the clearance of its slider–guide interface. The quantity of
interest is the **initial disturbance** — the attitude and body rates the vehicle carries
away from the rail.

The model is a rigid–flexible multibody system solved by the **Multibody System Transfer
Matrix Method** (MSTMM, after Rui), coupled to a penalty contact model with clearance.

## Installation

```bash
pip install -e .
```

Requires Python 3.12+.

## Quick start

```python
import numpy as np
import dynaramp.mstmm as dyn
from dynaramp.materials import STEEL
from dynaramp.projectile import GuideModalField, Projectile, Slider
from dynaramp.contact import RailProfile, ContactSolver, linear_contact_stiffness
from dynaramp.simulation import LaunchSimulator, Gravity, Thrust, G0

# structure -> modes -> modal field -> projectile -> contact -> integrate
```

The full, runnable pipeline is in [`examples/launch_demo.py`](examples/launch_demo.py).

**→ Start with the [Getting Started guide](docs/getting-started.md).** It walks the five
stages in order, states what each one consumes and returns, and flags the handful of
parameters that actually change the answer.

## Package layout

| Package | Contents |
|---|---|
| `dynaramp.common` | Vector/frame maths, shared modal helpers, type aliases |
| `dynaramp.materials` | Isotropic linear-elastic materials and presets |
| `dynaramp.mstmm` | Elements, topology, the transfer-matrix eigenproblem, forced response |
| `dynaramp.projectile` | Guide modal field, projectile kinematics and equations of motion |
| `dynaramp.contact` | Guide cross-sections, slider contact detection, contact solver |
| `dynaramp.simulation` | Coupled assembly, external loads, time integration |

## Conventions

**NED**: `+x` forward, `+y` right, `+z` down. Gravity is `(0, 0, +9.81)`. Attitude uses
intrinsic z-y-x Euler angles `(gamma, psi, phi)` = (yaw, pitch, roll). Frames are `K_I`
(inertial), `K_R` (guide reference), `K_Pi` (guide cross-section) and `K_B` (projectile
body, origin at the rear slider O1).

If you are transcribing values from the source paper, its frame is
(+x forward, +y up, +z right) — see the guide's *Frames and sign conventions* section
before setting canister clearances.

## References

- Rui, X. et al., *Transfer Matrix Method for Multibody Systems: Theory and Applications*
- *Modeling and simulation framework for missile launch dynamics in a rigid–flexible
  multibody system with slider–guide clearance*
