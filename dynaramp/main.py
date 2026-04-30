import numpy as np

from beam import EulerBernoulliBeam
from physics import nnr_solver
from launchrail import LaunchRail
from vehicle import RigidRocket2D, SimpleMotor
from dyngine import System

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
mpl.use("TkAgg")



b = 0.5
h = 1.0
area = b * h
rho = 7850
mu = rho * area  # linear mass density, kg/m
e_mod = 210e9  # elastic modulus, Pa
sma = b * h ** 3 / 12  # second moment of area, m^4
length = 20.0  # length, m
Beam = EulerBernoulliBeam(mu, e_mod, sma, length)
Rail = LaunchRail(Beam, n_modes=5, angle=np.pi / 4)

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
q0[:3] = [5.0, -(r + shoe_l0), 0.0]  # y = rk + l0 → zero spring deformation
q_dot0 = np.zeros_like(q0)

t_arr = []
q_arr = []
q_dot_arr = []

t_step = 0.005
for t_val, vals in enumerate(nnr_solver(
        m=Sys.M,
        c=Sys.C,
        f_int_func=Sys.internal_forces,
        f_ext_func=Sys.external_forces,
        init_state=(q0, q_dot0),
        t_stop=1.0,
        dt=t_step,
)):
    q, q_dot, q_ddot = vals

    active_shoe_count = sum(q[0] + shoe.dk - shoe.rk * q[2] <= shoe.x_release for shoe in Rocket.shoes)

    t_arr.append(t_val * t_step)
    q_arr.append(q)
    q_dot_arr.append(q_dot)

    print(f"t={t_val * t_step:.3f}  s={q[0]:.4f}  y={q[1]:.6f}  theta={q[2]:.6f}  shoes={active_shoe_count}")

    if active_shoe_count == 0:
        print("Free flight!")
        break

print(f"Final state: theta={np.degrees(q[2]):.2f}°  theta_dot={np.degrees(q_dot[2]):.3f}°")

# Convert to numpy arrays
t_arr = np.array(t_arr)
q_arr = np.array(q_arr)
q_dot_arr = np.array(q_dot_arr)

# Plotting and Animating beam
fig, ax = plt.subplots()

etas = q_dot_arr[:, 3:]
x = np.linspace(0, Rail.beam.L, 100)  # abscissa along the beam
w = np.array([Rail.displacement(s, etas[0]) for s in x])

line = ax.plot(x, w, label=f"Beam @ t={t_arr[0]:.3f}")[0]


ax.set(xlim=(0, Rail.beam.L), ylim=(-1, 1))
ax.set(xlabel="x (m)", ylabel="w (m)", title="Beam Displacement")
ax.grid()
leg = ax.legend()

def update(frame):
    w = np.array([Rail.displacement(s, etas[frame]) for s in x])
    line.set_ydata(w)
    lab = f"Beam @ t={t_arr[frame]:.3f}"
    leg.get_texts()[0].set_text(lab)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t_arr), interval=5)
plt.show()
