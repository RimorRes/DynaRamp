import numpy as np

from beam import EulerBernoulliBeam
from geometry import Basis
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

World_ref = Basis()
Rocket = RigidRocket2D(parent_basis=World_ref, motor=Motor, mass=mass, inertia=inertia)
Rocket.add_shoe(rel_pos=np.array([5.0, r]), friction_coef=0.5, release_point=Rail.beam.L)
Rocket.add_shoe(rel_pos=np.array([-5.0, r]), friction_coef=0.5, release_point=Rail.beam.L)

Sys = System(Rail, Rocket)

# Compute static equilibrium before starting dynamic simulation.
# We fix the CG position s=5.0 and solve for y, theta and modal coords so
# the elastic internal forces balance the external loads (gravity, thrust).
shoe_l0 = Rocket.shoes[0].l0
q0_guess = np.zeros(2 + Rail.n_modes)
q0_guess[:2] = [-(r + shoe_l0), 0.0]

# Solve for static equilibrium with s fixed at 5.0
q0 = Sys.compute_static_equilibrium(s=5.0, q_free0=q0_guess)
q_dot0 = np.zeros_like(q0)
w0 = Rail.displacement(Rail.beam.L, q0[3:])

print("Parameters:")
print('='*40)
print(f"Rail length:           {length:.2f} m")
print(f"Rail width:             {b:.2f} m^2")
print(f"Rail height:            {h:.2f} m^2")
print(f"Rail elastic modulus:   {e_mod/1e9:.2f} GPa")
print(f"Rail density:          {rho:.2f} kg/m^3")
print("-"*40)
print(f"Rail 2nd moment of area: {sma:.4f} m^4")
print(f"Rail linear mass:       {mu:.2f} kg/m^2")
print(f"Rough cantilever sag estimate:  {rho*area*9.81*np.cos(Rail.angle)*length**4/(8*e_mod*sma)*1000:.2f} mm")
print("-"*40)
print(f"Rocket mass:            {mass:.2f} kg")
print("-"*40)
for shoe in Rocket.shoes:
    print(f"Shoe at dk={shoe.dk:.2f} m:  release point={shoe.x_release:.2f} m")
print('='*40)
print("Starting static equilibrium")
print(f"equili.  s={q0[0]:.4f}  y={q0[1]:.6f}  theta={q0[2]:.6f}  w(L)={w0*1000:.2f} mm")

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
        conv_err=1e-5,
)):
    q, q_dot, q_ddot = vals
    current_time = (t_val + 1) * t_step

    active_shoe_count = sum(q[0] + shoe.dk - shoe.rk * q[2] <= shoe.x_release for shoe in Rocket.shoes)

    t_arr.append(current_time)
    q_arr.append(q)
    q_dot_arr.append(q_dot)

    print(f"t={current_time:.3f}  s={q[0]:.4f}  y={q[1]:.6f}  theta={q[2]:.6f}  shoes={active_shoe_count}")

    if active_shoe_count == 0:
        print("Free flight!")
        break

print(f"Final state: theta={np.degrees(q_arr[-1][2]):.2f}°  theta_dot={np.degrees(q_dot_arr[-1][2]):.3f}°/s")

# Convert to numpy arrays
t_arr = np.array(t_arr)
q_arr = np.array(q_arr)
q_dot_arr = np.array(q_dot_arr)

# Plotting and Animating beam
fig, ax = plt.subplots()

s = q_arr[:, 0]
y = q_arr[:, 1] + (r + shoe_l0)
theta = q_arr[:, 2]
etas = q_arr[:, 3:]
x = np.linspace(0, Rail.beam.L, 100)  # abscissa along the beam
w = np.array([Rail.displacement(s, etas[0]) for s in x])

line = ax.plot(x, w, label=f"Beam @ t={t_arr[0]:.3f}")[0]
xs = np.array([shoe.dk for shoe in Rocket.shoes])*np.cos(theta[0]) + s[0]
ys = np.array([shoe.dk for shoe in Rocket.shoes])*np.sin(theta[0]) + y[0]
points = ax.scatter(xs, ys, c='orange', label="Shoes")
trail = ax.scatter(s[0], y[0], s=2, c='gray')

ax.set(xlim=(0, 30), ylim=(-0.2, 0.05))
ax.set(xlabel="x (m)", ylabel="w (m)", title="Beam Displacement")
ax.grid()
leg = ax.legend()

def update(frame):
    w = np.array([Rail.displacement(xi, etas[frame]) for xi in x])
    trail_data = np.stack([s[:frame:2], y[:frame:2]]).T

    xs = np.array([shoe.dk for shoe in Rocket.shoes])*np.cos(theta[frame]) + s[frame]
    ys = np.array([shoe.dk for shoe in Rocket.shoes])*np.sin(theta[frame]) + y[frame]
    points_data = np.stack([xs, ys]).T
    # Update beam and rocket tracker
    line.set_ydata(w)
    points.set_offsets(points_data)
    trail.set_offsets(trail_data)
    # Update label
    lab = f"Beam @ t={t_arr[frame]:.3f}"
    leg.get_texts()[0].set_text(lab)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t_arr), interval=5)
ani.save('tip-off_animation.mp4', fps=30)
plt.show()
