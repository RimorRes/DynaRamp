import numpy as np

from dynaramp import (
    EulerBernoulliBeam,
    Basis,
    nnr_solver,
    LaunchRail,
    RigidRocket2D,
    SimpleMotor,
    System,
)

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation



b = 0.5
h = 1.0
area = b * h
rho = 7850
mu = rho * area  # linear mass density, kg/m
e_mod = 210e9  # elastic modulus, Pa
sma = b * h ** 3 / 12  # second moment of area, m^4
length = 518 * 25.4e-3 # length, m
Beam = EulerBernoulliBeam(mu, e_mod, sma, length)
Rail = LaunchRail(Beam, n_modes=5, angle=np.pi / 4)

# Black Brandt X approximate parameters
Motor = SimpleMotor(thrust=257e3, isp=280)
mass = 2468.1
r = 22 * 25.4e-3
h_rocket = 439 * 25.4e-3
inertia = 1 / 12 * mass * (3 * r ** 2 + h_rocket ** 2)

lug1 = (679.40 - (533.94 + 660.44)/2) * 25.4e-3
lug2 = (679.40 - (358.16 + 478.42)/2) * 25.4e-3
com = (438.92/2) * 25.4e-3

World_ref = Basis()
Rocket = RigidRocket2D(parent_basis=World_ref, motor=Motor, mass=mass, inertia=inertia)
Rocket.add_shoe(rel_pos=np.array([lug2 - com, r]), friction_coef=0.5, release_point=Rail.beam.L)
Rocket.add_shoe(rel_pos=np.array([lug1 - com, r]), friction_coef=0.5, release_point=Rail.beam.L-(lug2 - lug1))

Sys = System(Rail, Rocket)

# Compute static equilibrium before starting dynamic simulation.
# We fix the CG position s=5.0 and solve for y, theta and modal coords so
# the elastic internal forces balance the external loads (gravity, thrust).
shoe_l0 = Rocket.shoes[0].l0
q0_guess = np.zeros(2 + Rail.n_modes)
q0_guess[:2] = [-(r + shoe_l0), 0.0]

# Solve for static equilibrium with s fixed at 5.0
q0 = Sys.compute_static_equilibrium(s=3.5, q_free0=q0_guess)
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
print("-"*40)
print("Natural frequencies", Beam.omegas)
print('='*40)
print("Starting static equilibrium")
print(f"equili.  s={q0[0]:.4f}  y={q0[1]:.6f}  theta={q0[2]:.6f}  w(L)={w0*1000:.2f} mm")

t_arr = []
q_arr = []
q_dot_arr = []

ffa = False
exit_theta = 0
exit_theta_dot = 0
exit_vel = 0

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
    w = Rail.displacement(Rail.beam.L, q[3:])
    current_time = (t_val + 1) * t_step

    active_shoe_count = sum(q[0] + shoe.dk - shoe.rk * q[2] <= shoe.x_release for shoe in Rocket.shoes)

    t_arr.append(current_time)
    q_arr.append(q)
    q_dot_arr.append(q_dot)

    print(f"t={current_time:.3f}  s={q[0]:.3f} m  y={q[1]:.3f} m  theta={np.degrees(q[2]):.3f}° "
          f"∆w(L)={(w-w0)*1e3:.3f} mm  shoes={active_shoe_count}")

    if active_shoe_count == 0 and not ffa:
        print("Free flight!")
        exit_theta = np.degrees(float(q_arr[-1][2]))
        exit_theta_dot = np.degrees(float(q_dot_arr[-1][2]))
        exit_vel = q_dot_arr[-1][0]
        ffa = True

print(f"Final state: theta={exit_theta:.3f}°  theta_dot={exit_theta_dot:.3f}°/s, v={exit_vel:.3f}m/s")

# Convert to numpy arrays
t_arr = np.array(t_arr)
q_arr = np.array(q_arr)
q_dot_arr = np.array(q_dot_arr)

# Plotting and Animating beam
fig, ax = plt.subplots()
fig.set_size_inches(16, 6)
ax.set(xlim=(0, 30), ylim=(-5, 2.5))
ax.set_aspect('equal')
ax.set(xlabel="x (m)", ylabel="z (m)", title="Beam Displacement vs Length")
ax.grid()
nx = 100
ny = 10
scale = 1e3

s = q_arr[:, 0]
y = q_arr[:, 1] + (r + shoe_l0) - h/2
theta = q_arr[:, 2]
etas = q_arr[:, 3:]

x_val = np.linspace(0, Rail.beam.L, nx)  # abscissa along the beam
y_val = np.linspace(-h/2, h/2, ny)
# Create a 2D mesh grid for the undeformed assembly
X, Y = np.meshgrid(x_val, y_val)

w_static = np.array([Rail.displacement(s, etas[0]) for s in x_val])

static_deflection = np.tile(w_static, (ny, 1))
Y_disp_initial = Y + static_deflection

mesh = ax.pcolormesh(X,
                     Y_disp_initial,
                     np.tile(np.zeros_like(w_static), (ny, 1)),
                     cmap='jet',
                     shading='gouraud',
                     vmin=-1,
                     vmax=1
                     )
fig.colorbar(mesh, ax=ax, label=f"Deflection from static (mm)")

mesh_container = [mesh]

xs = np.array([shoe.dk for shoe in Rocket.shoes])*np.cos(theta[0]) + s[0]
ys = np.array([shoe.dk for shoe in Rocket.shoes])*np.sin(theta[0]) + y[0]
com_point = ax.scatter(s[0], y[0], c='black', label="Center of mass")
points = ax.scatter(xs, ys, c='orange', label="Lugs")
trail = ax.scatter(s[0], y[0], s=2, c='gray')

leg = ax.legend()

def update(frame):
    w = np.array([Rail.displacement(xi, etas[frame]) for xi in x_val])
    trail_data = np.stack([s[:frame:2], y[:frame:2]]).T

    xs = np.array([shoe.dk for shoe in Rocket.shoes])*np.cos(theta[frame]) + s[frame]
    ys = np.array([shoe.dk for shoe in Rocket.shoes])*np.sin(theta[frame]) + y[frame]
    points_data = np.stack([xs, ys]).T
    # Update beam and rocket tracker
    mesh_container[0].remove()
    vibr_amp = np.tile((w - w_static) * scale, (ny, 1))
    deflection = np.tile(w, (ny, 1))
    #Y_disp_current = Y + static_deflection + vibr_amp
    Y_disp_current = Y + deflection

    mesh_container[0] = ax.pcolormesh(
        X,
        Y_disp_current,
        vibr_amp,
        cmap='jet',
        shading='gouraud',
        vmin=-1,
        vmax=1
    )

    points.set_offsets(points_data)
    trail.set_offsets(trail_data)
    com_point.set_offsets([s[frame], y[frame]])
    # Update label
    lab = f"Center of mass @ t={t_arr[frame]:.3f}"
    leg.get_texts()[0].set_text(lab)
    return mesh_container[0],

ani = animation.FuncAnimation(fig, update, frames=len(t_arr), interval=5)

plt.show()
