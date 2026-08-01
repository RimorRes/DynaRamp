import numpy as np
from scipy.linalg import solve
import dynaramp.mstmm as dyn


def create_mass_spring_oscillator():
    system = dyn.System()

    m1 = 5
    s = 1
    unit_inertia = 1/6 * s**2 * np.eye(3)
    mass_elem = dyn.RigidBody(
        e_id='mass',
        mass=m1,
        inertia=m1 * unit_inertia,
        com=(0, 0, s/2)
    )

    length = 1
    k = 20
    spring_elem = dyn.SpatialElasticHinge(
        e_id='spring',
        k=(np.inf, np.inf, k),
        k_rot=(np.inf, np.inf, np.inf)
    )

    m2 = 5
    mass_elem2 = dyn.RigidBody(
        e_id='mass2',
        mass=m2,
        inertia=unit_inertia,
        com=(0, 0, s / 2)
    )

    spring_elem2 = dyn.SpatialElasticHinge(
        e_id='spring2',
        k=(np.inf, np.inf, k),
        k_rot=(np.inf, np.inf, np.inf)
    )

    system.add_elements([spring_elem, mass_elem, spring_elem2, mass_elem2])
    system.connect_elements(spring_elem, mass_elem, src_pos=(0, 0, length), dst_pos=(0, 0, 0))
    system.connect_elements(mass_elem, spring_elem2, src_pos=(0, 0, s), dst_pos=(0, 0, 0))
    system.connect_elements(spring_elem2, mass_elem2, src_pos=(0, 0, length), dst_pos=(0, 0, 0))

    # z = [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y, Q_z, 1]
    tip_boundary = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None, 1])
    root_boundary = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 1])

    system.add_root(mass_elem2, root_boundary, output_pos=(0, 0, s))
    system.add_tip(spring_elem, tip_boundary, input_pos=(0, 0, 0))

    system.make_tree()

    return system


def create_parallel_mass_spring_oscillator(n):
    system = dyn.System()

    m = 5
    s = 1
    unit_inertia = 1 / 6 * s ** 2 * np.eye(3)

    mass_elem = dyn.RigidBody(
        e_id='mass',
        mass=m,
        inertia=m * unit_inertia,
        com=(0, 0, s / 2)
    )

    length = 1
    k = 20
    k_penalty = 1e5
    spring_elems = []
    for i in range(n):
        i_spring = dyn.SpatialElasticHinge(
            e_id='spring' + str(i),
            k=(k_penalty, k_penalty, k),
            k_rot=(k_penalty, k_penalty, k_penalty)
        )
        spring_elems.append(i_spring)

    system.add_elements(mass_elem)

    system.add_elements(spring_elems)
    for spring in spring_elems:
        system.connect_elements(spring, mass_elem, src_pos=(0, 0, length), dst_pos=(0, 0, 0))
        tip_boundary = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None, 1])
        system.add_tip(spring, tip_boundary, (0, 0, 0))

    # z = [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y, Q_z, 1]
    root_boundary = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 1])
    system.add_root(mass_elem, root_boundary, output_pos=(0, 0, s))

    system.make_tree()

    return system


def create_simple_multi_output_system():
    system = dyn.System()

    m = 5
    s = 1
    unit_inertia = 1 / 6 * s ** 2 * np.eye(3)
    mass_1 = dyn.RigidBody(
        e_id='mass_1',
        mass=m,
        inertia=m * unit_inertia,
        com=(0, 0, s / 2)
    )
    mass_4 = dyn.RigidBody(
        e_id='mass_4',
        mass=m,
        inertia=m * unit_inertia,
        com=(s / 2, 0, s / 2)
    )

    length = 1
    k = 10
    k_penalty = 1e5
    spring_2 = dyn.SpatialElasticHinge(
        e_id='spring_2',
        k=(k_penalty, k_penalty, k),
        k_rot=(k_penalty, k_penalty, k_penalty)
    )
    spring_3 = dyn.SpatialElasticHinge(
        e_id='spring_3',
        k=(k_penalty, k_penalty, k),
        k_rot=(k_penalty, k_penalty, k_penalty)
    )

    system.add_elements([mass_1, spring_2, spring_3, mass_4])
    system.connect_elements(mass_1, spring_2, src_pos=(-s / 2, 0, s), dst_pos=(0, 0, 0))
    system.connect_elements(mass_1, spring_3, src_pos=(s / 2, 0, s), dst_pos=(0, 0, 0))
    system.connect_elements(spring_2, mass_4, src_pos=(0, 0, length), dst_pos=(0, 0, 0))
    system.connect_elements(spring_3, mass_4, src_pos=(0, 0, length), dst_pos=(s, 0, s),)
    # z = [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y, Q_z, 1]
    tip_sv = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 1])
    root_sv = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None, 1])

    system.add_root(mass_4, root_sv, output_pos=(s / 2, 0, s))
    system.add_tip(mass_1, tip_sv, input_pos=(0, 0, 0))

    system.make_tree()

    return system


def create_simple_closed_loop_system(auto_cut: bool = False):
    system = dyn.System()

    # Masses
    m = 2
    width = 1  # X
    depth = 1  # Y
    height = 2  # Z
    i_xx = 1 / 12 * m * (depth ** 2 + height ** 2)
    i_yy = 1 / 12 * m * (width ** 2 + height ** 2)
    i_zz = 1 / 12 * m * (width ** 2 + depth ** 2)
    inertia = np.diag([i_xx, i_yy, i_zz])
    # Springs
    k = 20
    k_penalty = 1e6
    length = 1

    mass_1 = dyn.RigidBody(
        e_id='mass_1',
        mass=m,
        inertia=inertia,
        com=(-width / 2, 0, height / 2)
    )

    spring_2 = dyn.SpatialElasticHinge(
        e_id='spring_2',
        k=(k, k_penalty, k_penalty),
        k_rot=(k_penalty, k_penalty, k_penalty)
    )

    mass_3 = dyn.RigidBody(
        e_id='mass_3',
        mass=m,
        inertia=inertia,
        com=(width / 2, 0, -height / 2)
    )

    spring_4 = dyn.SpatialElasticHinge(
        e_id='spring_4',
        k=(k, k_penalty, k_penalty),
        k_rot=(k_penalty, k_penalty, k_penalty)
    )

    system.add_elements((mass_1, spring_2, mass_3, spring_4))
    system.connect_elements(mass_1, spring_2, src_pos=(0, 0, height), dst_pos=(0, 0, 0))
    system.connect_elements(spring_2, mass_3, src_pos=(length, 0, 0), dst_pos=(0, 0, 0))
    system.connect_elements(mass_3, spring_4, src_pos=(0, 0, -height), dst_pos=(0, 0, 0))
    system.connect_elements(spring_4, mass_1, src_pos=(-length, 0, 0), dst_pos=(0, 0, 0))
    if not auto_cut:
        system.cut_connection((spring_4, mass_1))

    # No explicit tip boundaries
    system.make_tree()

    return system


def create_static_loading():
    system = dyn.System()

    m = 5
    s = 1
    unit_inertia = 1 / 6 * s ** 2 * np.eye(3)
    mass_1 = dyn.RigidBody(
        e_id='mass_1',
        mass=m,
        inertia=m * unit_inertia,
        com=(0, 0, s / 2)
    )

    length = 1
    k = 20
    k_penalty = 1e5
    spring_2 = dyn.SpatialElasticHinge(
        e_id='spring_2',
        k=(k_penalty, k_penalty, k),
        k_rot=(k_penalty, k_penalty, k_penalty)
    )

    mass_1.apply_force(force=(0, 0, -9.81*m), point=(0, 0, s / 2))

    system.add_elements([mass_1, spring_2])
    system.connect_elements(mass_1, spring_2, src_pos=(0, 0, s), dst_pos=(0, 0, 0))

    system.add_root(spring_2, np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None, 1]), output_pos=(0, 0, length))
    system.add_tip(mass_1, np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 1]), input_pos=(0, 0, 0))

    system.make_tree()

    return system


def test_mass_spring_oscillator():
    # Theoretical natural frequencies for a mass-spring system with two masses and two identical springs in series
    k = 20
    m1 = 5
    m2 = 5
    omega1 = np.sqrt(k / (2 * m1) * (2 + (m1 / m2) - np.sqrt(4 + (m1 / m2) ** 2)))
    omega2 = np.sqrt(k / (2 * m1) * (2 + (m1 / m2) + np.sqrt(4 + (m1 / m2) ** 2)))

    oscillator = create_mass_spring_oscillator()
    u, f, _, __ = oscillator.overall_transfer_mat(omega1)
    assert np.allclose(f, 0)

    modes = oscillator.natural_modes(2, omega_max=50)
    ws, shapes = zip(*modes)

    assert np.allclose(ws, [omega1, omega2], rtol=1e-3)


def test_parallel_mass_spring_oscillator():
    n = 3
    k = 20
    m = 5
    # Theoretical natural frequency for a mass-spring system with one mass and two identical springs in parallel
    omega = np.sqrt(n*k/m)
    omega_min = omega - 1
    omega_max = omega + 1

    oscillator = create_parallel_mass_spring_oscillator(n)
    u, f, _, __ = oscillator.overall_transfer_mat(omega)
    assert np.allclose(f, 0)

    w, _ = oscillator.natural_modes(1, omega_min=omega_min, omega_max=omega_max)[0]

    assert np.isclose(w, omega, rtol=1e-3)


def test_multi_output_element():
    k = 10
    m = 5
    # Theoretical natural frequency for a mass-spring system with one mass and two identical springs in parallel
    omega = np.sqrt(2 * k / m)
    omega_min = omega - 1
    omega_max = omega + 1

    system = create_simple_multi_output_system()
    w, _ = system.natural_modes(1, omega_min=omega_min, omega_max=omega_max)[0]

    assert np.isclose(w, omega, rtol=1e-3)


def test_simple_closed_loop_system():
    k = 20
    m = 2
    # Theoretical natural frequency for a mass-spring system with one mass and two identical springs in parallel
    omega = np.sqrt(4*k/m)

    system = create_simple_closed_loop_system()
    w, _ = system.natural_modes(1, omega_min=6, omega_max=7)[0]

    assert np.isclose(w, omega, rtol=1e-3)


def test_closed_loop_auto_cut():
    k = 20
    m = 2
    # Theoretical natural frequency for a mass-spring system with one mass and two identical springs in parallel
    omega = np.sqrt(4 * k / m)

    system = create_simple_closed_loop_system(auto_cut=True)
    w, _ = system.natural_modes(1, omega_min=6, omega_max=7)[0]

    assert np.isclose(w, omega, rtol=1e-3)


def test_static_loading():
    k = 20
    m = 5

    system = create_static_loading()
    u, f, z_rem, rem_bounds = system.overall_transfer_mat(0)
    z_red = solve(u, f)

    state_vecs = system.propagate_state(0, z_red, z_rem, rem_bounds)
    tip = next(iter(system.tips))
    assert np.isclose(state_vecs[tip][2], -9.81*m/k, rtol=1e-3)
