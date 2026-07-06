import numpy as np
import dynaramp.mstmm as dyn


def create_mass_spring_oscillator():
    system = dyn.MBS()

    m1 = 5
    s = 1
    unit_inertia = 1/6 * s**2 * np.eye(3)
    mass_elem = dyn.RigidBody(
        e_id='mass',
        mass=m1,
        inertia=m1 * unit_inertia,
        com=(0, 0, s/2),
        slot_coords={
            'output': (0, 0, s)
        }
    )

    length = 1
    k = 20
    spring_elem = dyn.SpatialElasticHinge(
        e_id='spring',
        k=(np.inf, np.inf, k),
        k_rot=(np.inf, np.inf, np.inf),
        slot_coords={
            'output': (0, 0, length)
        }
    )

    m2 = 5
    mass_elem2 = dyn.RigidBody(
        e_id='mass2',
        mass=m2,
        inertia=unit_inertia,
        com=(0, 0, s / 2),
        slot_coords={
            'output': (0, 0, s)
        }
    )

    spring_elem2 = dyn.SpatialElasticHinge(
        e_id='spring2',
        k=(np.inf, np.inf, k),
        k_rot=(np.inf, np.inf, np.inf),
        slot_coords={
            'output': (0, 0, length)
        }
    )

    system.add_elements([spring_elem, mass_elem, spring_elem2, mass_elem2])
    system.connect_elements(spring_elem, mass_elem, src_slot='output', dst_slot=None)
    system.connect_elements(mass_elem, spring_elem2, src_slot='output', dst_slot=None)
    system.connect_elements(spring_elem2, mass_elem2, src_slot='output', dst_slot=None)

    # z = [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y, Q_z, 1]
    tip_boundary = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None, 1])
    root_boundary = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 1])

    system.add_root(root_boundary, mass_elem2, 'output')
    system.add_tip(tip_boundary, spring_elem, None)

    system.make_tree()

    return system

def create_parallel_mass_spring_oscillator():
    system = dyn.MBS()

    m = 5
    s = 1
    unit_inertia = 1 / 6 * s ** 2 * np.eye(3)
    mass_elem = dyn.RigidBody(
        e_id='mass',
        mass=m,
        inertia=m * unit_inertia,
        com=(0, 0, s / 2),
        slot_coords={
            'aux_in': (0, 0, 0),
            'output': (0, 0, s)
        }
    )

    length = 1
    k = 20
    spring_elem1 = dyn.SpatialElasticHinge(
        e_id='spring1',
        k=(np.inf, np.inf, k),
        k_rot=(np.inf, np.inf, np.inf),
        slot_coords={
            'output': (0, 0, length)
        }
    )

    spring_elem2 = dyn.SpatialElasticHinge(
        e_id='spring2',
        k=(np.inf, np.inf, k),
        k_rot=(np.inf, np.inf, np.inf),
        slot_coords={
            'output': (0, 0, length)
        }
    )

    system.add_elements([spring_elem1, spring_elem2, mass_elem])
    system.connect_elements(spring_elem1, mass_elem, src_slot='output', dst_slot=None)
    system.connect_elements(spring_elem2, mass_elem, src_slot='output', dst_slot='aux_in')

    # z = [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y, Q_z, 1]
    tip_boundary1 = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None, 1])
    tip_boundary2 = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None, 1])
    root_boundary = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 1])

    system.add_root(root_boundary, mass_elem, 'output')
    system.add_tip(tip_boundary1, spring_elem1, None)
    system.add_tip(tip_boundary2, spring_elem2, None)

    system.make_tree()

    return system


def test_mass_spring_oscillator():
    oscillator = create_mass_spring_oscillator()
    u, f, _ = oscillator.overall_transfer(5)
    assert np.allclose(f, 0)

    modes = oscillator.natural_modes(2, omega_max=50)
    ws, shapes = zip(*modes)
    # Theoretical natural frequencies for a mass-spring system with two masses and two identical springs in series
    k = 20
    m1 = 5
    m2 = 5
    omega1 = np.sqrt(k/(2*m1) * (2 + (m1/m2) - np.sqrt(4 + (m1/m2)**2)))
    omega2 = np.sqrt(k/(2*m1) * (2 + (m1/m2) + np.sqrt(4 + (m1/m2)**2)))
    assert np.allclose(ws, [omega1, omega2], rtol=1e-3)

def test_parallel_mass_spring_oscillator():
    oscillator = create_parallel_mass_spring_oscillator()
    u, f, _ = oscillator.overall_transfer(5)
    assert np.allclose(f, 0)

    mode = oscillator.natural_modes(1, omega_max=10)[0]
    # Theoretical natural frequency for a mass-spring system with one mass and two identical springs in parallel
    k = 20
    m = 5
    omega = np.sqrt(2*k/m)
    assert np.allclose(mode[0], omega, rtol=1e-3)
