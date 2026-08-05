import numpy as np

from dynaramp.projectile import Projectile, ProjectileKinematicState
from dynaramp.simulation import ExternalForce, Gravity, Thrust, sum_external_forces, G0
from dynaramp.common.vecmath import euler_zyx, skew_sym_mat


def _kin(a_ib=None):
    a_ib = np.identity(3) if a_ib is None else a_ib
    return ProjectileKinematicState(
        a_il=np.identity(3), a_ib=a_ib,
        omega_il=np.zeros(3), omega_ib=np.zeros(3),
        r_o1=np.zeros(3), r_dot_o1=np.zeros(3),
        l_to1=np.zeros((3, 1)), j_to1=np.zeros((3, 6)), zeta_to1=np.zeros(3),
        l_rp=np.zeros((3, 1)), j_ro1=np.zeros((3, 6)), zeta_ro1=np.zeros(3),
    )


def _projectile(com=(0.0, 0.0, 0.0), mass=100.0):
    return Projectile(mass=mass, inertia_com=np.diag([1.0, 2.0, 3.0]), com_o1=com)


def test_gravity_force_and_moment():
    m, com = 100.0, np.array([0.5, 0.0, 0.0])
    grav = Gravity()  # default (0, -g, 0)
    q, mom = grav(_kin(), _projectile(com=com, mass=m), 0.0)
    assert np.allclose(q, [0.0, -m * G0, 0.0])
    assert np.allclose(mom, skew_sym_mat(com) @ q)


def test_gravity_no_moment_when_com_at_o1():
    _, mom = Gravity()(_kin(), _projectile(com=(0, 0, 0)), 0.0)
    assert np.allclose(mom, 0.0)


def test_gravity_rotates_with_frame_only_for_moment():
    # Gravity direction is fixed in K_I; the lever arm rotates with the body.
    a_ib = euler_zyx(0.3, -0.2, 0.1)
    com = np.array([0.4, 0.1, -0.2])
    q, mom = Gravity()(_kin(a_ib), _projectile(com=com), 0.0)
    assert np.allclose(q, [0.0, -100.0 * G0, 0.0])           # force unchanged
    assert np.allclose(mom, skew_sym_mat(a_ib @ com) @ q)     # arm = A_IB * com


def test_thrust_constant_on_axis_no_moment():
    thr = Thrust(curve=lambda t: 1000.0)  # on-axis application -> no moment
    q, mom = thr(_kin(), _projectile(), 0.0)
    assert np.allclose(q, [1000.0, 0.0, 0.0])
    assert np.allclose(mom, 0.0)


def test_thrust_offset_application_makes_moment():
    thr = Thrust(curve=lambda t: 1000.0, application_point=(0.0, 0.1, 0.0))
    q, mom = thr(_kin(), _projectile(), 0.0)
    assert np.allclose(q, [1000.0, 0.0, 0.0])
    assert np.allclose(mom, skew_sym_mat([0.0, 0.1, 0.0]) @ q)  # -> [0, 0, -100]


def test_thrust_time_curve_and_rotation():
    a_ib = euler_zyx(0.2, 0.1, -0.1)
    thr = Thrust(curve=lambda t: 2000.0 * t)   # ramps with time
    q, _ = thr(_kin(a_ib), _projectile(), 0.5)  # T(0.5) = 1000
    assert np.allclose(q, 1000.0 * (a_ib @ np.array([1.0, 0.0, 0.0])))


def test_sum_external_forces():
    proj = _projectile(com=(0.3, 0.0, 0.0))
    kin = _kin()
    forces = [Gravity(), Thrust(curve=lambda t: 500.0)]
    q, mom = sum_external_forces(forces, kin, proj, 0.0)
    gq, gm = Gravity()(kin, proj, 0.0)
    tq, tm = Thrust(curve=lambda t: 500.0)(kin, proj, 0.0)
    assert np.allclose(q, gq + tq)
    assert np.allclose(mom, gm + tm)


def test_forces_satisfy_protocol():
    assert isinstance(Gravity(), ExternalForce)
    assert isinstance(Thrust(curve=lambda t: 0.0), ExternalForce)
