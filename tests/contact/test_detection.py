import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, ProjectileKinematics, ProjectileState, Slider
from dynaramp.contact import RailProfile, contact_station, evaluate_slider
from dynaramp.common.vecmath import small_rot

I3 = np.identity(3)


def _build_guide():
    topo = dyn.TopologyHandler()
    length = 10
    h, b = 1, 2
    beam = dyn.EulerBernoulliBeam(
        e_id="beam", length=length, density=7800, youngs_mod=210e9,
        shear_mod=80e9, area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12,
    )
    topo.add_elements(beam)
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


@pytest.fixture(scope="module")
def field():
    system = _build_guide()
    modes = system.natural_modes(3, omega_max=300, search_res=3000)
    assert len(modes) >= 1
    return GuideModalField.from_elements(system, ["beam"], modes)


def _rail():
    return RailProfile(1e-3, 1e-3, 1e-3, stiffness=1e7, restitution=0.5, friction=0.2)


def _g(field, x, p, i_r_vi):
    """The perpendicularity residual of Eq. 55 (A_IR = I), recomputed independently."""
    phi_r, phi_theta = field.phi(x)
    a_ipi = small_rot(phi_theta @ p)
    i_r_pi = x * np.array([1.0, 0.0, 0.0]) + phi_r @ p
    return float((i_r_vi - i_r_pi) @ a_ipi[:, 0])


def test_contact_station_straight_guide(field):
    n = field.n_modes
    i_r_vi = np.array([3.5, 0.02, 0.01])
    x = contact_station(field, I3, i_r_vi, np.zeros(n), x0=3.5)
    assert np.isclose(x, 3.5)  # undeformed guide: cross-section at the slider's x-coord


def test_contact_station_perpendicularity_bent_guide(field):
    n = field.n_modes
    p = 1e-3 * np.linspace(1.0, 2.0, n)
    i_r_vi = np.array([3.5, 0.02, 0.01])
    x = contact_station(field, I3, i_r_vi, p, x0=3.5)
    assert abs(_g(field, x, p, i_r_vi)) < 1e-9   # residual driven to zero


def _kin(field, state, p, p_dot):
    return ProjectileKinematics(field).evaluate(state, p, p_dot)


def test_penetration_from_lateral_offset(field):
    n = field.n_modes
    # O1 offset laterally by (y_L, z_L); no attitude, no deformation.
    state = ProjectileState(np.array([3.0, 0.02, -0.008, 0.0, 0.0, 0.0]), np.zeros(6))
    kin = _kin(field, state, np.zeros(n), np.zeros(n))
    slider = Slider(position=(0.5, 0.0, 0.0), radius=0.0)
    sc = evaluate_slider(kin, field, I3, np.zeros(n), np.zeros(n), slider, _rail(),
                         x_r=3.0, l_c=100.0)
    labels = {s.label: s.penetration for s in sc.surfaces}
    assert np.isclose(labels["side_+y"], 0.02 - 1e-3)     # |y| - lateral clearance
    assert np.isclose(labels["bottom"], 0.008 - 1e-3)     # -z - bottom clearance


def test_newtons_third_law(field):
    n = field.n_modes
    state = ProjectileState(np.array([3.0, 0.02, -0.008, 0.05, -0.03, 0.02]),
                            np.array([0.0, 0.4, 0.3, 0.1, -0.2, 0.15]))
    kin = _kin(field, state, 1e-3 * np.ones(n), 1e-3 * np.ones(n))
    slider = Slider(position=(0.5, 0.0, 0.0), radius=0.0)
    sc = evaluate_slider(kin, field, I3, 1e-3 * np.ones(n), 1e-3 * np.ones(n), slider,
                         _rail(), x_r=3.0, l_c=100.0)
    assert len(sc.surfaces) > 0
    assert np.allclose(sc.i_q_pi, -sc.i_q_o1)             # Eq. B24
    # moment on the guide uses the guide-side lever arm, not O1's
    assert not np.allclose(sc.i_m_pi, -sc.i_m_o1)


def test_out_of_phase_no_contact(field):
    n = field.n_modes
    state = ProjectileState(np.array([3.0, 0.02, -0.008, 0.0, 0.0, 0.0]), np.zeros(6))
    kin = _kin(field, state, np.zeros(n), np.zeros(n))
    slider = Slider(position=(0.5, 0.0, 0.0), radius=0.0)
    sc = evaluate_slider(kin, field, I3, np.zeros(n), np.zeros(n), slider, _rail(),
                         x_r=3.0, l_c=1.0)   # exit well behind the slider (~3.5)
    assert sc.in_phase is False
    assert sc.surfaces == []
    assert np.allclose(sc.i_q_o1, 0.0)


def test_impact_velocity_matches_finite_difference(field):
    # Rigid, undeformed guide (p = 0); O1 translates laterally. The per-face penetration
    # velocity must equal d(penetration)/dt from a central finite difference.
    n = field.n_modes
    slider = Slider(position=(0.5, 0.0, 0.0), radius=0.0)
    x_dot = np.array([0.0, 0.5, 0.3, 0.0, 0.0, 0.0])   # pure lateral translation

    def delta(t, label):
        x = np.array([3.0, 0.02, -0.008, 0.0, 0.0, 0.0]) + x_dot * t
        st = ProjectileState.from_config_rates(x, x_dot)
        kin = _kin(field, st, np.zeros(n), np.zeros(n))
        sc = evaluate_slider(kin, field, I3, np.zeros(n), np.zeros(n), slider, _rail(),
                             x_r=x[0], l_c=100.0)
        return {s.label: s.penetration for s in sc.surfaces}[label]

    st0 = ProjectileState.from_config_rates(np.array([3.0, 0.02, -0.008, 0.0, 0.0, 0.0]), x_dot)
    kin0 = _kin(field, st0, np.zeros(n), np.zeros(n))
    sc0 = evaluate_slider(kin0, field, I3, np.zeros(n), np.zeros(n), slider, _rail(),
                          x_r=3.0, l_c=100.0)

    dt = 1e-6
    for label in ("side_+y", "bottom"):
        num = (delta(dt, label) - delta(-dt, label)) / (2 * dt)
        assert np.isclose(sc0.penetration_velocities[label], num, atol=1e-7)
