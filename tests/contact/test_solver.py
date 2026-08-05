import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, ProjectileKinematics, ProjectileState, Slider
from dynaramp.contact import RailProfile, ContactSolver

I3 = np.identity(3)


def _beam(e_id, length):
    h, b = 1, 2
    return dyn.EulerBernoulliBeam(
        e_id=e_id, length=length, density=7800, youngs_mod=210e9,
        shear_mod=80e9, area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12,
    )


TIP = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
ROOT = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])


def _single_guide():
    topo = dyn.TopologyHandler()
    beam = _beam("beam", 10)
    topo.add_elements(beam)
    topo.add_root(beam, ROOT, output_pos=(10, 0, 0))
    topo.add_tip(beam, TIP, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


def _two_segment_guide(l1, l2):
    topo = dyn.TopologyHandler()
    seg1, seg2 = _beam("seg1", l1), _beam("seg2", l2)
    topo.add_elements([seg1, seg2])
    topo.connect_elements(seg1, seg2, src_pos=(l1, 0, 0), dst_pos=(0, 0, 0))
    topo.add_tip(seg1, TIP, input_pos=(0, 0, 0))
    topo.add_root(seg2, ROOT, output_pos=(l2, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


@pytest.fixture(scope="module")
def field():
    system = _single_guide()
    modes = system.natural_modes(3, omega_max=300, search_res=3000)
    return GuideModalField.from_elements(system, ["beam"], modes)


def _profile():
    return RailProfile(1e-3, 1e-3, 1e-3, stiffness=1e7, restitution=0.5, friction=0.2)


def _sliders():
    # rear pair at O1, front pair 2 m ahead, all on the axis (only axial offset).
    return [Slider((0.0, 0.0, 0.0), 0.0), Slider((2.0, 0.0, 0.0), 0.0)]


def _kin(field, state, p, p_dot):
    return ProjectileKinematics(field).evaluate(state, p, p_dot)


def test_solver_sums_and_aggregate_third_law(field):
    n = field.n_modes
    p, p_dot = 1e-3 * np.ones(n), 1e-3 * np.ones(n)
    state = ProjectileState(np.array([3.0, 0.02, -0.008, 0.0, 0.0, 0.0]),
                            np.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0]))
    kin = _kin(field, state, p, p_dot)
    solver = ContactSolver(field, _profile(), _sliders())
    res = solver.evaluate(kin, x_r=3.0, p=p, p_dot=p_dot)

    assert len(res.guide_reactions) == 2                 # both sliders in contact
    assert np.all(np.isfinite(res.sum_q_o1))
    assert np.linalg.norm(res.sum_q_o1) > 0.0
    # Aggregate Newton's third law: guide reactions balance the projectile force.
    total_guide = sum((r.i_q_pi for r in res.guide_reactions), np.zeros(3))
    assert np.allclose(total_guide, -res.sum_q_o1)


def test_solver_no_contact_gives_empty_result(field):
    n = field.n_modes
    state = ProjectileState(np.array([3.0, 1e-4, -1e-4, 0.0, 0.0, 0.0]), np.zeros(6))
    kin = _kin(field, state, np.zeros(n), np.zeros(n))
    res = ContactSolver(field, _profile(), _sliders()).evaluate(kin, 3.0, np.zeros(n), np.zeros(n))
    assert np.allclose(res.sum_q_o1, 0.0)
    assert np.allclose(res.sum_m_o1, 0.0)
    assert res.guide_reactions == []
    assert res.memory == {}


def test_solver_out_of_phase_slider_excluded(field):
    n = field.n_modes
    p, p_dot = np.zeros(n), np.zeros(n)
    state = ProjectileState(np.array([3.0, 0.02, -0.008, 0.0, 0.0, 0.0]),
                            np.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0]))
    kin = _kin(field, state, p, p_dot)
    # Exit at 4.5 m: the rear slider (~3.0) stays in, the front slider (~5.0) has exited.
    solver = ContactSolver(field, _profile(), _sliders(), l_c=4.5)
    res = solver.evaluate(kin, 3.0, p, p_dot)
    assert res.slider_contacts[0].in_phase is True
    assert res.slider_contacts[1].in_phase is False
    assert len(res.guide_reactions) == 1
    assert set(res.memory.keys()) == {0}


def test_solver_memory_latches_onset_velocity(field):
    n = field.n_modes
    p, p_dot = 1e-3 * np.ones(n), 1e-3 * np.ones(n)
    x = np.array([3.0, 0.02, -0.008, 0.0, 0.0, 0.0])
    solver = ContactSolver(field, _profile(), _sliders())

    # Step 1: first contact -> onset velocity latched.
    kin1 = _kin(field, ProjectileState(x, np.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0])), p, p_dot)
    res1 = solver.evaluate(kin1, 3.0, p, p_dot)
    # Step 2: same penetration, DIFFERENT velocity -> onset memory must be unchanged.
    kin2 = _kin(field, ProjectileState(x, np.array([0.0, 0.9, 0.7, 0.0, 0.0, 0.0])), p, p_dot)
    res2 = solver.evaluate(kin2, 3.0, p, p_dot, memory=res1.memory)

    assert res1.memory.keys() == res2.memory.keys()
    for idx in res1.memory:
        assert res1.memory[idx] == res2.memory[idx]  # latched onset values preserved


def test_solver_end_to_end_two_segment_bent_guide():
    # Full section 2 -> 3 -> 4 chain on a subdivided, deformed guide.
    system = _two_segment_guide(4.0, 6.0)
    modes = system.natural_modes(3, omega_max=300, search_res=3000)
    field = GuideModalField.from_elements(system, ["seg1", "seg2"], modes)
    n = field.n_modes
    p = 2e-3 * np.linspace(1.0, 2.0, n)
    p_dot = 1e-3 * np.linspace(-1.0, 1.0, n)
    state = ProjectileState(np.array([3.0, 0.015, -0.01, 0.02, -0.01, 0.0]),
                            np.array([8.0, 0.2, 0.1, 0.05, -0.03, 0.0]))
    kin = _kin(field, state, p, p_dot)
    solver = ContactSolver(field, _profile(), _sliders())
    res = solver.evaluate(kin, x_r=3.0, p=p, p_dot=p_dot)

    assert np.all(np.isfinite(res.sum_q_o1)) and np.all(np.isfinite(res.sum_m_o1))
    total_guide = sum((r.i_q_pi for r in res.guide_reactions), np.zeros(3))
    assert np.allclose(total_guide, -res.sum_q_o1)
    # every contacting slider resolved to a station on the guide
    for r in res.guide_reactions:
        assert 0.0 <= r.x_r_i <= field.total_length + 0.5


def test_solver_per_slider_exit_enables_simultaneous_detachment(field):
    n = field.n_modes
    p, p_dot = np.zeros(n), np.zeros(n)

    def state_at(x_r):
        return ProjectileState(np.array([x_r, 0.02, -0.008, 0.0, 0.0, 0.0]),
                               np.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0]))

    # Single exit at 4.0: the front slider (station x_r + 2) leaves before the rear one.
    single = ContactSolver(field, _profile(), _sliders(), l_c=4.0)
    res = single.evaluate(_kin(field, state_at(3.5), p, p_dot), 3.5, p, p_dot)
    assert res.slider_contacts[0].in_phase is True       # rear still engaged
    assert res.slider_contacts[1].in_phase is False      # front already released (sequential)

    # Per-slider exits [4.0, 6.0]: both cross at x_r = 4.0 -> simultaneous detachment.
    pair = ContactSolver(field, _profile(), _sliders(), l_c=[4.0, 6.0])
    before = pair.evaluate(_kin(field, state_at(3.5), p, p_dot), 3.5, p, p_dot)
    assert before.slider_contacts[0].in_phase and before.slider_contacts[1].in_phase
    after = pair.evaluate(_kin(field, state_at(4.5), p, p_dot), 4.5, p, p_dot)
    assert not after.slider_contacts[0].in_phase and not after.slider_contacts[1].in_phase


def test_solver_l_c_length_mismatch_raises(field):
    with pytest.raises(ValueError):
        ContactSolver(field, _profile(), _sliders(), l_c=[1.0, 2.0, 3.0])  # 3 != 2 sliders
