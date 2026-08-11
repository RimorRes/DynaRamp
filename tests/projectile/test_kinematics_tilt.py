"""
Phase B: A_IR (guide attitude) propagation through the section 3 kinematics.

Reorienting the whole guide by a rigid rotation R (a_ir -> R @ a_ir) while holding the
projectile's *relative* state (x, y, p, p_dot) fixed must rotate every K_I-projected
kinematic quantity by R. This is an exact covariance property of the model, so it is a
tight end-to-end check that A_IR is threaded consistently.
"""
import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, ProjectileKinematics, ProjectileState, Slider
from dynaramp.contact import RailProfile, ContactSolver
from dynaramp.common.vecmath import euler_zyx

X0 = 3.5
R = euler_zyx(0.3, -0.2, 0.15)          # arbitrary guide reorientation (K_R -> K_I)


def _build_cantilever(orientation=None):
    topo = dyn.TopologyHandler()
    length, h, b = 10.0, 1.0, 2.0
    beam = dyn.EulerBernoulliBeam(
        e_id="beam", length=length, density=7800.0, youngs_mod=210e9, shear_mod=80e9,
        area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12, orientation=orientation,
    )
    topo.add_elements(beam)
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


@pytest.fixture(scope="module")
def modes():
    system = _build_cantilever()
    m = sorted(system.natural_modes(3, omega_max=300, search_res=3000), key=lambda md: md.frequency)
    assert len(m) >= 1
    return system, m


@pytest.fixture(scope="module")
def fields(modes):
    system, m = modes
    field_i = GuideModalField.from_elements(system, ["beam"], m)              # upright
    field_r = GuideModalField.from_elements(system, ["beam"], m, a_ir=R)      # reoriented
    return field_i, field_r


def _state_and_modal(n):
    rng = np.random.default_rng(7)
    p = 1e-3 * rng.standard_normal(n)
    p_dot = 1e-3 * rng.standard_normal(n)
    y = np.array([1.3, 0.2, -0.15, 0.3, -0.25, 0.2])
    state = ProjectileState(np.array([X0, 0.03, -0.02, 0.1, -0.2, 0.15]), y)
    return state, p, p_dot


def test_kinematic_state_is_covariant_under_guide_rotation(fields):
    field_i, field_r = fields
    n = field_i.n_modes
    state, p, p_dot = _state_and_modal(n)

    kin_i = ProjectileKinematics(field_i).evaluate(state, p, p_dot)
    kin_r = ProjectileKinematics(field_r).evaluate(state, p, p_dot)

    # Every projected block is left-multiplied by R (matrices A_IL/A_IB and the influence
    # blocks) or rotated as a vector (r_o1, omega, zeta).
    for attr in (
        "a_il", "a_ib", "omega_il", "omega_ib", "r_o1", "r_dot_o1",
        "l_to1", "j_to1", "zeta_to1", "l_rp", "j_ro1", "zeta_ro1",
    ):
        got = getattr(kin_r, attr)
        expected = R @ getattr(kin_i, attr)
        assert np.allclose(got, expected, atol=1e-9), attr


def test_relative_attitude_is_frame_invariant(fields):
    # A_LB (body attitude relative to the guide) must not depend on the guide's own
    # attitude: A_LB = A_IL^T A_IB is identical in both frames.
    field_i, field_r = fields
    n = field_i.n_modes
    state, p, p_dot = _state_and_modal(n)
    kin_i = ProjectileKinematics(field_i).evaluate(state, p, p_dot)
    kin_r = ProjectileKinematics(field_r).evaluate(state, p, p_dot)
    a_lb_i = kin_i.a_il.T @ kin_i.a_ib
    a_lb_r = kin_r.a_il.T @ kin_r.a_ib
    assert np.allclose(a_lb_i, a_lb_r, atol=1e-12)


def test_solver_defaults_a_ir_to_field(fields):
    field_i, field_r = fields
    profile = RailProfile(1e-3, 1e-3, 1e-3, stiffness=1e7, restitution=0.5, friction=0.2)
    sliders = [Slider(position=(0.5, 0.0, 0.0), radius=0.0)]

    assert np.allclose(ContactSolver(field_r, profile, sliders).a_ir, R)
    assert np.allclose(ContactSolver(field_i, profile, sliders).a_ir, np.identity(3))
    # An explicit a_ir still overrides (advanced use).
    assert np.allclose(ContactSolver(field_i, profile, sliders, a_ir=R).a_ir, R)


def test_element_orientation_leaves_spectrum_invariant(modes):
    # Cross-check on the Phase A <-> Phase B bridge: tilting the single straight rail via
    # the beam element's own orientation (Phase A) yields the same natural frequencies as
    # the axis-aligned rail later projected by a_ir (Phase B). (Mode-shape magnitudes are
    # only defined up to the modal-normalization gauge, so only the spectrum is compared.)
    _, m_axis = modes
    system_rot = _build_cantilever(orientation=R)
    m_rot = sorted(system_rot.natural_modes(3, omega_max=300, search_res=3000),
                   key=lambda md: md.frequency)
    assert np.allclose([md.frequency for md in m_axis],
                       [md.frequency for md in m_rot], rtol=1e-4)
