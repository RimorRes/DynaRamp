"""
Phase B: A_IR propagation through the section 4 contact analysis.

Reorienting the guide by R (a_ir -> R @ a_ir) with the projectile's relative state held
fixed must leave the *scalar* contact quantities (station, penetrations, penetration
velocities) invariant -- they live in the cross-section frame K_Pi -- while rotating the
K_I-projected resultant force/moment on the projectile and on the guide by R.
"""
import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, ProjectileKinematics, ProjectileState, Slider
from dynaramp.contact import RailProfile, evaluate_slider
from dynaramp.common.vecmath import euler_zyx

I3 = np.identity(3)
R = euler_zyx(0.25, -0.15, 0.2)


def _build_guide():
    topo = dyn.TopologyHandler()
    length, h, b = 10.0, 1.0, 2.0
    beam = dyn.EulerBernoulliBeam(
        e_id="beam", length=length, density=7800.0, youngs_mod=210e9, shear_mod=80e9,
        area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12,
    )
    topo.add_elements(beam)
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


@pytest.fixture(scope="module")
def fields():
    system = _build_guide()
    modes = system.natural_modes(3, omega_max=300, search_res=3000)
    assert len(modes) >= 1
    field_i = GuideModalField.from_elements(system, ["beam"], modes)
    field_r = GuideModalField.from_elements(system, ["beam"], modes, a_ir=R)
    return field_i, field_r


def _rail():
    return RailProfile(1e-3, 1e-3, 1e-3, stiffness=1e7, restitution=0.5, friction=0.2)


def test_contact_resultants_covariant_under_guide_rotation(fields):
    field_i, field_r = fields
    n = field_i.n_modes
    p = 1e-3 * np.ones(n)
    p_dot = 1e-3 * np.ones(n)
    # A state that puts the slider firmly in contact (lateral offset > clearance).
    state = ProjectileState(np.array([3.0, 0.02, -0.008, 0.05, -0.03, 0.02]),
                            np.array([0.0, 0.4, 0.3, 0.1, -0.2, 0.15]))
    slider = Slider(position=(0.5, 0.0, 0.0), radius=0.0)

    kin_i = ProjectileKinematics(field_i).evaluate(state, p, p_dot)
    kin_r = ProjectileKinematics(field_r).evaluate(state, p, p_dot)

    sc_i = evaluate_slider(kin_i, field_i, I3, p, p_dot, slider, _rail(), x_r=3.0, l_c=100.0)
    sc_r = evaluate_slider(kin_r, field_r, R, p, p_dot, slider, _rail(), x_r=3.0, l_c=100.0)

    assert sc_i.in_phase and sc_r.in_phase
    assert len(sc_i.surfaces) > 0

    # --- Scalars are frame invariant ---
    assert np.isclose(sc_r.x_r_i, sc_i.x_r_i, atol=1e-9)
    pen_i = {s.label: s.penetration for s in sc_i.surfaces}
    pen_r = {s.label: s.penetration for s in sc_r.surfaces}
    assert pen_i.keys() == pen_r.keys()
    for k in pen_i:
        assert np.isclose(pen_i[k], pen_r[k], atol=1e-12)
    for k in sc_i.penetration_velocities:
        assert np.isclose(sc_i.penetration_velocities[k],
                          sc_r.penetration_velocities[k], atol=1e-10)

    # --- Resultants rotate with the guide ---
    assert np.allclose(sc_r.i_q_o1, R @ sc_i.i_q_o1, atol=1e-9)
    assert np.allclose(sc_r.i_m_o1, R @ sc_i.i_m_o1, atol=1e-9)
    assert np.allclose(sc_r.i_q_pi, R @ sc_i.i_q_pi, atol=1e-9)
    assert np.allclose(sc_r.i_m_pi, R @ sc_i.i_m_pi, atol=1e-9)
