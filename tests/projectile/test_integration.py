import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import (
    GuideModalField,
    ProjectileKinematics,
    ProjectileEOM,
    Projectile,
    Slider,
    ProjectileState,
)


def _build_canister():
    """A single Euler-Bernoulli beam standing in for the launch canister."""
    topo = dyn.TopologyHandler()
    length = 6.0
    h, b = 0.3, 0.3
    i_y = b * (h ** 3) / 12
    i_z = (b ** 3) * h / 12
    beam = dyn.EulerBernoulliBeam(
        e_id="canister", length=length, density=7800, youngs_mod=210e9,
        shear_mod=80e9, area=b * h, i_y=i_y, i_z=i_z,
    )
    topo.add_elements(beam)
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo), "canister"


@pytest.fixture(scope="module")
def pipeline():
    system, elem = _build_canister()
    modes = system.natural_modes(3, omega_max=400, search_res=4000)
    assert len(modes) >= 1
    field = GuideModalField.from_elements(system, [elem], modes)
    projectile = Projectile(
        mass=250.0,
        inertia_com=np.diag([3.0, 90.0, 90.0]),
        com_o1=(1.5, 0.0, 0.0),               # COM 1.5 m ahead of the rear slider pair
        sliders=[
            Slider((0.0, 0.12, 0.0), 0.02),   # rear pair
            Slider((0.0, -0.12, 0.0), 0.02),
            Slider((2.8, 0.12, 0.0), 0.02),   # front pair
            Slider((2.8, -0.12, 0.0), 0.02),
        ],
    )
    return field, projectile


def test_pipeline_runs_shapes_finite(pipeline):
    field, projectile = pipeline
    n = field.n_modes
    # A sliding projectile: 12 m/s axially, small lateral motion and attitude, small deformation.
    x = np.array([3.0, 0.02, -0.01, 0.03, -0.02, 0.01])
    x_dot = np.array([12.0, 0.10, -0.05, 0.02, 0.01, -0.03])
    st = ProjectileState.from_config_rates(x, x_dot)
    p = 1e-3 * np.linspace(1.0, 2.0, n)
    p_dot = 1e-3 * np.linspace(-1.0, 1.0, n)

    kin = ProjectileKinematics(field).evaluate(st, p, p_dot)
    eom = ProjectileEOM.assemble(kin, projectile)

    for block in (eom.m_tp, eom.m_ty, eom.m_rp, eom.m_ry, eom.h_t, eom.h_r):
        assert np.all(np.isfinite(block))
    assert eom.m_tp.shape == (3, n)
    assert eom.m_rp.shape == (3, n)
    assert eom.m_ty.shape == (3, 6)
    assert eom.m_ry.shape == (3, 6)
    assert eom.h_t.shape == (3,)
    assert eom.h_r.shape == (3,)


def test_coupled_block_row_dimensions(pipeline):
    # The two projectile block-rows of the coupled system (Eq. 67) must be 3 x (n + 6).
    field, projectile = pipeline
    n = field.n_modes
    st = ProjectileState(np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6))
    kin = ProjectileKinematics(field).evaluate(st, np.zeros(n), np.zeros(n))
    eom = ProjectileEOM.assemble(kin, projectile)

    trans_row = np.hstack([eom.m_tp, eom.m_ty])   # [M_Tp | M_Ty]
    rot_row = np.hstack([eom.m_rp, eom.m_ry])      # [M_Rp | M_Ry]
    assert trans_row.shape == (3, n + 6)
    assert rot_row.shape == (3, n + 6)
    assert np.vstack([trans_row, rot_row]).shape == (6, n + 6)


def test_full_rest_has_zero_bias(pipeline):
    # p = p_dot = 0 and y = 0: no motion => omega_IB = 0 and both bias vectors vanish.
    field, projectile = pipeline
    n = field.n_modes
    st = ProjectileState(np.array([3.0, 0.05, -0.03, 0.1, -0.05, 0.2]), np.zeros(6))
    kin = ProjectileKinematics(field).evaluate(st, np.zeros(n), np.zeros(n))
    eom = ProjectileEOM.assemble(kin, projectile)

    assert np.allclose(kin.omega_ib, 0.0, atol=1e-12)
    assert np.allclose(eom.h_t, 0.0, atol=1e-12)
    assert np.allclose(eom.h_r, 0.0, atol=1e-12)


def test_axial_slide_produces_quadratic_convective_coupling(pipeline):
    # A pure axial slide over a deformed canister (p_dot = 0) generates convective terms
    # that scale exactly as v_P'^2 (zeta_TP' ~ v^2 Phi'' p, omega_IL ~ v). This is a
    # normalization-independent signature of the convective coupling, robust to the
    # arbitrary scale of the MSTMM mode shapes.
    field, projectile = pipeline
    n = field.n_modes
    p = np.linspace(1e-3, 3e-3, n)  # nonzero canister deformation

    def zeta(v):
        x = np.array([3.0, 0.02, -0.01, 0.0, 0.0, 0.0])
        x_dot = np.array([v, 0.0, 0.0, 0.0, 0.0, 0.0])
        st = ProjectileState.from_config_rates(x, x_dot)
        return ProjectileKinematics(field).evaluate(st, p, np.zeros(n)).zeta_to1

    z1 = zeta(15.0)
    z2 = zeta(30.0)
    # Present (not identically zero) and quadratic in v: zeta(2v) == 4 zeta(v).
    assert np.linalg.norm(z1) > 0.0
    assert np.allclose(z2, 4.0 * z1, rtol=1e-9, atol=0.0)
