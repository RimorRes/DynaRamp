import numpy as np
import pytest
from scipy.differentiate import derivative

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, ModalShape
from dynaramp.common.vecmath import euler_zyx


def _build_cantilever():
    topo = dyn.TopologyHandler()
    length = 10
    h, b = 1, 2
    i_y = b * (h ** 3) / 12
    i_z = (b ** 3) * h / 12
    beam = dyn.EulerBernoulliBeam(
        e_id="beam", length=length, density=7800, youngs_mod=210e9,
        shear_mod=80e9, area=b * h, i_y=i_y, i_z=i_z,
    )
    topo.add_elements(beam)
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo), "beam"


@pytest.fixture(scope="module")
def cantilever_modes():
    system, elem = _build_cantilever()
    modes = system.natural_modes(3, omega_max=300, search_res=3000)
    assert len(modes) >= 1
    return system, elem, modes


X0 = 3.7  # interior axial station on the length-10 beam


def test_modal_shape_shapes(cantilever_modes):
    system, elem, modes = cantilever_modes
    field = GuideModalField.from_elements(system, [elem], modes)
    n = len(modes)
    ms = field.evaluate(X0)
    assert isinstance(ms, ModalShape)
    assert ms.x_r == X0
    for m in (ms.phi_r, ms.phi_theta, ms.phi_r_d1, ms.phi_r_d2, ms.phi_theta_d1, ms.phi_theta_d2):
        assert m.shape == (3, n)
        assert m.dtype == np.float64


def _scalar_entry(field, block, i, j):
    """Vectorized scalar accessor for finite-difference oracles."""
    def f(x):
        x = np.asarray(x, dtype=float)
        out = np.empty(x.shape, dtype=float)
        it = np.nditer(x, flags=["multi_index"])
        for v in it:
            phi_r, phi_theta = field.phi(float(v))
            mat = phi_r if block == "r" else phi_theta
            out[it.multi_index] = mat[i, j]
        return out
    return f


def test_first_derivative_matches_scipy(cantilever_modes):
    system, elem, modes = cantilever_modes
    field = GuideModalField.from_elements(system, [elem], modes)
    ms = field.evaluate(X0)
    n = len(modes)
    for block, mine in (("r", ms.phi_r_d1), ("theta", ms.phi_theta_d1)):
        for i in range(3):
            for j in range(n):
                f = _scalar_entry(field, block, i, j)
                oracle = float(derivative(f, X0).df)
                assert np.isclose(mine[i, j], oracle, rtol=1e-5, atol=1e-8)


def test_second_derivative_matches_central_difference(cantilever_modes):
    system, elem, modes = cantilever_modes
    field = GuideModalField.from_elements(system, [elem], modes)
    ms = field.evaluate(X0)
    n = len(modes)
    h = 5e-4  # independent step, different from the implementation's 1e-4
    for block, mine in (("r", ms.phi_r_d2), ("theta", ms.phi_theta_d2)):
        for i in range(3):
            for j in range(n):
                f = _scalar_entry(field, block, i, j)
                ref = (f(X0 + h) - 2 * f(X0) + f(X0 - h)) / h ** 2
                assert np.isclose(mine[i, j], float(ref), rtol=1e-3, atol=1e-6)


def test_a_ir_projection(cantilever_modes):
    # With a non-identity A_IR, every projected block must be R @ (identity-frame block).
    system, elem, modes = cantilever_modes
    field_i = GuideModalField.from_elements(system, [elem], modes)
    r = euler_zyx(0.2, -0.3, 0.1)
    field_r = GuideModalField.from_elements(system, [elem], modes, a_ir=r)

    mi = field_i.evaluate(X0)
    mr = field_r.evaluate(X0)
    for attr in ("phi_r", "phi_theta", "phi_r_d1", "phi_r_d2", "phi_theta_d1", "phi_theta_d2"):
        assert np.allclose(getattr(mr, attr), r @ getattr(mi, attr), atol=1e-10)
