import numpy as np
from scipy.spatial.transform import Rotation

import dynaramp.mstmm as dyn
from dynaramp.common.vecmath import block_rotation


def test_identity_gives_identity():
    assert np.allclose(block_rotation(np.identity(3)), np.identity(12))


def test_block_structure():
    d = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
    h = block_rotation(d)
    assert h.shape == (12, 12)
    for i in range(4):
        assert np.allclose(h[3 * i:3 * i + 3, 3 * i:3 * i + 3], d)     # diagonal blocks = D
        for j in range(4):
            if i != j:
                assert np.allclose(h[3 * i:3 * i + 3, 3 * j:3 * j + 3], 0.0)  # off-diagonal zero


def test_transforms_each_state_subblock():
    d = Rotation.from_euler("xyz", [0.1, 0.4, -0.2]).as_matrix()
    rng = np.random.default_rng(0)
    r, th, m, q = (rng.standard_normal(3) for _ in range(4))
    z = np.concatenate([r, th, m, q])
    assert np.allclose(block_rotation(d) @ z, np.concatenate([d @ r, d @ th, d @ m, d @ q]))


def test_orthogonal():
    h = block_rotation(Rotation.from_euler("zyx", [0.5, -0.3, 0.7]).as_matrix())
    assert np.allclose(h @ h.T, np.identity(12))


def test_composition_is_matrix_product():
    d1 = Rotation.from_euler("xyz", [0.2, 0.3, -0.1]).as_matrix()
    d2 = Rotation.from_euler("zyx", [0.4, -0.2, 0.1]).as_matrix()
    assert np.allclose(block_rotation(d1 @ d2), block_rotation(d1) @ block_rotation(d2))


# --------------------------------------------------------------------------- #
# Element-level: U_global = H @ U_local @ H^T (Rui Eq. 15.103)
# --------------------------------------------------------------------------- #
def _beam(orientation=None):
    h, b = 1.0, 2.0
    return dyn.EulerBernoulliBeam(
        e_id="beam", length=10.0, density=7800.0, youngs_mod=210e9, shear_mod=80e9,
        area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12, orientation=orientation,
    )


def _rigid_body(orientation=None):
    return dyn.RigidBody(
        e_id="body", mass=5.0, inertia=np.diag([2.0, 3.0, 4.0]),
        com=(0.3, -0.1, 0.2), orientation=orientation,
    )


def test_identity_orientation_matches_local():
    # With no orientation, u must equal the bare local transfer (bit-for-bit path).
    beam = _beam()
    u = beam.u((0, 0, 0), (10, 0, 0), 120.0)
    u_local = beam._u_local((0, 0, 0), (10, 0, 0), 120.0)
    assert np.allclose(u, u_local)


def test_rotated_beam_transfer_is_conjugated_by_H():
    d = Rotation.from_euler("zyx", [0.5, -0.3, 0.2]).as_matrix()
    h = block_rotation(d)
    in_pos, out_pos, omega = (0, 0, 0), (10, 0, 0), 95.0
    u_axis = _beam().u(in_pos, out_pos, omega)
    u_rot = _beam(orientation=d).u(in_pos, out_pos, omega)
    assert np.allclose(u_rot, h @ u_axis @ h.T)


def test_rotated_rigid_body_transfer_is_conjugated_by_H():
    d = Rotation.from_euler("xyz", [0.2, 0.4, -0.6]).as_matrix()
    h = block_rotation(d)
    in_pos, out_pos, omega = (0, 0, 0), (0.4, 0.0, 0.0), 30.0
    u_axis = _rigid_body().u(in_pos, out_pos, omega)
    u_rot = _rigid_body(orientation=d).u(in_pos, out_pos, omega)
    assert np.allclose(u_rot, h @ u_axis @ h.T)


def test_beam_modal_mass_invariant_under_rotation():
    # Modal mass is an intrinsic scalar: rotating the element AND its input state by
    # H leaves the quadratic form <v, M v> unchanged.
    d = Rotation.from_euler("zyx", [0.7, 0.1, -0.4]).as_matrix()
    h = block_rotation(d)
    rng = np.random.default_rng(3)
    z0 = rng.standard_normal(12)
    omega = 140.0
    mm_axis = _beam().modal_mass((0, 0, 0), (10, 0, 0), omega, z0)
    mm_rot = _beam(orientation=d).modal_mass((0, 0, 0), (10, 0, 0), omega, h @ z0)
    assert np.isclose(mm_axis, mm_rot)


# --------------------------------------------------------------------------- #
# System-level: reorienting a cantilever leaves its spectrum invariant
# --------------------------------------------------------------------------- #
def _cantilever(orientation=None):
    topo = dyn.TopologyHandler()
    length, h, b = 10.0, 1.0, 2.0
    beam = dyn.EulerBernoulliBeam(
        e_id="beam", length=length, density=7800.0, youngs_mod=210e9, shear_mod=80e9,
        area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12, orientation=orientation,
    )
    topo.add_elements(beam)
    # Clamped-free BCs are isotropic (all kinematics fixed at the tip, all loads
    # free at the root), hence invariant under the block rotation H.
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


def test_reoriented_cantilever_reproduces_frequencies():
    axis_modes = _cantilever().natural_modes(6)
    # Arbitrary rigid reorientation of the whole beam (axial x -> some global dir).
    d = Rotation.from_euler("zyx", [0.6, -0.25, 0.4]).as_matrix()
    rot_modes = _cantilever(orientation=d).natural_modes(6)

    axis_f = np.sort([m.frequency for m in axis_modes])
    rot_f = np.sort([m.frequency for m in rot_modes])
    assert len(axis_f) == len(rot_f) and len(axis_f) >= 1
    assert np.allclose(axis_f, rot_f, rtol=1e-4)


def test_axial_rotation_leaves_symmetric_beam_spectrum_invariant():
    # A 90 deg spin of the beam about global x maps the guide onto global +y.
    d = Rotation.from_euler("z", np.pi / 2).as_matrix()
    axis_f = np.sort([m.frequency for m in _cantilever().natural_modes(5)])
    rot_f = np.sort([m.frequency for m in _cantilever(orientation=d).natural_modes(5)])
    assert np.allclose(axis_f, rot_f, rtol=1e-4)
