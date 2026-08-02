import numpy as np
from scipy.linalg import null_space
import dynaramp.mstmm as dyn


def create_cantilever_beam():
    topo = dyn.TopologyHandler()

    length = 10
    h = 1
    b = 2
    i_y = b*(h**3)/12
    i_z = (b**3)*h/12
    beam_elem = dyn.EulerBernoulliBeam(
        e_id='beam',
        length=length,
        density=7800,
        youngs_mod=210e9,
        shear_mod=80e9,
        area=b*h,
        i_y=i_y,
        i_z=i_z
    )

    topo.add_elements(beam_elem)

    # z = [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y, Q_z, 1]
    tip_boundary = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root_boundary = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])

    topo.add_root(beam_elem, root_boundary, output_pos=(length, 0, 0))
    topo.add_tip(beam_elem, tip_boundary, input_pos=(0, 0, 0))

    topo.make_tree()
    system = dyn.System(topo)

    return system


def test_cantilever_beam():
    cant_beam = create_cantilever_beam()
    modes = cant_beam.natural_modes(7)

    for w, shape in modes:
        u, f, z_merged, rem_bounds = cant_beam.overall_transfer_mat(w)
        assert np.allclose(f, 0)

        z_red = null_space(u, rcond=1e-8).reshape(12)
        z_merged_homogenous = np.array([i if i is None else 0.0 for i in z_merged])

        bound_svs = cant_beam.reconstruct_boundary_states(z_red, z_merged_homogenous, rem_bounds)

        assert np.allclose(np.abs(bound_svs[cant_beam.topology.root.b_id]), np.abs(shape[cant_beam.topology.root.b_id]))
