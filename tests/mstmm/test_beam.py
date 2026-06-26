import numpy as np
import dynaramp.mstmm as dyn


def create_cantilever_beam():
    system = dyn.MBS()

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
        i_z=i_z,
        slot_coords={'end': (length, 0, 0)},
    )

    system.add_elements(beam_elem)

    # z = [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y, Q_z, 1]
    tip_boundary = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None, 1])
    root_boundary = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 1])

    system.add_root(root_boundary, beam_elem, 'end')
    system.add_tip(tip_boundary, beam_elem, None)

    system.make_tree()

    return system


cant_beam = create_cantilever_beam()
print(cant_beam.overall_transfer(52.665)[0].shape)
print(cant_beam.natural_modes(7))
