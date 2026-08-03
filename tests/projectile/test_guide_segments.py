import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, GuideSegment


def _beam(e_id, length):
    h, b = 1, 2
    return dyn.EulerBernoulliBeam(
        e_id=e_id, length=length, density=7800, youngs_mod=210e9,
        shear_mod=80e9, area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12,
    )


TIP = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
ROOT = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])


def _single(length):
    topo = dyn.TopologyHandler()
    beam = _beam("beam", length)
    topo.add_elements(beam)
    topo.add_root(beam, ROOT, output_pos=(length, 0, 0))
    topo.add_tip(beam, TIP, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


def _two_segment(l1, l2):
    topo = dyn.TopologyHandler()
    seg1, seg2 = _beam("seg1", l1), _beam("seg2", l2)
    topo.add_elements([seg1, seg2])
    topo.connect_elements(seg1, seg2, src_pos=(l1, 0, 0), dst_pos=(0, 0, 0))
    topo.add_tip(seg1, TIP, input_pos=(0, 0, 0))
    topo.add_root(seg2, ROOT, output_pos=(l2, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


L1, L2 = 4.0, 6.0
LTOT = L1 + L2


@pytest.fixture(scope="module")
def systems():
    single = _single(LTOT)
    two = _two_segment(L1, L2)
    modes_single = single.natural_modes(3, omega_max=300, search_res=3000)
    modes_two = two.natural_modes(3, omega_max=300, search_res=3000)
    assert len(modes_single) >= 1 and len(modes_two) >= 1
    return single, modes_single, two, modes_two


def test_resolve_and_total_length(systems):
    _, _, two, modes_two = systems
    field = GuideModalField(two, [GuideSegment("seg1", L1), GuideSegment("seg2", L2)], modes_two)
    assert np.isclose(field.total_length, LTOT)
    assert field._resolve(2.0) == (0, 2.0)          # inside seg1
    assert field._resolve(4.0) == (0, 4.0)          # junction -> end of seg1
    assert field._resolve(5.0) == (1, 1.0)          # inside seg2
    assert field._resolve(-0.5) == (0, -0.5)        # clamp below -> seg1
    assert field._resolve(11.0) == (1, 7.0)         # clamp above -> seg2 (local = 11 - 4)


def test_length_read_from_element(systems):
    # Segments given as bare ids should pull their length from the beam element.
    _, _, two, modes_two = systems
    field = GuideModalField.from_elements(two, ["seg1", "seg2"], modes_two)
    assert np.isclose(field.total_length, LTOT)
    assert np.allclose(field._lengths, [L1, L2])


def test_multisegment_matches_single_segment_field(systems):
    # Within the SAME system (same modes/normalization), the multi-segment field must
    # equal a field bound to just the relevant segment, evaluated at the local offset.
    _, _, two, modes_two = systems
    full = GuideModalField(two, [GuideSegment("seg1", L1), GuideSegment("seg2", L2)], modes_two)
    seg1_only = GuideModalField.from_elements(two, ["seg1"], modes_two)
    seg2_only = GuideModalField.from_elements(two, ["seg2"], modes_two)

    # deep inside seg1
    a = full.evaluate(1.5)
    b = seg1_only.evaluate(1.5)
    # deep inside seg2 (global 4 + 2 = 6)
    c = full.evaluate(L1 + 2.0)
    d = seg2_only.evaluate(2.0)
    for attr in ("phi_r", "phi_theta", "phi_r_d1", "phi_r_d2", "phi_theta_d1", "phi_theta_d2"):
        assert np.allclose(getattr(a, attr), getattr(b, attr))
        assert np.allclose(getattr(c, attr), getattr(d, attr))


def test_continuity_across_junction(systems):
    _, _, two, modes_two = systems
    full = GuideModalField(two, [GuideSegment("seg1", L1), GuideSegment("seg2", L2)], modes_two)
    pr_lo, pt_lo = full.phi(L1 - 1e-7)
    pr_hi, pt_hi = full.phi(L1 + 1e-7)
    assert np.allclose(pr_lo, pr_hi, atol=1e-6)
    assert np.allclose(pt_lo, pt_hi, atol=1e-6)


def test_split_beam_frequencies_match_whole(systems):
    # A beam split into two rigidly-connected collinear segments has the same natural
    # frequencies as the equivalent single beam (MSTMM series composition).
    _, modes_single, _, modes_two = systems
    f_single = sorted(m.frequency for m in modes_single)
    f_two = sorted(m.frequency for m in modes_two)
    k = min(len(f_single), len(f_two))
    assert np.allclose(f_single[:k], f_two[:k], rtol=1e-3)


def test_single_segment_backward_compatible(systems):
    single, modes_single, _, _ = systems
    field = GuideModalField.from_elements(single, ["beam"], modes_single)
    assert np.isclose(field.total_length, LTOT)
    ms = field.evaluate(3.7)
    assert ms.phi_r.shape == (3, len(modes_single))
