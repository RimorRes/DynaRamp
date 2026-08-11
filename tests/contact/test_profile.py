import numpy as np
import pytest

from dynaramp.materials import Material, STEEL
from dynaramp.contact import (
    linear_contact_stiffness,
    GuideProfile,
    RailProfile,
    CanisterProfile,
    StationVaryingProfile,
)


# --------------------------------------------------------------------------- #
# linear_contact_stiffness
# --------------------------------------------------------------------------- #
def test_linear_stiffness_formula():
    a = Material(200e9, 0.3)
    b = Material(200e9, 0.3)
    area, length = 4e-4, 2e-3
    e_star = 1.0 / (a.contact_sigma + b.contact_sigma)
    assert np.isclose(linear_contact_stiffness(area, a, b, length), e_star * area / length)


def test_linear_stiffness_scales():
    a = b = STEEL
    k1 = linear_contact_stiffness(1e-4, a, b, 1e-3)
    k2 = linear_contact_stiffness(2e-4, a, b, 1e-3)   # double area -> double K
    k3 = linear_contact_stiffness(1e-4, a, b, 2e-3)   # double length -> half K
    assert np.isclose(k2 / k1, 2.0)
    assert np.isclose(k3 / k1, 0.5)


def test_linear_stiffness_validation():
    with pytest.raises(ValueError):
        linear_contact_stiffness(0.0, STEEL, STEEL, 1e-3)
    with pytest.raises(ValueError):
        linear_contact_stiffness(1e-4, STEEL, STEEL, 0.0)


# --------------------------------------------------------------------------- #
# RailProfile (rectangular groove, flat faces)
# --------------------------------------------------------------------------- #
CLAT, CBOT, CTOP = 1e-3, 2e-3, 1.5e-3


def _rail():
    return RailProfile(CLAT, CBOT, CTOP, stiffness=1e7, restitution=0.5, friction=0.2)


def test_rail_is_a_guide_profile():
    assert isinstance(_rail(), GuideProfile)


def test_rail_no_contact_within_clearances():
    assert _rail().contacts(np.array([0.0, 0.5 * CLAT, 0.5 * CBOT])) == []    # +z within floor
    assert _rail().contacts(np.array([0.0, -0.5 * CLAT, -0.5 * CTOP])) == []  # -z within lip


def test_rail_side_faces():
    p = 3e-4
    (c,) = _rail().contacts(np.array([0.0, CLAT + p, 0.0]))
    assert c.label == "side_+y"
    assert np.isclose(c.penetration, p)
    assert np.allclose(c.normal, [0, -1, 0])
    assert c.exponent == 1.0 and c.stiffness == 1e7 and c.restitution == 0.5 and c.friction == 0.2

    (c,) = _rail().contacts(np.array([0.0, -(CLAT + p), 0.0]))
    assert c.label == "side_-y"
    assert np.allclose(c.normal, [0, 1, 0])


def test_rail_bottom_and_top_faces():
    # NED: the bottom floor is the +z ("down") face; the top lip is the -z ("up") face.
    p = 4e-4
    (c,) = _rail().contacts(np.array([0.0, 0.0, CBOT + p]))
    assert c.label == "bottom"
    assert np.isclose(c.penetration, p)
    assert np.allclose(c.normal, [0, 0, -1])   # floor reacts up (-z)

    (c,) = _rail().contacts(np.array([0.0, 0.0, -(CTOP + p)]))
    assert c.label == "top_lip"
    assert np.allclose(c.normal, [0, 0, 1])     # lip reacts down (+z)


def test_rail_corner_two_faces():
    cs = _rail().contacts(np.array([0.0, CLAT + 1e-4, CBOT + 1e-4]))
    labels = {c.label for c in cs}
    assert labels == {"side_+y", "bottom"}


# --------------------------------------------------------------------------- #
# CanisterProfile (groove: +/- y side walls, +z deep bottom wall; Fig. 4)
# --------------------------------------------------------------------------- #
CY, CZ = 1e-3, 1e-3


def _canister():
    return CanisterProfile(CY, CZ, stiffness=1e9, restitution=0.7, friction=0.15)


def test_canister_no_contact_when_centered():
    assert _canister().contacts(np.array([0.0, 0.0, 0.0])) == []


def test_canister_side_contact():
    p = 2e-4
    (c,) = _canister().contacts(np.array([0.0, CY + p, 0.0]))
    assert c.label == "side"
    assert np.isclose(c.penetration, p)
    assert np.allclose(c.normal, [0, -1, 0])
    assert c.exponent == 1.5


def test_canister_bottom_contact():
    # Slider pressed past the bottom clearance into the +z deep wall.
    p = 3e-4
    (c,) = _canister().contacts(np.array([0.0, 0.0, CZ + p]))
    assert c.label == "bottom"
    assert np.isclose(c.penetration, p)
    assert np.allclose(c.normal, [0, 0, -1])  # deep wall reacts back in -z


def test_canister_side_and_bottom_together():
    labels = {c.label for c in _canister().contacts(np.array([0.0, CY + 1e-4, CZ + 1e-4]))}
    assert labels == {"side", "bottom"}


def test_canister_open_side_no_negative_z_contact():
    # The groove is open toward the missile (-z): a slider offset in -z makes no contact.
    assert _canister().contacts(np.array([0.0, 0.0, -0.05])) == []


def test_uniform_profile_ignores_station():
    r = np.array([0.0, CLAT + 1e-4, 0.0])
    assert _rail().contacts(r, station=99.0)[0].penetration == _rail().contacts(r)[0].penetration


# --------------------------------------------------------------------------- #
# StationVaryingProfile (groove that widens along the guide)
# --------------------------------------------------------------------------- #
def test_station_varying_profile_widens_along_guide():
    # Lateral clearance grows 1 mm per unit station: a fixed offset contacts near the rear
    # but clears once the groove has widened past it.
    def profile_at(station):
        c = 1e-3 + 1e-3 * station
        return RailProfile(c, c, c, stiffness=1e7, restitution=0.5, friction=0.2)

    prof = StationVaryingProfile(profile_at)
    assert isinstance(prof, GuideProfile)
    r = np.array([0.0, 3e-3, 0.0])  # 3 mm lateral offset

    near = prof.contacts(r, station=0.0)          # clearance 1 mm -> penetration 2 mm
    assert near and np.isclose(near[0].penetration, 3e-3 - 1e-3)
    assert prof.contacts(r, station=5.0) == []    # clearance 6 mm -> cleared
