import numpy as np
import pytest

from dynaramp.materials import Material, STEEL
from dynaramp.contact import hertz_stiffness, normal_force, friction_force


# --------------------------------------------------------------------------- #
# Hertz stiffness (Eqs. B13-B14)
# --------------------------------------------------------------------------- #
def test_hertz_stiffness_formula():
    a = Material(200e9, 0.3)
    b = Material(200e9, 0.3)
    sigma = a.contact_sigma + b.contact_sigma
    r = 0.02
    assert np.isclose(hertz_stiffness(r, a, b), 4.0 / (3.0 * sigma) * np.sqrt(r))


def test_hertz_stiffness_scales_as_sqrt_radius():
    k1 = hertz_stiffness(0.01, STEEL, STEEL)
    k4 = hertz_stiffness(0.04, STEEL, STEEL)
    assert np.isclose(k4 / k1, 2.0)  # sqrt(0.04/0.01) = 2


def test_hertz_softer_material_gives_lower_stiffness():
    steel = Material(210e9, 0.30)
    alu = Material(69e9, 0.33)
    assert hertz_stiffness(0.02, alu, alu) < hertz_stiffness(0.02, steel, steel)


def test_hertz_radius_nonpositive_raises():
    with pytest.raises(ValueError):
        hertz_stiffness(0.0, STEEL, STEEL)


# --------------------------------------------------------------------------- #
# Normal force (Lankarani-Nikravesh, Eq. B12)
# --------------------------------------------------------------------------- #
def test_no_force_without_penetration():
    assert normal_force(0.0, 1.0, 1.0, 1e9, 0.8) == 0.0
    assert normal_force(-1e-3, 1.0, 1.0, 1e9, 0.8) == 0.0


def test_elastic_is_pure_hertz():
    # e = 1 -> damping term vanishes; force is K delta^1.5 regardless of velocity sign.
    k, d = 1e9, 1e-4
    assert np.isclose(normal_force(d, 5.0, 5.0, k, 1.0), k * d ** 1.5)
    assert np.isclose(normal_force(d, -5.0, 5.0, k, 1.0), k * d ** 1.5)


def test_onset_value_matches_formula():
    k, d, v, e = 1e9, 1e-4, 3.0, 0.7
    expected = k * d ** 1.5 * (1.0 + 3.0 * (1.0 - e ** 2) / 4.0)  # delta_dot == impact_velocity
    assert np.isclose(normal_force(d, v, v, k, e), expected)


def test_hysteresis_dissipates_energy_for_inelastic():
    # At equal penetration and |velocity|, loading (approach) force exceeds unloading
    # (restitution) force when e < 1 -> a hysteresis loop -> energy dissipated.
    k, d, v = 1e9, 1e-4, 3.0
    f_app = normal_force(d, v, v, k, 0.6)
    f_res = normal_force(d, -v, v, k, 0.6)
    assert f_app > f_res

    def gap(e):
        return normal_force(d, v, v, k, e) - normal_force(d, -v, v, k, e)

    assert gap(0.4) > gap(0.9)  # more dissipation for lower restitution


def test_force_never_negative():
    # A large negative penetration velocity during restitution must not create adhesion.
    assert normal_force(1e-4, -1000.0, 0.1, 1e9, 0.5) == 0.0


def test_zero_impact_velocity_drops_damping():
    k, d = 1e9, 1e-4
    assert np.isclose(normal_force(d, 5.0, 0.0, k, 0.5), k * d ** 1.5)


def test_restitution_out_of_range_raises():
    with pytest.raises(ValueError):
        normal_force(1e-4, 1.0, 1.0, 1e9, 1.5)


# --------------------------------------------------------------------------- #
# Coulomb friction (Eq. B15)
# --------------------------------------------------------------------------- #
def test_friction_is_mu_times_normal():
    assert np.isclose(friction_force(100.0, 0.2), 20.0)
    assert friction_force(0.0, 0.3) == 0.0
