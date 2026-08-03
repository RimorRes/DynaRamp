import numpy as np

from dynaramp.materials import Material, STEEL, ALUMINIUM


def test_contact_sigma_formula():
    m = Material(youngs_modulus=200e9, poisson_ratio=0.3)
    assert np.isclose(m.contact_sigma, (1 - 0.3 ** 2) / 200e9)


def test_shear_modulus_formula():
    m = Material(youngs_modulus=210e9, poisson_ratio=0.3)
    assert np.isclose(m.shear_modulus, 210e9 / (2 * (1 + 0.3)))


def test_presets_reasonable():
    for mat in (STEEL, ALUMINIUM):
        assert mat.youngs_modulus > 0
        assert 0 < mat.poisson_ratio < 0.5
        assert mat.density and mat.density > 0
        assert mat.contact_sigma > 0
    # Aluminum is more compliant than steel (larger sigma).
    assert ALUMINIUM.contact_sigma > STEEL.contact_sigma
