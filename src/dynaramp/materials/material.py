from __future__ import annotations
import logging

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Material:
    """
    Isotropic linear-elastic material.

    Attributes
    ----------
    youngs_modulus : float
        Young's modulus E [Pa].
    poisson_ratio : float
        Poisson's ratio nu [-].
    density : float | None
        Mass density rho [kg/m^3], optional (useful when a Material also backs a
        structural element such as a beam).
    name : str | None
        Optional human-readable label.
    """
    youngs_modulus: float
    poisson_ratio: float
    density: float | None = None
    name: str | None = None

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G = E / (2 (1 + nu))."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def contact_sigma(self) -> float:
        """
        Hertzian contact material parameter sigma = (1 - nu^2) / E (Eq. B14),
        used to build the generalized contact stiffness K (Eq. B13).
        """
        return (1.0 - self.poisson_ratio ** 2) / self.youngs_modulus


# Common presets (nominal room-temperature values).
STEEL = Material(youngs_modulus=210e9, poisson_ratio=0.30, density=7850.0, name="steel")
ALUMINIUM = Material(youngs_modulus=69e9, poisson_ratio=0.33, density=2700.0, name="aluminium")
