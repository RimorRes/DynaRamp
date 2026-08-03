from __future__ import annotations
import logging

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..common.types import VectorLike, Matrix

logger = logging.getLogger(__name__)


@dataclass
class Slider:
    """
    A slider on the missile: the spherical ball head that rides inside a guide.
    Used by the section 4 slider-guide contact model.

    Attributes
    ----------
    position : VectorLike
        B_r_O1Vi, the center Vi of the ball head relative to O1 (rear slider pair center),
        expressed in the body frame K_B. Constant.
    radius : float
        Ri, the radius of the ball head.
    """
    position: VectorLike
    radius: float

    def __post_init__(self) -> None:
        self.position = np.array(self.position, dtype=np.float64).reshape(3)
        self.radius = float(self.radius)


@dataclass
class Missile:
    """
    Rigid-body parameters of the missile (section 3.3).

    The missile body frame K_B has its origin at O1, the center of the rear slider pair,
    with x along the missile symmetry axis toward the head. Inertia is supplied about the
    center of mass and shifted to O1 on demand (parallel-axis), mirroring the convention
    used by ``mstmm.element_lib.RigidBody``.

    Attributes
    ----------
    mass : float
        Total mass m of the missile.
    inertia_com : Matrix
        3x3 inertia tensor about the center of mass C, in the body frame K_B.
    com_o1 : VectorLike
        B_r_O1C, position of the center of mass C relative to O1, in K_B. Defaults to O1.
    sliders : List[Slider]
        Sliders distributed on the missile (front/rear pairs, optionally a middle pair),
        consumed by the section 4 contact model.
    """
    mass: float
    inertia_com: Matrix
    com_o1: VectorLike = (0.0, 0.0, 0.0)
    sliders: List[Slider] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mass = float(self.mass)
        self.inertia_com = np.array(self.inertia_com, dtype=np.float64).reshape(3, 3)
        self.com_o1 = np.array(self.com_o1, dtype=np.float64).reshape(3)

    @property
    def inertia_o1(self) -> Matrix:
        """
        Body-frame inertia tensor about O1, obtained from ``inertia_com`` by the
        parallel-axis theorem: I_O1 = I_C + m (|r|^2 I3 - r r^T), with r = B_r_O1C.
        """
        r = self.com_o1
        shift = self.mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
        return (self.inertia_com + shift).astype(np.float64)
