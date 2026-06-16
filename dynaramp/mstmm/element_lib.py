from __future__ import annotations
import logging

from typing import Tuple, List, Dict, Iterable, Hashable

import numpy as np

from .data_struct import Element
from dynaramp.vecmath import skew_sym_mat

logger = logging.getLogger(__name__)


class RigidBody(Element):

    def __init__(self,
                 mass: float,
                 inertia: np.ndarray,
                 com: Tuple[float, float, float],
                 pos_list: Iterable[Tuple[float, float, float]],
                 ):
        super().__init__()
        self.mass = mass
        self.j = inertia + mass * (np.dot(com, com) * np.identity(3) - np.outer(com, com))

    @property
    def U(self):
        return