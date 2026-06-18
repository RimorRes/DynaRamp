from __future__ import annotations
import logging

from typing import Dict

import numpy as np

from .structs import Element
from ..vecmath import skew_sym_mat
from ..common_types import EntityID, Vector, Matrix

logger = logging.getLogger(__name__)


class RigidBody(Element):

    def __init__(self,
                 e_id,
                 mass: float,
                 inertia: Matrix,
                 com: Vector,
                 slot_coords: Dict[EntityID, Vector],
                 ):
        slots_pos = {s_id: np.array(pos) for s_id, pos in slot_coords.items()}
        super().__init__(e_id, slots_pos)

        self.mass = mass
        self.inertia = inertia
        self.com_pos = np.array(com)
        r = - self.com_pos
        self.j = inertia + mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))

    def u(self, omega: float, output_pos: Vector) -> Matrix:
        l_io = skew_sym_mat(output_pos)
        l_ic = skew_sym_mat(self.com_pos)
        l_co = l_io - l_ic

        u = np.block([
            [np.identity(3), -l_io, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.identity(3), np.zeros((3, 3)), np.zeros((3, 3))],
            [self.mass * (omega**2) * l_co, -(omega**2) * (self.mass * l_io @ l_ic + self.j), np.identity(3), l_io],
            [self.mass * (omega**2) * np.identity(3), -self.mass * (omega**2) * l_ic, np.zeros((3, 3)), np.identity(3)],
        ])
        return u
