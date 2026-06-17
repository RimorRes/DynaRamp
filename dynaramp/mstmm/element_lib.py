from __future__ import annotations
import logging

from typing import Tuple, Dict, Hashable

import numpy as np

from .structs import Element
from dynaramp.vecmath import skew_sym_mat

logger = logging.getLogger(__name__)


class RigidBody(Element):

    def __init__(self,
                 e_id,
                 mass: float,
                 inertia: np.ndarray,
                 com: Tuple[float, float, float],
                 slot_coords: Dict[Hashable, Tuple[float, float, float]],
                 ):
        slots_pos = {s_id: np.array(pos) for s_id, pos in slot_coords.items()}
        super().__init__(e_id, slots_pos)

        self.mass = mass
        self.inertia = inertia
        self.com_pos = np.array(com)
        r = - self.com_pos
        self.j = inertia + mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))

    def u(self, omega: float) -> np.ndarray:
        output_slot = next(
            slot
            for slot, value in self.slot_occupancy.items()
            if value == "output"
        )
        out_pos = self.slots_pos[output_slot]

        l_io = skew_sym_mat(out_pos)
        l_ic = skew_sym_mat(self.com_pos)
        l_co = l_io - l_ic

        u = np.block([
            [np.identity(3), -l_io, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.identity(3), np.zeros((3, 3)), np.zeros((3, 3))],
            [self.mass * (omega**2) * l_co, -(omega**2) * (self.mass * l_io @ l_ic + self.j), np.identity(3), l_io],
            [self.mass * (omega**2) * np.identity(3), -self.mass * (omega**2) * l_ic, np.zeros((3, 3)), np.identity(3)],
        ])
        return u
