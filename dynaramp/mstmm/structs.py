from __future__ import annotations
import logging

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from ..vecmath import skew_sym_mat
from ..common_types import EntityID, Vector, Matrix


logger = logging.getLogger(__name__)


type ElemLike = EntityID | Element


@dataclass
class Element(ABC):
    e_id: EntityID
    # positions of slots relative to the main input
    slots_pos: Dict[EntityID, Vector]

    h_ext: Matrix = field(init=False)
    u_exts: Dict[EntityID, Matrix] = field(init=False)
    h_incs: Dict[EntityID, Matrix] = field(init=False)

    def __post_init__(self):
        self.u_exts = {}
        self.h_incs = {}
        # Auto-generate the extraction and incidence matrices
        for s in self.slots_pos:

            r = self.slots_pos[s]  # position of slot relative to the main input
            moment_block = np.block([
                [np.identity(3), skew_sym_mat(r)],
                [np.zeros((3, 3)), np.identity(3)]
            ])
            # H extraction
            self.h_ext = np.block([
                np.identity(6), np.zeros((6, 6))
             ])
            # U extraction
            self.u_exts[s] = np.block([
                [np.zeros((6, 6)), np.zeros((6, 6))],
                [np.zeros((6, 6)), moment_block]
            ])
            # H incidence
            self.h_incs[s] = np.block([
                moment_block, np.zeros((6, 6))
            ])

    @abstractmethod
    def u(self, omega: float, output_pos: Vector) -> Matrix:
        pass


@dataclass
class Boundary:
    b_id: EntityID
    state_vector: Vector  # Numerical value for known boundary value, None for unknown


@dataclass
class CutPoint:
    b_id1: EntityID
    b_id2: EntityID
    sign_matrix: bool = True
    mat: Matrix = field(init=False)

    def __post_init__(self):
        self.mat = np.identity(12)
        if self.sign_matrix:
            self.mat[6:, 6:] *= -1
