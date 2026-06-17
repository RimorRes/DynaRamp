from __future__ import annotations
import logging

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Hashable

import numpy as np

from dynaramp.vecmath import skew_sym_mat


logger = logging.getLogger(__name__)

type ElemLike = Hashable | Element


@dataclass
class Element(ABC):
    e_id: Hashable
    slots_pos: Dict[Hashable, np.ndarray]  # positions of slots relative to the main input

    slot_occupancy: Dict[Hashable, str | None] = field(init=False)  # 'input', 'output' or None
    h_ext: np.ndarray = field(init=False)
    u_exts: Dict[Hashable, np.ndarray] = field(init=False)
    h_incs: Dict[Hashable, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.slot_occupancy = {None: None} # init `None` a.k.a `MAIN` slot
        self.u_exts = {}
        self.h_incs = {}
        # Auto-generate the extraction and incidence matrices
        for s in self.slots_pos:
            self.slot_occupancy[s] = None

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
    def u(self, omega: float) -> np.ndarray:
        pass


@dataclass
class Boundary:
    b_id: Hashable
    state_vector: np.ndarray  # Numerical value for known boundary value, None for unknown

@dataclass
class CutPoint:
    b_id1: Hashable
    b_id2: Hashable
    sign_matrix: bool = True
    mat: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mat = np.identity(12)
        if self.sign_matrix:
            self.mat[6:, 6:] *= -1



