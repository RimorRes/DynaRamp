from __future__ import annotations
import logging

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Hashable

import numpy as np

from dynaramp.vecmath import skew_sym_mat


logger = logging.getLogger(__name__)

type ElemLike = Hashable | Element


@dataclass
class Element:
    e_id: Hashable
    U: np.ndarray
    slots: Dict[Hashable, np.ndarray]  # positions of slots relative to the main input

    H_ext: np.ndarray = field(init=False)
    U_exts: Dict[Hashable, np.ndarray] = field(init=False)
    H_incs: Dict[Hashable, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.U_exts = {}
        self.H_incs = {}
        # Auto-generate the extraction and incidence matrices
        for s in self.slots:
            r = self.slots[s]  # position of slot relative to the main input
            # H extraction
            self.H_ext = np.zeros((6, 12))
            self.H_ext[:, :6] = np.identity(6)
            # U extraction
            self.U_exts[s] = np.zeros((12, 12))
            ublock = np.identity(6)
            ublock[3:, :3] = skew_sym_mat(r)
            self.U_exts[s][6:, 6:] = ublock
            # H incidence
            self.H_incs[s] = np.zeros((6, 12))
            hblock = np.identity(6)
            hblock[:3, 3:] = skew_sym_mat(r)
            self.H_incs[s][:, :6] = hblock


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



