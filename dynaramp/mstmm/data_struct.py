from __future__ import annotations
import logging

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Hashable

import numpy as np

from dynaramp.vecmath import skew_sym_mat


logger = logging.getLogger(__name__)

type ElemLike = Hashable | Element


class ElementType(Enum):
    BODY = auto()
    HINGE = auto()


@dataclass
class Element:
    e_id: Hashable
    etype: ElementType
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
            self.H_ext = np.block([
                np.identity(6), np.zeros((6, 6))
             ])
            # U extraction
            ublock = np.block([
                [np.identity(3), np.zeros((3, 3))],
                [skew_sym_mat(r), np.identity(3)]
            ])
            self.U_exts[s] = np.block([
                [np.zeros((6, 6)), np.zeros((6, 6))],
                [np.zeros((6, 6)), ublock]
            ])
            # H incidence
            hblock = np.block([
                [np.identity(3), skew_sym_mat(r)],
                [np.zeros((3, 3)), np.identity(3)]
            ])
            self.H_incs[s] = np.block([
                hblock, np.zeros((6, 6))
            ])


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



