from __future__ import annotations
import logging

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set

import numpy as np

from dynaramp.vecmath import skew_sym_mat


logger = logging.getLogger(__name__)

type ElemLike = int | Element


class ElementType(Enum):
    BODY = auto()
    HINGE = auto()


@dataclass
class Element:
    eid: int
    etype: ElementType
    U: np.ndarray


@dataclass
class ComplexElement(Element):
    slots_pos: Dict[int, np.ndarray]  # positions of slots relative to the main input
    occ_slots: Set[int]  # slot occupancy
    H_ext: np.ndarray = field(init=False)
    U_exts: Dict[int, np.ndarray] = field(init=False)
    H_incs: Dict[int, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.occ_slots = set()
        self.U_exts = {}
        self.H_incs = {}
        # Auto-generate the extraction and incidence matrices
        for s in self.slots_pos:
            r = self.slots_pos[s]  # position of slot relative to the main input
            # H extraction
            self.H_ext = np.zeros((6, 12))
            self.H_ext[:, :6] = np.identity(6)
            # U extraction
            self.U_exts[s] = np.zeros((12,12))
            ublock = np.identity(6)
            ublock[3:, :3] = skew_sym_mat(r)
            self.U_exts[s][6:, 6:] = ublock
            # H incidence
            self.H_incs[s] = np.zeros((6, 12))
            hblock = np.identity(6)
            hblock[:3, 3:] = skew_sym_mat(r)
            self.H_incs[s][:, :6] = hblock


@dataclass
class Link:
    # For internal use only
    source: int
    target: int
    # Slot number for multi-input elements
    slot: int | None = None  # Leaving slot unspecified treats the target as a single-input element


@dataclass
class Boundary:
    eid: int
    free_dofs: List[int]
    fixed_dofs: List[int]
