from __future__ import annotations
import logging

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List

from numpy import ndarray


logger = logging.getLogger(__name__)

type ElemLike = int | Element


class ElementType(Enum):
    BODY = auto()
    HINGE = auto()


@dataclass
class Element:
    eid: int
    etype: ElementType
    U: ndarray


@dataclass
class MultiInputElement(Element):
    H_ext: ndarray
    U_exts: Dict[int, ndarray]
    H_incs: Dict[int, ndarray]

    def __post_init__(self):
        # Ensure that U and H have the same keys
        if self.U_exts.keys() != self.H_incs.keys():
            err_msg = f"U and H must be defined for the same input slots. Got Us = {self.U_exts} and Hs = {self.H_incs}"
            logger.error(err_msg)
            raise ValueError(err_msg)


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

