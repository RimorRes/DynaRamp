from __future__ import annotations
import logging

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from ..vecmath import skew_sym_mat
from ..common_types import EntityID, Vector, VectorLike, Matrix


logger = logging.getLogger(__name__)


type ElemLike = EntityID | Element


@dataclass
class Element(ABC):
    e_id: EntityID
    # positions of slots relative to the main input
    slots_pos: Dict[EntityID, Vector]

    _f_load_at_in: Vector = field(default_factory=lambda: np.zeros((12, 1)))  # Load vector transported to main input

    h_ext: Matrix = field(init=False)
    h_incs: Dict[EntityID, Matrix] = field(init=False)

    def __post_init__(self):
        self.h_incs = {}
        # Auto-generate the geometric extraction and incidence matrices (extended 13x1 state vectors)
        for s in self.slots_pos:
            r = self.slots_pos[s]  # position of slot relative to the main input
            transform = np.block([  # Transform force and moments from slot Ik to slot I1
                [np.identity(3), skew_sym_mat(r)],
                [np.zeros((3, 3)), np.identity(3)]
            ])
            # H extraction
            self.h_ext = np.block([
                np.identity(6), np.zeros((6, 7))
             ])
            # H incidence
            self.h_incs[s] = np.block([
                transform, np.zeros((6, 7))
            ])

    def u(self, output_pos: VectorLike, omega: float) -> Matrix:
        """
        Returns the extended state transfer matrix for the element.
        :param output_pos: Position of the output slot relative to the main input
        :param omega: vibration frequency (rad/s)
        :return: Transfer matrix (13x13)
        """
        # Transport external load vector from I1 to O
        r = - np.array(output_pos)
        transform = np.block([
            [np.zeros((6, 12))],
            [np.zeros((3, 6)), np.identity(3), skew_sym_mat(r)],
            [np.zeros((3, 6)), np.zeros((3, 3)), np.identity(3)]
        ])
        f = transform @ self._f_load_at_in

        u_extend = np.block([
            [self._u(output_pos, omega), f],
            [np.zeros((1, 12)), 1]
        ])
        return u_extend

    @abstractmethod
    def _u(self, output_pos: VectorLike, omega: float) -> Matrix:
        pass

    def u_ext(self, output_pos: VectorLike, slot_id: EntityID) -> Matrix:
        r = self.slots_pos[slot_id] - np.array(output_pos)
        transform = np.block([  # Transform force and moments from slot Ik to slot O
            [np.identity(3), skew_sym_mat(r)],
            [np.zeros((3, 3)), np.identity(3)]
        ])

        u_ext_extend = np.block([
            [np.zeros((6, 6)), np.zeros((6, 7))],
            [np.zeros((6, 6)), transform, np.zeros((6, 1))],
            [np.zeros((1, 12)), 1],
        ])
        return u_ext_extend

    def apply_force(self, force: VectorLike, point: VectorLike) -> None:
        """
        Apply force to the element at a point defined relatively to the main input
        :param force:
        :param point:
        :return:
        """
        r = np.array(point)
        q = np.array(force)
        m = np.cross(r, q)
        self._f_load_at_in += np.hstack((np.zeros(6), m, q)).reshape((12, 1))

    def apply_torque(self, torque: VectorLike) -> None:
        """
        Apply a pure torque to the element
        :param torque:
        :return:
        """
        m = np.array(torque)
        self._f_load_at_in += np.hstack((np.zeros(6), m, np.zeros(3))).reshape((12, 1))


@dataclass
class Boundary:
    b_id: EntityID
    state_vector: VectorLike  # Numerical value for known boundary value, None for unknown


@dataclass
class CutPoint:
    b_id1: EntityID
    b_id2: EntityID
    sign_matrix: bool = True
    mat: Matrix = field(init=False)

    def __post_init__(self):
        self.mat = np.identity(13)
        if self.sign_matrix:
            self.mat[6:12, 6:12] *= -1
