from __future__ import annotations
import logging

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from common.vecmath import skew_sym_mat
from common.types import EntityID, Vector, VectorLike, Matrix


logger = logging.getLogger(__name__)

# Constants
NULL_SV = np.r_[np.full(12, None), 1]
NULL_SV.setflags(write=False)

type ElemLike = EntityID | Element

# TODO: QoL: better slot definition (main slot and relative coords) and add copy method
# TODO: Constructor using geometry primitives
# TODO: Add support for rotations


class Element(ABC):

    def __init__(self, e_id: EntityID):
        self.e_id = e_id

        self._applied_loads = []

        # Caches
        self._h_cache: Dict[bytes, Matrix] = {}
        self._u_ext_cache: Dict[bytes, Matrix] = {}

    def u(self, input_pos: VectorLike, output_pos: VectorLike, omega: float) -> Matrix:
        """
        Returns the extended state transfer matrix for the element.
        :param input_pos: Reference input position.
        :param output_pos: Position of the output slot relative to the main input
        :param omega: vibration frequency (rad/s)
        :return: Transfer matrix (13x13)
        """
        # Get the base 12x12 transfer matrix
        input_pos_arr = np.array(input_pos)
        output_pos_arr = np.array(output_pos)  # Expensive array-cast outside the loop
        # Gurantee that _u gets fed Vectors, saves on casts
        u_base = self._u(input_pos_arr, output_pos_arr, omega)

        # Calculate the total external load vector directly at the output
        f_load_at_out = np.zeros(12)

        for load in self._applied_loads:
            if load['type'] == 'force':
                f = load['force']
                p = load['point']

                # The lever arm FROM the application point P TO the output O
                r_po = output_pos_arr - p
                m = np.cross(r_po, f)

                # Add to the 12x1 load vector (indices 6:9 for moments, 9:12 for forces)
                f_load_at_out[6:9] += m
                f_load_at_out[9:12] += f
            elif load['type'] == 'torque':
                t = load['torque']
                f_load_at_out[6:9] += t

        # Assemble the 13x13 extended matrix
        u_extend = np.block([
            [u_base, f_load_at_out.reshape(12, 1)],
            [np.zeros((1, 12)), 1]
        ])
        return u_extend

    @abstractmethod
    def _u(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        pass

    def u_ext(self, input_pos: VectorLike, output_pos: VectorLike) -> Matrix:
        # TODO: Quantization of floating-point positions could help avoid cache misses
        input_pos_arr = np.array(input_pos)
        output_pos_arr = np.array(output_pos)
        signature = input_pos_arr.tobytes() + output_pos_arr.tobytes()
        # Try to hit the cache
        if signature in self._u_ext_cache:
            return self._u_ext_cache[signature]

        # The vector FROM the auxiliary input TO the output
        r_io = np.array(output_pos) - np.array(input_pos)

        transform = np.block([  # Transform force and moments from Ik to O
            [np.identity(3), skew_sym_mat(r_io)],  # <-- positive sign on the skew!
            [np.zeros((3, 3)), np.identity(3)]
        ])

        u_ext_extend = np.block([
            [np.zeros((6, 13))],
            [np.zeros((6, 6)), transform, np.zeros((6, 1))],
            [np.zeros((1, 13))],
        ])

        self._u_ext_cache[signature] = u_ext_extend
        return u_ext_extend

    def h(self, ref_pos: VectorLike, input_pos: VectorLike) -> Matrix:
        """
        Calculates the 6x13 geometric incidence matrix mapping the 13x1 state
        vector at ref_pos to the 6x1 kinematic state vector at target_pos.
        :param ref_pos: Reference input position.
        :param input_pos: Input position.
        """
        input_pos_arr = np.array(input_pos)
        signature = input_pos_arr.tobytes()
        # Try to hit the cache
        if signature in self._h_cache:
            return self._h_cache[signature]

        # The vector FROM the reference point TO the target point
        r_ab = input_pos_arr - np.array(ref_pos)

        # Kinematic transform (Notice the NEGATIVE skew matrix)
        transform = np.block([
            [np.identity(3), -skew_sym_mat(r_ab)],
            [np.zeros((3, 3)), np.identity(3)]
        ])

        # Pad to 6x13 to map from a full [Z_all] state vector
        h_matrix = np.block([
            transform, np.zeros((6, 7))
        ])

        self._h_cache[signature] = h_matrix
        return h_matrix

    def apply_force(self, force: VectorLike, point: VectorLike) -> None:
        """
        Store the force and its absolute application point.
        :param force:
        :param point:
        """
        self._applied_loads.append({
            'type': 'force',
            'force': np.array(force),
            'point': np.array(point)
        })

    def apply_torque(self, torque: VectorLike, point: VectorLike) -> None:
        """
        Apply a pure moment to the element
        :param torque:
        :param point:
        :return:
        """
        self._applied_loads.append({
            'type': 'torque',
            'torque': np.array(torque),
            'point': np.array(point),  # Point is needed for the sweep evaluation of continuous elements
        })


@dataclass
class Boundary:
    b_id: EntityID
    # Numerical value for known boundary value, None for unknown
    state_vector: VectorLike


@dataclass
class CutPoint:
    b_id1: EntityID
    b_id2: EntityID
    sign_mat_flag: bool = True
    mat: Matrix = field(init=False)

    def __post_init__(self):
        self.mat = np.identity(13)
        if self.sign_mat_flag:
            self.mat[6:12, 6:12] *= -1
