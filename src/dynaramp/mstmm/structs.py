from __future__ import annotations
import logging

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

import numpy as np
from scipy import integrate

from ..common.vecmath import skew_sym_mat
from ..common.types import EntityID, Vector, VectorLike, Matrix


logger = logging.getLogger(__name__)

# Constants
NULL_SV = np.full(12, None)
NULL_SV.setflags(write=False)


class PortType(Enum):
    INPUT = 1
    OUTPUT = 2


# Aliases
type ElemLike = EntityID | Element

# TODO: QoL: make sure copy/deepcopy work as intended
# TODO: Constructor using geometry primitives
# TODO: Add support for rotations


# Data Structures
class Element(ABC):
    # Use None as an explicit "unbounded" sentinel
    MAX_INPUTS = 1
    MAX_OUTPUTS = 1

    def __init__(self, e_id: EntityID):
        self.e_id = e_id

        # Caches
        self._h_cache: Dict[bytes, Matrix] = {}
        self._u_extract_cache: Dict[bytes, Matrix] = {}

    @abstractmethod
    def u(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        """
        Returns the extended state transfer matrix for the element.
        :param input_pos: Reference input position.
        :param output_pos: Position of the output slot relative to the main input
        :param omega: vibration frequency (rad/s)
        :return: Transfer matrix (12x12)
        """
        pass

    def u_extract(self, input_pos: VectorLike, output_pos: VectorLike) -> Matrix:
        # TODO: Quantization of floating-point positions could help avoid cache misses
        input_pos_arr = np.array(input_pos, dtype=np.float64)
        output_pos_arr = np.array(output_pos, dtype=np.float64)
        signature = input_pos_arr.tobytes() + output_pos_arr.tobytes()
        # Try to hit the cache
        if signature in self._u_extract_cache:
            return self._u_extract_cache[signature]

        # The vector FROM the auxiliary input TO the output
        r_io = output_pos_arr - input_pos_arr

        transform = np.block([  # Transform force and moments from Ik to O
            [np.identity(3), skew_sym_mat(r_io)],  # <-- positive sign on the skew!
            [np.zeros((3, 3)), np.identity(3)]
        ]).astype(np.float64)

        u_extract = np.block([
            [np.zeros((6, 12))],
            [np.zeros((6, 6)), transform],
        ]).astype(np.float64)

        self._u_extract_cache[signature] = u_extract
        return u_extract

    def h(self, ref_pos: VectorLike, input_pos: VectorLike) -> Matrix:
        """
        Calculates the 6x12 geometric incidence matrix mapping the 12x1 state
        vector at ref_pos to the 6x1 kinematic state vector at target_pos.
        :param ref_pos: Reference input position.
        :param input_pos: Input position.
        """
        input_pos_arr = np.array(input_pos, dtype=np.float64)
        signature = input_pos_arr.tobytes()
        # Try to hit the cache
        if signature in self._h_cache:
            return self._h_cache[signature]

        # The vector FROM the reference point TO the target point
        r_ab = input_pos_arr - np.array(ref_pos, dtype=np.float64)

        # Kinematic transform (Notice the NEGATIVE skew matrix)
        transform = np.block([
            [np.identity(3), -skew_sym_mat(r_ab)],
            [np.zeros((3, 3)), np.identity(3)]
        ]).astype(np.float64)

        # Pad to 6x12 to map from a full [Z_all] state vector
        h_matrix = np.block([
            transform, np.zeros((6, 6))
        ]).astype(np.float64)

        self._h_cache[signature] = h_matrix
        return h_matrix

    @abstractmethod
    def modal_mass(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                   local_input_state: Vector) -> np.float64:
        pass


class DiscreteElement(Element, ABC):
    """
    Base class for discrete elements, mostly rigid bodies
    """
    def __init__(self, e_id: EntityID):
        super().__init__(e_id)

    def modal_mass(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                   local_input_state: Vector) -> np.float64:
        """

        Parameters
        ----------
        input_pos : Vector
        output_pos : Vector
            For a rigid body this should be the center of mass
        omega : float
        local_input_state : Vector
            The mode's state vector at the element's main input

        Returns
        -------
        np.float64
            The modal mass of the element for the given mode
        """
        # Cast inputs to arrays
        input_pos_arr = np.array(input_pos, dtype=np.float64)
        com_pos_arr = np.array(output_pos, dtype=np.float64)

        # First, we need the state at the COM of the rigid body
        u_com = self.u(input_pos_arr, com_pos_arr, omega)
        state_vector = u_com @ np.array(local_input_state, dtype=np.float64)

        # Truncate the state vectors to keep only the kinematics -> mode shape
        v = state_vector[0:6].reshape(6, 1).astype(np.float64)

        # Generalized matrix multiplication (extract the scalar from the 1x1 quadratic form)
        return np.float64((v.T @ self._m_param_mat @ v).item())

    @property
    @abstractmethod
    def _m_param_mat(self) -> Matrix:
        """Returns the 6x6 spatial mass/inertia matrix."""
        pass


class ContinuousElement(Element, ABC):
    """
    Base class for continuous elements, mostly flexible bodies
    """
    def __init__(self, e_id: EntityID):
        super().__init__(e_id)

    def modal_mass(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                   local_input_state: np.ndarray) -> np.float64:
        """

        Parameters
        ----------
        input_pos : Vector
        output_pos : Vector
        omega : float
        local_input_state : Vector
            The mode's state vector at the element's main input

        Returns
        -------
        np.float64
            The modal mass of the element for the given mode
        """
        output_pos_arr = np.array(output_pos, dtype=np.float64)
        input_pos_arr = np.array(input_pos, dtype=np.float64)
        # TODO: make sure this a generalized, it should be, but double-check
        beam_len = np.float64(np.linalg.norm(output_pos_arr - input_pos_arr))  # Beam of element

        def mass_integrand(x: np.float64) -> np.float64:
            direction = (output_pos_arr - input_pos_arr) / beam_len
            current_pos = input_pos_arr + direction * x

            # Propagate from the LOCAL input to the intermediate point x
            u_x = self.u(input_pos_arr, current_pos, omega)
            state_at_x = u_x @ np.array(local_input_state, dtype=np.float64)

            # Truncate the state vectors to keep only the kinematics -> mode shape
            v = state_at_x[0:6].reshape(6, 1)

            return np.float64((v.T @ self._m_bar_param_mat @ v).item())

        element_modal_mass, _ = integrate.quad(mass_integrand, 0.0, beam_len)
        return np.float64(element_modal_mass)

    @property
    @abstractmethod
    def _m_bar_param_mat(self) -> Matrix:
        """Returns the 6x6 mass/inertia distribution matrix per unit length."""
        pass


class MasslessMixin:
    @property
    def _m_param_mat(self) -> Matrix:
        return np.zeros((6, 6)).astype(np.float64)

    _m_bar_param_mat = _m_param_mat


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
        self.mat = np.identity(12, dtype=np.float64)
        if self.sign_mat_flag:
            self.mat[6:12, 6:12] *= np.float64(-1)
