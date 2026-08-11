from __future__ import annotations
import logging

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

import numpy as np
from scipy import integrate

from ..common.vecmath import skew_sym_mat, block_rotation
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

    def __init__(self, e_id: EntityID, orientation: Matrix | None = None):
        self.e_id = e_id

        # Absolute orientation of the element's local frame with respect to the
        # system reference frame K_R, as a 3x3 direction-cosine matrix D that maps
        # local -> global coordinates. Defaults to the identity (axis-aligned).
        self.orientation = (
            np.identity(3, dtype=np.float64) if orientation is None
            else np.asarray(orientation, dtype=np.float64)
        )

        # Caches
        self._h_cache: Dict[bytes, Matrix] = {}
        self._u_extract_cache: Dict[bytes, Matrix] = {}

    def u(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        """
        Returns the extended state transfer matrix for the element, expressed in
        the global (system reference) frame K_R.

        Port positions are the element's own local geometry, so the element
        assembles its transfer matrix in its local frame via `_u_local` and the
        result is mapped to the global frame with the MSTMM coordinate-
        transformation matrix H (Rui, Transfer Matrix Method for Multibody
        Systems, Eq. 15.103):

            U_global = H @ U_local @ H^T,   H = blkdiag(D, D, D, D),

        where D = self.orientation is the local->global direction-cosine matrix.
        For the default identity orientation this reduces to U_local exactly.

        :param input_pos: Main-input reference position (element-local frame).
        :param output_pos: Output position (element-local frame).
        :param omega: vibration frequency (rad/s)
        :return: Transfer matrix (12x12), global frame.
        """
        u_local = self._u_local(input_pos, output_pos, omega)
        d = self.orientation
        if np.allclose(d, np.identity(3)):
            return u_local.astype(np.float64)
        h = block_rotation(d)
        return (h @ u_local @ h.T).astype(np.float64)

    @abstractmethod
    def _u_local(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        """
        Returns the extended state transfer matrix in the element's LOCAL frame,
        built from the element's local port geometry as if it were axis-aligned.
        The base-class `u` wraps this with the orientation to express it globally.

        :param input_pos: Main-input reference position (element-local frame).
        :param output_pos: Output position (element-local frame).
        :param omega: vibration frequency (rad/s)
        :return: Transfer matrix (12x12), local frame.
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
    def modal_product(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                      state_k: Vector, state_p: Vector) -> np.float64:
        """
        This element's contribution to the augmented inner product ``<M V^k, V^p>``.

        The bilinear form, not just its diagonal: off-diagonal terms are what make it
        possible to orthogonalise the eigenvectors of a repeated eigenfrequency against
        each other, which the SVD basis does not do on its own.

        :param input_pos: Main-input reference position (element-local frame).
        :param output_pos: Far end of the element's extent (its centre of mass for a
            rigid body, its output port for a beam).
        :param omega: Frequency of the modes being paired [rad/s].
        :param state_k: Mode k's state vector at the element's main input.
        :param state_p: Mode p's state vector at the element's main input.
        """

    def modal_mass(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                   local_input_state: Vector) -> np.float64:
        """
        The element's modal mass for one mode -- the diagonal ``<M V^p, V^p>`` of
        :meth:`modal_product`.
        """
        return self.modal_product(input_pos, output_pos, omega,
                                  local_input_state, local_input_state)


class DiscreteElement(Element, ABC):
    """
    Base class for discrete elements, mostly rigid bodies
    """
    def __init__(self, e_id: EntityID, orientation: Matrix | None = None):
        super().__init__(e_id, orientation)

    def modal_product(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                      state_k: Vector, state_p: Vector) -> np.float64:
        """
        A discrete element's ``<M V^k, V^p>``: the mass matrix is lumped, so the product
        is a single bilinear form evaluated at the body's centre of mass.

        Parameters
        ----------
        input_pos : Vector
        output_pos : Vector
            For a rigid body this should be the center of mass
        omega : float
        state_k, state_p : Vector
            The two modes' state vectors at the element's main input. Pass the same
            vector twice to recover the modal mass (see :meth:`Element.modal_mass`).
        """
        input_pos_arr = np.array(input_pos, dtype=np.float64)
        com_pos_arr = np.array(output_pos, dtype=np.float64)

        # Transfer both modes' states to the centre of mass, keeping only the kinematic
        # half -- the mode shape. The parametric mass matrix lives in the element's local
        # frame, so rotate the (global) shapes into it; identity for axis-aligned bodies.
        u_com = self.u(input_pos_arr, com_pos_arr, omega)
        r6 = block_rotation(self.orientation)[0:6, 0:6]
        v_k = r6.T @ (u_com @ np.asarray(state_k, dtype=np.float64))[0:6]
        v_p = r6.T @ (u_com @ np.asarray(state_p, dtype=np.float64))[0:6]

        return np.float64(v_k @ self._m_param_mat @ v_p)

    @property
    @abstractmethod
    def _m_param_mat(self) -> Matrix:
        """Returns the 6x6 spatial mass/inertia matrix."""
        pass


class ContinuousElement(Element, ABC):
    """
    Base class for continuous elements, mostly flexible bodies
    """
    def __init__(self, e_id: EntityID, orientation: Matrix | None = None):
        super().__init__(e_id, orientation)

    def modal_product(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                      state_k: Vector, state_p: Vector) -> np.float64:
        """
        A continuous element's ``<M V^k, V^p>``: the mass is distributed, so the bilinear
        form is integrated along the element's span.

        Parameters
        ----------
        input_pos : Vector
        output_pos : Vector
        omega : float
        state_k, state_p : Vector
            The two modes' state vectors at the element's main input. Pass the same
            vector twice to recover the modal mass (see :meth:`Element.modal_mass`).
        """
        input_pos_arr = np.array(input_pos, dtype=np.float64)
        output_pos_arr = np.array(output_pos, dtype=np.float64)
        span = output_pos_arr - input_pos_arr
        beam_len = float(np.linalg.norm(span))
        if beam_len == 0.0:
            return np.float64(0.0)
        direction = span / beam_len

        # The distributed mass matrix is expressed in the element's local frame; rotate the
        # (global) mode shapes into it before pairing them. Identity for axis-aligned beams.
        r6 = block_rotation(self.orientation)[0:6, 0:6]
        state_k = np.asarray(state_k, dtype=np.float64)
        state_p = np.asarray(state_p, dtype=np.float64)
        m_bar = self._m_bar_param_mat

        def integrand(x: float) -> float:
            # Propagate from the LOCAL input to the intermediate point x
            u_x = self.u(input_pos_arr, input_pos_arr + direction * x, omega)
            v_k = r6.T @ (u_x @ state_k)[0:6]
            v_p = r6.T @ (u_x @ state_p)[0:6]
            return float(v_k @ m_bar @ v_p)

        value, _ = integrate.quad(integrand, 0.0, beam_len)
        return np.float64(value)

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
