from __future__ import annotations
import logging

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from ..common.vecmath import I3, block_rotation, lever_transform
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


@lru_cache(maxsize=16)
def _gauss_legendre(order: int) -> Tuple[Vector, Vector]:
    """
    Gauss-Legendre nodes and weights on ``[-1, 1]``, cached per order.

    Parameters
    ----------
    order : int
        Number of quadrature points.

    Returns
    -------
    tuple of Vector
        ``(nodes, weights)``, both read-only.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _freeze(arr: Matrix) -> Matrix:
    """Mark an array read-only before it enters a cache, so callers cannot alias-mutate it."""
    arr.setflags(write=False)
    return arr


# Data Structures
class Element(ABC):
    """
    Base class for every element of an MSTMM system.

    An element is defined by its transfer matrix, which propagates a 12-component
    extended state vector ``Z = [r; theta; m; q]`` from an input port to an output port.
    Subclasses supply that matrix in the element's own local frame through
    :meth:`_u_local`; the base class handles the change of frame, the purely geometric
    port-to-port transforms, and the modal inner product.

    Attributes
    ----------
    MAX_INPUTS, MAX_OUTPUTS : int | None
        Port capacity. ``None`` is an explicit "unbounded" sentinel.
    e_id : EntityID
        Unique identifier of the element within its system.
    orientation : Matrix
        3x3 direction-cosine matrix ``D`` mapping the element's local frame to the
        system reference frame K_R. Defaults to the identity (axis-aligned).
    """

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

        # The axis-aligned test is answered once here rather than on every transfer-matrix
        # build. It used to be an `np.allclose` inside `u`, which profiling showed to be
        # the single largest cost of a coupled time integration -- the check was more
        # expensive than the matrix it was guarding.
        self._axis_aligned: bool = bool(np.array_equal(self.orientation, I3))
        # 12x12 state-vector rotation and its 6x6 kinematic sub-block, both frame
        # constants. Only built when they are actually needed.
        self._h_rot: Matrix | None = None if self._axis_aligned else block_rotation(self.orientation)
        self._rot6: Matrix | None = None if self._axis_aligned else self._h_rot[0:6, 0:6]

        # Caches for the frequency-independent geometric transforms. Both are keyed by
        # port geometry, of which an element has only a handful, so they are self-limiting.
        self._h_cache: Dict[bytes, Matrix] = {}
        self._u_extract_cache: Dict[bytes, Matrix] = {}

    def u(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        """
        Extended state transfer matrix, expressed in the global frame K_R.

        Port positions are the element's own local geometry, so the element assembles
        its transfer matrix in its local frame via :meth:`_u_local` and the result is
        mapped to the global frame with the MSTMM coordinate-transformation matrix H
        (Rui, *Transfer Matrix Method for Multibody Systems*, Eq. 15.103):

            ``U_global = H @ U_local @ H^T``,  ``H = blkdiag(D, D, D, D)``

        where ``D = self.orientation``. For the default identity orientation this
        reduces to ``U_local`` exactly, and the rotation is skipped entirely.

        Parameters
        ----------
        input_pos : Vector
            Main-input reference position, in the element-local frame.
        output_pos : Vector
            Output position, in the element-local frame.
        omega : float
            Vibration frequency [rad/s].

        Returns
        -------
        Matrix
            The 12x12 transfer matrix in the global frame.
        """
        u_local = self._u_local(input_pos, output_pos, omega)
        if self._axis_aligned:
            return u_local
        h = self._h_rot
        return h @ u_local @ h.T

    @abstractmethod
    def _u_local(self, input_pos: Vector, output_pos: Vector, omega: float) -> Matrix:
        """
        Extended state transfer matrix in the element's LOCAL frame.

        Built from the element's local port geometry as if it were axis-aligned;
        :meth:`u` wraps this with the orientation to express it globally.

        Parameters
        ----------
        input_pos : Vector
            Main-input reference position, in the element-local frame.
        output_pos : Vector
            Output position, in the element-local frame.
        omega : float
            Vibration frequency [rad/s].

        Returns
        -------
        Matrix
            The 12x12 transfer matrix in the local frame.
        """

    def u_extract(self, input_pos: VectorLike, output_pos: VectorLike) -> Matrix:
        """
        Transfer matrix for an auxiliary (non-main) input port.

        An auxiliary input contributes no kinematics of its own: it only carries the
        wrench arriving there across to the output, picking up the lever-arm moment.
        The result is therefore the static lever transform in the force/moment quadrant
        and zero elsewhere.

        Parameters
        ----------
        input_pos : VectorLike
            Auxiliary input position, in the element-local frame.
        output_pos : VectorLike
            Output position, in the element-local frame.

        Returns
        -------
        Matrix
            The 12x12 extraction matrix. Read-only, as it is shared from a cache.
        """
        # TODO: Quantization of floating-point positions could help avoid cache misses
        input_pos_arr = np.asarray(input_pos, dtype=np.float64)
        output_pos_arr = np.asarray(output_pos, dtype=np.float64)
        signature = input_pos_arr.tobytes() + output_pos_arr.tobytes()
        cached = self._u_extract_cache.get(signature)
        if cached is not None:
            return cached

        # The vector FROM the auxiliary input TO the output. Positive lever sign: a force
        # at the input produces r x F about the output.
        r_io = output_pos_arr - input_pos_arr

        u_extract = np.zeros((12, 12), dtype=np.float64)
        u_extract[6:12, 6:12] = lever_transform(r_io, sign=1.0)

        self._u_extract_cache[signature] = _freeze(u_extract)
        return u_extract

    def h(self, ref_pos: VectorLike, input_pos: VectorLike) -> Matrix:
        """
        Geometric incidence matrix mapping a full state vector to a kinematic state.

        Maps the 12x1 state vector referred to ``ref_pos`` to the 6x1 kinematic state
        ``[r; theta]`` at ``input_pos``. Negative lever sign: a rotation at the reference
        point displaces the target point by ``theta x r``.

        Parameters
        ----------
        ref_pos : VectorLike
            Reference position, in the element-local frame.
        input_pos : VectorLike
            Target position, in the element-local frame.

        Returns
        -------
        Matrix
            The 6x12 incidence matrix. Read-only, as it is shared from a cache.

        Notes
        -----
        The cache is keyed on *both* positions. Keying it on ``input_pos`` alone -- as an
        earlier revision did -- returns a stale matrix as soon as the same element is
        queried from two different reference points, which happens for any element with
        more than one input port. The failure is silent: the geometric constraint
        equations are assembled from a lever arm measured to the wrong origin.
        """
        ref_pos_arr = np.asarray(ref_pos, dtype=np.float64)
        input_pos_arr = np.asarray(input_pos, dtype=np.float64)
        signature = ref_pos_arr.tobytes() + input_pos_arr.tobytes()
        cached = self._h_cache.get(signature)
        if cached is not None:
            return cached

        # The vector FROM the reference point TO the target point.
        r_ab = input_pos_arr - ref_pos_arr

        # Pad the kinematic transform to 6x12 to map from a full [Z_all] state vector.
        h_matrix = np.zeros((6, 12), dtype=np.float64)
        h_matrix[:, 0:6] = lever_transform(r_ab, sign=-1.0)

        self._h_cache[signature] = _freeze(h_matrix)
        return h_matrix

    @abstractmethod
    def modal_product(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                      state_k: Vector, state_p: Vector) -> np.float64:
        """
        This element's contribution to the augmented inner product ``<M V^k, V^p>``.

        The bilinear form, not just its diagonal: off-diagonal terms are what make it
        possible to orthogonalize the eigenvectors of a repeated eigenfrequency against
        each other, which the SVD basis does not do on its own.

        Parameters
        ----------
        input_pos : VectorLike
            Main-input reference position, in the element-local frame.
        output_pos : VectorLike
            Far end of the element's extent: its center of mass for a rigid body, its
            output port for a beam.
        omega : float
            Frequency of the modes being paired [rad/s].
        state_k, state_p : Vector
            The two modes' state vectors at the element's main input.

        Returns
        -------
        np.float64
            The scalar contribution.
        """

    def modal_mass(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                   local_input_state: Vector) -> np.float64:
        """
        The element's modal mass for one mode.

        The diagonal ``<M V^p, V^p>`` of :meth:`modal_product`.

        Parameters
        ----------
        input_pos : VectorLike
            Main-input reference position, in the element-local frame.
        output_pos : VectorLike
            Far end of the element's extent.
        omega : float
            Mode frequency [rad/s].
        local_input_state : Vector
            The mode's state vector at the element's main input.

        Returns
        -------
        np.float64
            The element's modal mass.
        """
        return self.modal_product(input_pos, output_pos, omega,
                                  local_input_state, local_input_state)

    def _shapes_in_local_frame(self, u_mat: Matrix, states: Matrix) -> Matrix:
        """
        Kinematic halves of one or more propagated state vectors, in the element's frame.

        The parametric mass matrices are written in the element's local frame, so the
        (global) mode shapes must be rotated into it before they are paired. For an
        axis-aligned element this is a no-op and is skipped.

        Parameters
        ----------
        u_mat : Matrix
            The 12x12 transfer matrix from the main input to the evaluation point.
        states : Matrix
            State vectors at the main input, stacked column-wise as ``(12, k)``.

        Returns
        -------
        Matrix
            The ``(6, k)`` kinematic blocks, expressed in the element's local frame.
        """
        propagated = (u_mat @ states)[0:6]
        if self._axis_aligned:
            return propagated
        return self._rot6.T @ propagated


class DiscreteElement(Element, ABC):
    """
    Base class for discrete elements, mostly rigid bodies.

    The mass is lumped, so the augmented inner product is a single bilinear form
    evaluated at the body's center of mass.
    """

    def modal_product(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                      state_k: Vector, state_p: Vector) -> np.float64:
        """
        A discrete element's ``<M V^k, V^p>``.

        Parameters
        ----------
        input_pos : VectorLike
            Main-input reference position, in the element-local frame.
        output_pos : VectorLike
            The body's center of mass.
        omega : float
            Frequency of the modes being paired [rad/s].
        state_k, state_p : Vector
            The two modes' state vectors at the element's main input. Pass the same
            vector twice to recover the modal mass (see :meth:`Element.modal_mass`).

        Returns
        -------
        np.float64
            The scalar contribution.
        """
        input_pos_arr = np.asarray(input_pos, dtype=np.float64)
        com_pos_arr = np.asarray(output_pos, dtype=np.float64)

        # Transfer both modes' states to the center of mass in one product, keeping only
        # the kinematic half -- the mode shape.
        u_com = self.u(input_pos_arr, com_pos_arr, omega)
        states = np.column_stack([
            np.asarray(state_k, dtype=np.float64),
            np.asarray(state_p, dtype=np.float64),
        ])
        v = self._shapes_in_local_frame(u_com, states)

        return np.float64(v[:, 0] @ self._m_param_mat @ v[:, 1])

    @property
    @abstractmethod
    def _m_param_mat(self) -> Matrix:
        """The 6x6 spatial mass/inertia matrix."""


class ContinuousElement(Element, ABC):
    """
    Base class for continuous elements, mostly flexible bodies.

    The mass is distributed, so the augmented inner product is integrated along the
    element's span.

    Attributes
    ----------
    QUAD_ORDER : int
        Default number of Gauss-Legendre points used for that integration. Subclasses
        may override :meth:`_quad_order` to choose it from the mode being integrated.
    """

    QUAD_ORDER = 48

    def _quad_order(self, omega: float, span_length: float) -> int:
        """
        Number of quadrature points for one modal-product integration.

        Parameters
        ----------
        omega : float
            Frequency of the modes being paired [rad/s].
        span_length : float
            Length of the interval being integrated over.

        Returns
        -------
        int
            The quadrature order.
        """
        return self.QUAD_ORDER

    def modal_product(self, input_pos: VectorLike, output_pos: VectorLike, omega: float,
                      state_k: Vector, state_p: Vector) -> np.float64:
        """
        A continuous element's ``<M V^k, V^p>``, integrated along its span.

        Parameters
        ----------
        input_pos : VectorLike
            Main-input reference position, in the element-local frame.
        output_pos : VectorLike
            Output port position, in the element-local frame.
        omega : float
            Frequency of the modes being paired [rad/s].
        state_k, state_p : Vector
            The two modes' state vectors at the element's main input. Pass the same
            vector twice to recover the modal mass (see :meth:`Element.modal_mass`).

        Returns
        -------
        np.float64
            The scalar contribution.

        Notes
        -----
        Evaluated with fixed-order Gauss-Legendre quadrature rather than an adaptive
        rule. The integrand is a product of two mode shapes -- analytic, and smooth on
        the span -- which is precisely the case where Gauss-Legendre converges
        spectrally, so a fixed order reaches machine precision at a predictable cost.
        An adaptive rule spends most of its budget on error estimation for an integrand
        that never needed it, and makes the cost of a modal analysis data-dependent.
        """
        input_pos_arr = np.asarray(input_pos, dtype=np.float64)
        output_pos_arr = np.asarray(output_pos, dtype=np.float64)
        span = output_pos_arr - input_pos_arr
        beam_len = float(np.linalg.norm(span))
        if beam_len == 0.0:
            return np.float64(0.0)
        direction = span / beam_len

        states = np.column_stack([
            np.asarray(state_k, dtype=np.float64),
            np.asarray(state_p, dtype=np.float64),
        ])
        m_bar = self._m_bar_param_mat

        # Map the [-1, 1] rule onto [0, beam_len].
        nodes, weights = _gauss_legendre(self._quad_order(omega, beam_len))
        half = 0.5 * beam_len
        stations = half * (nodes + 1.0)

        total = 0.0
        for x, w in zip(stations, weights):
            u_x = self.u(input_pos_arr, input_pos_arr + direction * x, omega)
            v = self._shapes_in_local_frame(u_x, states)
            total += w * float(v[:, 0] @ m_bar @ v[:, 1])

        return np.float64(half * total)

    @property
    @abstractmethod
    def _m_bar_param_mat(self) -> Matrix:
        """The 6x6 mass/inertia distribution matrix per unit length."""


class MasslessMixin:
    """Mixin supplying null mass matrices for elements that carry no inertia."""

    _ZERO_6 = np.zeros((6, 6), dtype=np.float64)
    _ZERO_6.setflags(write=False)

    @property
    def _m_param_mat(self) -> Matrix:
        return self._ZERO_6

    @property
    def _m_bar_param_mat(self) -> Matrix:
        return self._ZERO_6


@dataclass
class Boundary:
    """
    A boundary of the system, carrying its partially-known state vector.

    Attributes
    ----------
    b_id : EntityID
        Identifier of the boundary.
    state_vector : VectorLike
        The 12-component boundary state, with ``None`` marking each unknown component
        and a number marking each known one.
    """
    b_id: EntityID
    state_vector: VectorLike


@dataclass
class CutPoint:
    """
    The relation between the two virtual boundaries created by cutting a connection.

    Attributes
    ----------
    b_id1, b_id2 : EntityID
        The two virtual boundaries produced by the cut.
    sign_mat_flag : bool
        Whether the wrench half of the state vector changes sign across the cut. True
        when the cut produced two input tips facing each other (action and reaction);
        False when it produced a matched output/input pair, whose states are equal.
    mat : Matrix
        The resulting 12x12 relation matrix, built in ``__post_init__``.
    """
    b_id1: EntityID
    b_id2: EntityID
    sign_mat_flag: bool = True
    mat: Matrix = field(init=False)

    def __post_init__(self):
        self.mat = np.identity(12, dtype=np.float64)
        if self.sign_mat_flag:
            self.mat[6:12, 6:12] *= np.float64(-1)
