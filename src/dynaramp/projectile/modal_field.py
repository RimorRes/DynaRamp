from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..common.types import EntityID, Matrix
from ..mstmm.system import System, Mode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalShape:
    """
    The launch rail's mode-shape interpolation matrices evaluated at a single axial
    station x_R, projected into the global inertial frame K_I. Each matrix is 3 x n, where
    n is the number of retained modes (Eqs. 15-17, 31, 33).

    Attributes
    ----------
    x_r : float
        Axial material coordinate along the rail where the field was evaluated.
    phi_r, phi_theta : Matrix
        Translational (Ir) and rotational (Itheta) mode-shape matrices.
    phi_r_d1, phi_r_d2 : Matrix
        First and second spatial derivatives of phi_r (Ir', Ir'').
    phi_theta_d1, phi_theta_d2 : Matrix
        First and second spatial derivatives of phi_theta (Itheta', Itheta'').
    """
    x_r: float
    phi_r: Matrix
    phi_theta: Matrix
    phi_r_d1: Matrix
    phi_r_d2: Matrix
    phi_theta_d1: Matrix
    phi_theta_d2: Matrix


@dataclass(frozen=True)
class GuideSegment:
    """
    One collinear beam segment of the guide, ordered from the K_R (reference) end.

    Attributes
    ----------
    element_id : EntityID
        Identifier of the segment's flexible element in the MSTMM system.
    length : float | None
        Axial length of the segment. If None, it is read from the element's ``length``
        attribute (as for an Euler-Bernoulli beam).
    """
    element_id: EntityID
    length: float | None = None


class GuideModalField:
    """
    The section 2 -> section 3 seam.

    Wraps an MSTMM ``System`` with one flexible element (the launch rail)
    and its precomputed modes, and exposes the rail's mode-shape interpolation field.
    Phi_r, Phi_theta and their first/second spatial derivatives at an axial station x_R,
    projected into the global inertial frame K_I via A_IR.

    It relies solely on the ``System``'s public API (topology, element transfer matrices,
    modes), so the ``mstmm`` package remains entirely agnostic of the projectile machinery.

    Parameters
    ----------
    system : System
        The solved MSTMM system providing the topology and element transfer matrices.
    segments : Sequence[GuideSegment]
        The guide's collinear beam segments, ordered from the K_R (reference) end. A global
        axial coordinate x_R is resolved to the containing segment on evaluation. To build
        directly from element ids (lengths read from the elements), use :meth:`from_elements`.
    modes : List[Mode]
        Retained modes (from ``System.natural_modes``), in ascending frequency order.
    a_ir : VectorLike | None
        3x3 rotation matrix A_IR from the reference frame K_R to the inertial frame K_I.
        Defaults to the identity (launcher upright; rotation not yet wired in).
    step : float
        Finite-difference step h for the spatial-derivative stencil (default 1e-4).
    """

    def __init__(
            self,
            system: System,
            segments: Sequence[GuideSegment],
            modes: List[Mode],
            a_ir: Matrix | None = None,
            step: float = 1e-4,
    ):
        self.system: System = system
        self.modes: List[Mode] = modes
        self.a_ir: Matrix = np.identity(3, dtype=np.float64) if a_ir is None else np.array(a_ir, dtype=np.float64)
        self.step: float = step

        if not segments:
            raise ValueError("A GuideModalField requires at least one GuideSegment.")

        self.segment_ids: List[EntityID] = []
        self._elems = []
        self._input_pos: List[np.ndarray] = []
        self._preds: List[EntityID] = []
        lengths: List[float] = []
        for seg in segments:
            elem_info = system.topology.get_element_info(seg.element_id)
            elem_obj = elem_info.obj
            length = seg.length if seg.length is not None else getattr(elem_obj, "length", None)
            if length is None:
                raise ValueError(
                    f"Segment [{seg.element_id!r}] has no `length`; set it on the GuideSegment."
                )
            input_pos = np.array(elem_info.ports[elem_info.main_input_idx].pos, dtype=np.float64)
            pred_id = next(
                n for n in system.topology.tree.predecessors(seg.element_id)
                if system.topology.tree[n][seg.element_id]['input_port'] == elem_info.main_input_idx
            )
            self.segment_ids.append(seg.element_id)
            self._elems.append(elem_obj)
            self._input_pos.append(input_pos)
            self._preds.append(pred_id)
            lengths.append(float(length))

        self._n_seg: int = len(self.segment_ids)
        self._lengths = np.array(lengths, dtype=np.float64)
        # Cumulative axial start of each segment along the guide (starts[0] = 0).
        self._starts = np.concatenate([[0.0], np.cumsum(self._lengths)[:-1]]).astype(np.float64)
        self.total_length: float = float(self._lengths.sum())

        logger.debug(
            "GuideModalField bound to %d segment(s) %r (total length %.4f) with %d modes.",
            self._n_seg, self.segment_ids, self.total_length, len(modes),
        )

    @classmethod
    def from_elements(
            cls,
            system: System,
            element_ids: Sequence[EntityID],
            modes: List[Mode],
            a_ir: Matrix | None = None,
            step: float = 1e-4,
    ) -> "GuideModalField":
        """
        Convenience constructor from an ordered sequence of element ids, one segment per
        element with its length read from the element. Equivalent to
        ``GuideModalField(system, [GuideSegment(e) for e in element_ids], modes, ...)``.
        """
        return cls(system, [GuideSegment(e) for e in element_ids], modes, a_ir=a_ir, step=step)

    def _resolve(self, x_r: float) -> Tuple[int, float]:
        """
        Map a global axial coordinate x_r to (segment_index, local_offset). Points below
        the first segment or beyond the last are clamped to the end segment (the beam
        transfer matrix continues analytically), so stencil abscissae near the guide ends
        remain valid, exactly as in the single-segment case.
        """
        if x_r <= 0.0:
            return 0, x_r
        for k in range(self._n_seg):
            if x_r <= self._starts[k] + self._lengths[k] or k == self._n_seg - 1:
                return k, float(x_r - self._starts[k])
        return self._n_seg - 1, float(x_r - self._starts[-1])

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def _phi_full_ref(self, x_r: float) -> Matrix:
        """
        Raw 6 x n mode-shape matrix Phi = [Phi_r; Phi_theta] at global axial station x_r,
        expressed in the reference frame K_R (before A_IR projection). The station is
        resolved to its containing collinear segment and evaluated at the local offset.
        """
        k, local = self._resolve(x_r)
        input_pos = self._input_pos[k]
        elem = self._elems[k]
        pred = self._preds[k]
        output_pos = input_pos + np.array([local, 0.0, 0.0])
        cols = []
        for mode in self.modes:
            z = elem.u(input_pos, output_pos, mode.frequency) @ mode.internal_states[pred]
            cols.append(z[0:6])  # kinematics only: [r; theta]
        return np.column_stack(cols).astype(np.float64)

    def _project(self, phi6: Matrix) -> Matrix:
        """Rotate the translational and rotational 3-row blocks of a 6 x n matrix by A_IR."""
        return np.vstack([self.a_ir @ phi6[0:3, :], self.a_ir @ phi6[3:6, :]]).astype(np.float64)

    def phi(self, x_r: float) -> Tuple[Matrix, Matrix]:
        """
        Zeroth-order shapes (Phi_r, Phi_theta), each 3 x n, projected in K_I.
        """
        p = self._project(self._phi_full_ref(x_r))
        return p[0:3, :], p[3:6, :]

    def evaluate(self, x_r: float) -> ModalShape:
        """
        Evaluate the full modal field at axial station x_r: Phi_r, Phi_theta and their
        first/second spatial derivatives, all projected in K_I. Shapes and both
        derivatives are read from a single shared 5-point central finite-difference
        stencil (fourth-order accurate).
        """
        h = self.step
        f_m2 = self._phi_full_ref(x_r - 2 * h)
        f_m1 = self._phi_full_ref(x_r - h)
        f_0 = self._phi_full_ref(x_r)
        f_p1 = self._phi_full_ref(x_r + h)
        f_p2 = self._phi_full_ref(x_r + 2 * h)

        d1 = (f_m2 - 8 * f_m1 + 8 * f_p1 - f_p2) / (12 * h)
        d2 = (-f_m2 + 16 * f_m1 - 30 * f_0 + 16 * f_p1 - f_p2) / (12 * h ** 2)
        d1 = np.asarray(d1, dtype=np.float64)
        d2 = np.asarray(d2, dtype=np.float64)

        p0 = self._project(f_0)
        pd1 = self._project(d1)
        pd2 = self._project(d2)

        return ModalShape(
            x_r=x_r,
            phi_r=p0[0:3, :],
            phi_theta=p0[3:6, :],
            phi_r_d1=pd1[0:3, :],
            phi_r_d2=pd2[0:3, :],
            phi_theta_d1=pd1[3:6, :],
            phi_theta_d2=pd2[3:6, :],
        )
