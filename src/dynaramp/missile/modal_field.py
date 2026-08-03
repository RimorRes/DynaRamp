from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..common.types import EntityID, Matrix, VectorLike
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


class RailModalField:
    """
    The section 2 -> section 3 seam.

    Wraps an MSTMM ``System`` together with one flexible element (the launch rail)
    and its precomputed modes, and exposes the rail's mode-shape interpolation field
    Phi_r, Phi_theta and their first/second spatial derivatives at an axial station x_R,
    projected into the global inertial frame K_I via A_IR.

    It relies solely on the ``System``'s public API (topology, element transfer matrices,
    modes), so the ``mstmm`` package remains entirely agnostic of the missile machinery.

    Parameters
    ----------
    system : System
        The solved MSTMM system providing the topology and element transfer matrices.
    elem_id : EntityID
        Identifier of the flexible element modeling the launch rail.
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
            elem_id: EntityID,
            modes: List[Mode],
            a_ir: Matrix | None = None,
            step: float = 1e-4,
    ):
        self.system: System = system
        self.elem_id: EntityID = elem_id
        self.modes: List[Mode] = modes
        self.a_ir: Matrix = np.identity(3, dtype=np.float64) if a_ir is None else np.array(a_ir, dtype=np.float64)
        self.step: float = step

        # Cache the (time-invariant) topology lookups used on every evaluation.
        elem_info = system.topology.get_element_info(elem_id)
        self._elem = elem_info.obj
        self._input_pos = np.array(elem_info.ports[elem_info.main_input_idx].pos, dtype=np.float64)
        self._pred_id = next(
            n for n in system.topology.tree.predecessors(elem_id)
            if system.topology.tree[n][elem_id]['input_port'] == elem_info.main_input_idx
        )

        logger.debug(
            "RailModalField bound to element [%r] with %d modes.", elem_id, len(modes)
        )

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def _phi_full_ref(self, x_r: float) -> Matrix:
        """
        Raw 6 x n mode-shape matrix Phi = [Phi_r; Phi_theta] at axial station x_r,
        expressed in the reference frame K_R (before A_IR projection).
        """
        output_pos = self._input_pos + np.array([x_r, 0.0, 0.0])
        cols = []
        for mode in self.modes:
            z = self._elem.u(self._input_pos, output_pos, mode.frequency) @ mode.internal_states[self._pred_id]
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
