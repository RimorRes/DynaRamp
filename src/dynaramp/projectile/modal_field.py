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
    The launch rail's mode-shape interpolation matrices at a single axial station.

    Projected into the global inertial frame K_I. Each matrix is ``3 x n``, where ``n``
    is the number of retained modes (Eqs. 15-17, 31, 33).

    Attributes
    ----------
    x_r : float
        Axial material coordinate along the rail where the field was evaluated.
    phi_r : Matrix
        Translational mode-shape matrix ``Ir``.
    phi_theta : Matrix
        Rotational mode-shape matrix ``Itheta``.
    phi_r_d1, phi_r_d2 : Matrix
        First and second spatial derivatives of ``phi_r``.
    phi_theta_d1, phi_theta_d2 : Matrix
        First and second spatial derivatives of ``phi_theta``.
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
        Axial length of the segment. If ``None``, it is read from the element's
        ``length`` attribute, as for an Euler-Bernoulli beam.
    """
    element_id: EntityID
    length: float | None = None


class GuideModalField:
    """
    The section 2 -> section 3 seam.

    Wraps an MSTMM :class:`~dynaramp.mstmm.system.System` describing the launch rail,
    together with its precomputed modes, and exposes the rail's mode-shape interpolation
    field: ``Phi_r``, ``Phi_theta`` and their first and second spatial derivatives at an
    axial station ``x_R``, projected into the global inertial frame K_I via ``A_IR``.

    It relies solely on the ``System``'s public API, so the ``mstmm`` package remains
    entirely agnostic of the projectile machinery.

    Parameters
    ----------
    system : System
        The solved MSTMM system providing the topology and element transfer matrices.
    segments : Sequence[GuideSegment]
        The guide's collinear beam segments, ordered from the K_R (reference) end. A
        global axial coordinate ``x_R`` is resolved to the containing segment on
        evaluation. To build directly from element ids, use :meth:`from_elements`.
    modes : List[Mode]
        Retained modes, from ``System.natural_modes``, in ascending frequency order.
    a_ir : Matrix | None
        3x3 rotation matrix ``A_IR`` mapping the reference frame K_R to the inertial
        frame K_I -- the guide's absolute attitude, e.g., launcher elevation and azimuth.
        It projects every mode-shape block into K_I and is consumed by the section 3
        kinematics and the section 4 contact analysis. Defaults to the identity
        (launcher upright, axes aligned with K_I).
    step : float | None
        Finite-difference step ``h`` for the spatial-derivative stencil. ``None``, the
        default, selects it from :meth:`optimal_step`. Pass a float to override.

    Raises
    ------
    ValueError
        If no segments are given, or a segment's length can be determined neither from
        the ``GuideSegment`` nor from its element.

    Notes
    -----
    The step size matters far more than it looks. The second-derivative stencil divides
    by ``h^2``, so its round-off error grows as ``eps / h^2`` while its truncation error
    falls as ``h^4``; the total error is therefore V-shaped in ``h``, and a step chosen
    "small for accuracy" lands on the wrong side of the minimum. A step of ``1e-4`` on a
    guide of order 10 m roughly carries ``5e-7`` relative error in ``phi_r_d2``, against
    the ``1e-9`` available at the optimum -- and ``phi_r_d2`` feeds the convective
    acceleration terms of Eqs. 31 and 33 directly.
    """

    def __init__(
            self,
            system: System,
            segments: Sequence[GuideSegment],
            modes: List[Mode],
            a_ir: Matrix | None = None,
            step: float | None = None,
    ):
        self.system: System = system
        self.modes: List[Mode] = modes
        self.a_ir: Matrix = np.identity(3, dtype=np.float64) if a_ir is None else np.array(a_ir, dtype=np.float64)

        if not segments:
            raise ValueError("A GuideModalField requires at least one GuideSegment.")

        self.segment_ids: List[EntityID] = []
        self._input_pos: List[np.ndarray] = []
        lengths: List[float] = []
        for seg in segments:
            elem_info = system.topology.get_element_info(seg.element_id)
            length = seg.length if seg.length is not None else getattr(elem_info.obj, "length", None)
            if length is None:
                raise ValueError(
                    f"Segment [{seg.element_id!r}] has no `length`; set it on the GuideSegment."
                )
            # The main input and its predecessor are resolved by the topology, which owns
            # that relation. Re-deriving it here would duplicate -- and could silently
            # diverge from -- `TopologyHandler.main_input_of`.
            _, input_pos = system.topology.main_input_of(seg.element_id)
            self.segment_ids.append(seg.element_id)
            self._input_pos.append(np.asarray(input_pos, dtype=np.float64))
            lengths.append(float(length))

        self._n_seg: int = len(self.segment_ids)
        self._lengths = np.array(lengths, dtype=np.float64)
        # Cumulative axial start of each segment along the guide (starts[0] = 0).
        self._starts = np.concatenate([[0.0], np.cumsum(self._lengths)[:-1]]).astype(np.float64)
        self.total_length: float = float(self._lengths.sum())
        self.step: float = self.optimal_step() if step is None else float(step)

        logger.debug(
            "GuideModalField bound to %d segment(s) %r (total length %.4f) with %d modes, "
            "stencil step %.4e.",
            self._n_seg, self.segment_ids, self.total_length, len(modes), self.step,
        )

    def optimal_step(self) -> float:
        """
        A finite-difference step near the accuracy optimum for this guide and mode set.

        Returns
        -------
        float
            The recommended step ``h``.

        Notes
        -----
        Balancing the fourth-order truncation error ``~ C h^4`` against the
        round-off error of the second-derivative stencil ``~ eps / h^2`` puts the
        minimum at ``h ~ L_c eps^(1/6)``, where ``L_c`` is the length over which the
        field being differentiated actually varies.

        That length is *not* the whole guide. The n-th mode of a beam carries roughly
        ``n`` half-waves along its span, so the finest feature the stencil has to
        resolve is of order ``L / n_modes``, and retaining more modes should shorten
        the step. Using the full length instead would overstep a high-mode basis.

        ``eps^(1/6)`` is about ``2.4e-3``, so a 13 m guide with seven modes lands near
        ``5e-3`` -- some fifty times larger than a naive ``1e-4``, and about two orders
        of magnitude more accurate for it.
        """
        characteristic_length = self.total_length / max(1, self.n_modes)
        return characteristic_length * float(np.finfo(np.float64).eps) ** (1.0 / 6.0)

    @classmethod
    def from_elements(
            cls,
            system: System,
            element_ids: Sequence[EntityID],
            modes: List[Mode],
            a_ir: Matrix | None = None,
            step: float | None = None,
    ) -> GuideModalField:
        """
        Convenience constructor from an ordered sequence of element ids.

        One segment per element, with its length read from the element.

        Parameters
        ----------
        system : System
            The solved MSTMM system.
        element_ids : Sequence[EntityID]
            The guide's elements, ordered from the K_R end.
        modes : List[Mode]
            Retained modes.
        a_ir : Matrix | None
            The guide's absolute attitude.
        step : float | None
            Finite-difference step; ``None`` selects :meth:`optimal_step`.

        Returns
        -------
        GuideModalField
            Equivalent to passing ``[GuideSegment(e) for e in element_ids]``.
        """
        return cls(system, [GuideSegment(e) for e in element_ids], modes, a_ir=a_ir, step=step)

    @property
    def n_modes(self) -> int:
        """Number of retained modes."""
        return len(self.modes)

    def _resolve(self, x_r: float) -> Tuple[int, float]:
        """
        Map a global axial coordinate to its segment and the offset within it.

        Parameters
        ----------
        x_r : float
            Global axial coordinate along the guide.

        Returns
        -------
        tuple
            ``(segment_index, local_offset)``.

        Notes
        -----
        Points below the first segment or beyond the last are clamped to the end
        segment, whose beam transfer matrix continues analytically. This keeps stencil
        abscissae near the guide ends valid, exactly as in the single-segment case.
        """
        if x_r <= 0.0:
            return 0, x_r
        for k in range(self._n_seg):
            if x_r <= self._starts[k] + self._lengths[k] or k == self._n_seg - 1:
                return k, float(x_r - self._starts[k])
        return self._n_seg - 1, float(x_r - self._starts[-1])

    def _phi_full_ref(self, x_r: float) -> Matrix:
        """
        Raw mode-shape matrix at a global axial station, in the reference frame K_R.

        Parameters
        ----------
        x_r : float
            Global axial coordinate along the guide.

        Returns
        -------
        Matrix
            The ``6 x n`` matrix ``Phi = [Phi_r; Phi_theta]``, before ``A_IR`` projection.
            The station is resolved to its containing collinear segment and evaluated at
            the local offset.
        """
        k, local = self._resolve(x_r)
        position = self._input_pos[k] + np.array([local, 0.0, 0.0])
        # (n_modes, 6) from the system, transposed to the (6, n_modes) column layout the
        # projection and the kinematics expect.
        return self.system.mode_shapes_at(self.modes, self.segment_ids[k], position).T

    def _project(self, phi6: Matrix) -> Matrix:
        """
        Rotate the translational and rotational blocks of a ``6 x n`` matrix by ``A_IR``.

        Parameters
        ----------
        phi6 : Matrix
            A ``6 x n`` matrix in the reference frame K_R.

        Returns
        -------
        Matrix
            The same matrix expressed in K_I.
        """
        return np.vstack([self.a_ir @ phi6[0:3, :], self.a_ir @ phi6[3:6, :]]).astype(np.float64)

    def phi(self, x_r: float) -> Tuple[Matrix, Matrix]:
        """
        Zeroth-order mode shapes at an axial station.

        Parameters
        ----------
        x_r : float
            Global axial coordinate along the guide.

        Returns
        -------
        tuple of Matrix
            ``(Phi_r, Phi_theta)``, each ``3 x n``, projected in K_I.
        """
        p = self._project(self._phi_full_ref(x_r))
        return p[0:3, :], p[3:6, :]

    def shape_at(self, x_r: float) -> Matrix:
        """
        The mode shapes at an axial station, in the row layout used for load projection.

        Parameters
        ----------
        x_r : float
            Global axial coordinate along the guide.

        Returns
        -------
        Matrix
            An ``(n_modes, 6)`` array, matching the convention of
            :meth:`dynaramp.mstmm.response.ModalBasis.shape_at` so that both can be fed
            to :func:`dynaramp.common.modal.project_wrench`.
        """
        return self._project(self._phi_full_ref(x_r)).T

    def evaluate(self, x_r: float) -> ModalShape:
        """
        Evaluate the full modal field at an axial station.

        ``Phi_r``, ``Phi_theta`` and their first and second spatial derivatives, all
        projected in K_I. Shapes and both derivatives are read from a single shared
        5-point central finite-difference stencil, fourth-order accurate.

        Parameters
        ----------
        x_r : float
            Global axial coordinate along the guide.

        Returns
        -------
        ModalShape
            The evaluated field.
        """
        h = self.step
        f_m2 = self._phi_full_ref(x_r - 2 * h)
        f_m1 = self._phi_full_ref(x_r - h)
        f_0 = self._phi_full_ref(x_r)
        f_p1 = self._phi_full_ref(x_r + h)
        f_p2 = self._phi_full_ref(x_r + 2 * h)

        d1 = (f_m2 - 8 * f_m1 + 8 * f_p1 - f_p2) / (12 * h)
        d2 = (-f_m2 + 16 * f_m1 - 30 * f_0 + 16 * f_p1 - f_p2) / (12 * h ** 2)

        p0 = self._project(f_0)
        pd1 = self._project(np.asarray(d1, dtype=np.float64))
        pd2 = self._project(np.asarray(d2, dtype=np.float64))

        return ModalShape(
            x_r=x_r,
            phi_r=p0[0:3, :],
            phi_theta=p0[3:6, :],
            phi_r_d1=pd1[0:3, :],
            phi_r_d2=pd2[0:3, :],
            phi_theta_d1=pd1[3:6, :],
            phi_theta_d2=pd2[3:6, :],
        )
