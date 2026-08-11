from __future__ import annotations

import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple, Any

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

from ..common.types import EntityID, Vector, Matrix
from ..common.modal import rayleigh_modal_damping
from .topology import PropagationStep, TopologyHandler
from .element_lib import RigidBody

logger = logging.getLogger(__name__)

# TODO: better error raising, custom exceptions?
# TODO: Add rich results classes

_I12 = np.identity(12, dtype=np.float64)
_I12.setflags(write=False)
_ZERO_6x12 = np.zeros((6, 12), dtype=np.float64)
_ZERO_6x12.setflags(write=False)

type _Plan = Tuple[PropagationStep, ...]


@dataclass
class Mode:
    """
    One eigenvector of the system at one eigenfrequency.

    Attributes
    ----------
    frequency : float
        The eigenfrequency [rad/s].
    internal_states : Dict[EntityID, Vector]
        The 12-component state vector at every node of the tree, keyed by node ID.
    """
    frequency: float
    internal_states: Dict[EntityID, Vector]


@dataclass(frozen=True)
class _ConstraintPlan:
    """
    The frequency-independent structure of one multi-input element's geometric constraints.

    Where several branches converge on a single element, their kinematic states at the
    junction must agree. Which tips feed which branch, and the incidence matrix carrying
    each branch's state to the element's main input, are all fixed by the topology; only
    the transfer matrices along the branches depend on frequency.

    Attributes
    ----------
    tip_terms : Dict[EntityID, tuple]
        Per upstream tip, the ``(propagation_plan, incidence_matrix)`` pair needed to
        build its kinematic contribution.
    pairs : tuple
        One ``(reference_tips, other_tips)`` pair per independent constraint set. With
        ``N`` incoming branches there are ``N - 1`` of them.
    """
    tip_terms: Dict[EntityID, Tuple[_Plan, Matrix]]
    pairs: Tuple[Tuple[FrozenSet[EntityID], FrozenSet[EntityID]], ...]


@dataclass(frozen=True)
class _Assembly:
    """
    Everything about the overall transfer equation that does not depend on frequency.

    Rebuilt only when the topology changes. A modal search evaluates the overall transfer
    matrix at thousands of frequencies, and without this every one of them re-walked the
    tree, re-resolved every path and re-derived the boundary-condition masks.

    Attributes
    ----------
    boundaries : List[EntityID]
        Boundary IDs in their original order, root first.
    tip_plans : List[tuple]
        Propagation plan from each tip to the root, in boundary order.
    constraints : tuple of _ConstraintPlan
        One entry per multi-input element.
    merges : tuple
        Cut-point column merges as ``(keep_index, drop_index, relation_matrix)``, applied
        in order.
    merged_boundaries : List[EntityID]
        Boundary IDs surviving the merges.
    z_merged : Vector
        Merged boundary state vector, still containing ``None`` for unknowns.
    known_mask : Any
        Boolean mask of components with a prescribed value.
    nonzero_mask : Any
        Boolean mask of components with a prescribed *non-zero* value.
    z_nz : Vector
        The prescribed non-zero values themselves.
    """
    boundaries: List[EntityID]
    tip_plans: List[_Plan]
    constraints: Tuple[_ConstraintPlan, ...]
    merges: Tuple[Tuple[int, int, Matrix], ...]
    merged_boundaries: List[EntityID]
    z_merged: Vector
    known_mask: Any
    nonzero_mask: Any
    z_nz: Vector


def null_space_dimension(sigma: Vector, max_dim: int = 6) -> int:
    """
    Estimate the null-space dimension of the overall transfer matrix from its spectrum.

    That dimension is the multiplicity of the eigenfrequency. A simple absolute
    threshold is unreliable here: at an eigenfrequency the smallest singular value can
    land anywhere from 1e-5 to 1e-13 depending on conditioning. What *is* robust is the
    gap. The null-space singular values sit orders of magnitude below the rest, so the
    spectrum is cut at its largest multiplicative jump.

    Parameters
    ----------
    sigma : Vector
        Singular values in descending order, as returned by ``np.linalg.svd``.
    max_dim : int
        Largest multiplicity to consider. Guards against a badly rank-deficient matrix
        being read as a hugely degenerate eigenspace.

    Returns
    -------
    int
        The number of trailing singular values belonging to the null space.
    """
    tail = np.asarray(sigma[-(max_dim + 1):], dtype=np.float64)
    ascending = tail[::-1]                      # walk up from the smallest
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(ascending[:-1] > 0, ascending[1:] / ascending[:-1], np.inf)
    if ratios.size == 0:
        return 1
    return int(np.argmax(ratios)) + 1


class System:
    """
    Solver operating on a reduced tree topology.

    Assembles the overall transfer equation, finds the system's natural modes, and
    evaluates mode shapes and the augmented inner product.

    Parameters
    ----------
    system_topo : TopologyHandler
        A topology on which :meth:`TopologyHandler.make_tree` has already been called.
    """

    def __init__(self, system_topo: TopologyHandler):
        self.topology = system_topo
        self._assembly: _Assembly | None = None
        self._assembly_key: Vector | None = None

    # ------------------------------------------------------------------------------
    # Frequency-independent assembly
    # ------------------------------------------------------------------------------

    def _get_assembly(self) -> _Assembly:
        """
        The cached frequency-independent assembly, rebuilt if the topology changed.

        Returns
        -------
        _Assembly
            The cached structure.

        Notes
        -----
        Staleness is detected by identity of the topology's ``z_all`` array, which is
        replaced wholesale whenever the tree is regenerated.
        """
        z_all = self.topology.z_all
        if self._assembly is not None and self._assembly_key is z_all:
            return self._assembly

        boundaries = self.topology.boundaries
        root_id = self.topology.root.b_id

        # --- propagation plans, one per tip, in boundary order (the root has none) ---
        tip_plans: List[_Plan] = []
        for b_id in boundaries[1:]:
            path = self.topology.resolve_branch_up_to(src=b_id, tgt=root_id)
            tip_plans.append(self.topology.propagation_plan(path))

        # --- geometric constraint structure at every converging element ---
        constraints: List[_ConstraintPlan] = []
        for m_id in self.topology.element_ids:
            if self.topology.tree.in_degree(m_id) <= 1:
                continue
            m_info = self.topology.get_element_info(m_id)
            preds = list(self.topology.tree.predecessors(m_id))
            if len(preds) <= 1:
                continue

            main_port_pos = m_info.main_input_pos
            tips_by_branch: Dict[EntityID, List[EntityID]] = {p: [] for p in preds}
            tip_terms: Dict[EntityID, Tuple[_Plan, Matrix]] = {}

            for t_id in m_info.upstream_tips:
                path = self.topology.resolve_branch_up_to(src=t_id, tgt=m_id)
                branch_node = path[-1]
                if branch_node in tips_by_branch:
                    tips_by_branch[branch_node].append(t_id)

                input_port_idx: int = self.topology.tree[branch_node][m_id]['input_port']
                input_port_pos: Vector = m_info.ports[input_port_idx].pos
                # The incidence matrix carries no frequency dependence at all.
                h_mat = m_info.obj.h(ref_pos=main_port_pos, input_pos=input_port_pos)
                tip_terms[t_id] = (self.topology.propagation_plan(path), h_mat)

            ref_pred = preds[0]
            pairs = tuple(
                (frozenset(tips_by_branch[ref_pred]), frozenset(tips_by_branch[other]))
                for other in preds[1:]
            )
            constraints.append(_ConstraintPlan(tip_terms=tip_terms, pairs=pairs))

        # --- cut-point column merges, and the surviving boundary state vector ---
        merged_boundaries = boundaries.copy()
        merged_z = self.topology.z_all.copy()
        merges: List[Tuple[int, int, Matrix]] = []
        for cut in self.topology.cut_points:
            try:
                idx1 = merged_boundaries.index(cut.b_id1)
                idx2 = merged_boundaries.index(cut.b_id2)
            except ValueError as exc:
                err_msg = f"Cut point boundaries [{cut.b_id1}] or [{cut.b_id2}] not found in tips."
                logger.error(err_msg)
                raise ValueError(err_msg) from exc

            merges.append((idx1, idx2, cut.mat))
            merged_boundaries.pop(idx2)
            mask = np.ones_like(merged_z, dtype=bool)
            mask[(12 * idx2):(12 * (idx2 + 1))] = False
            merged_z = merged_z[mask]

        # --- boundary-condition masks ---
        known_mask = np.array([x is not None for x in merged_z])
        nonzero_mask = np.array([x != 0 for x in merged_z]) & known_mask
        z_nz = merged_z[nonzero_mask].astype(np.float64)

        assembly = _Assembly(
            boundaries=boundaries,
            tip_plans=tip_plans,
            constraints=tuple(constraints),
            merges=tuple(merges),
            merged_boundaries=merged_boundaries,
            z_merged=merged_z,
            known_mask=known_mask,
            nonzero_mask=nonzero_mask,
            z_nz=z_nz,
        )
        self._assembly = assembly
        self._assembly_key = z_all
        logger.debug(
            "Cached overall-transfer assembly: %d boundaries, %d cut merge(s), "
            "%d converging element(s).",
            len(boundaries), len(merges), len(constraints),
        )
        return assembly

    # ------------------------------------------------------------------------------
    # Transfer matrices
    # ------------------------------------------------------------------------------

    @staticmethod
    def transfer_along_plan(plan: Sequence[PropagationStep], omega: float) -> Matrix:
        """
        Chain the transfer matrices of a resolved propagation plan.

        Parameters
        ----------
        plan : Sequence[PropagationStep]
            The plan, from :meth:`TopologyHandler.propagation_plan`.
        omega : float
            Vibration frequency [rad/s].

        Returns
        -------
        Matrix
            The 12x12 product, from the plan's origin to the last element's output.
        """
        u_chain = _I12
        for step in plan:
            u_step = (
                step.elem.u(step.input_pos, step.output_pos, omega)
                if step.is_main_input
                else step.elem.u_extract(step.input_pos, step.output_pos)
            )
            u_chain = u_step @ u_chain
        return u_chain

    def transfer_mat_along_path(self, path: Sequence[EntityID], omega: float) -> Matrix:
        """
        Chain the transfer matrices along a path of node IDs.

        Parameters
        ----------
        path : Sequence[EntityID]
            The path, as returned by :meth:`TopologyHandler.resolve_branch_up_to`.
        omega : float
            Vibration frequency [rad/s].

        Returns
        -------
        Matrix
            The 12x12 product, from the output state vector of the path's origin to the
            output state vector of its tail.
        """
        return self.transfer_along_plan(self.topology.propagation_plan(path), omega)

    def _geometric_constraint_columns(self, assembly: _Assembly, omega: float) -> List[Matrix]:
        """
        Column blocks enforcing kinematic compatibility where branches converge.

        Parameters
        ----------
        assembly : _Assembly
            The cached frequency-independent structure.
        omega : float
            Vibration frequency [rad/s].

        Returns
        -------
        List[Matrix]
            One column block per boundary, or an empty list if the topology has no
            converging elements.
        """
        if not assembly.constraints:
            return []

        # Initialize columns for every boundary; the root gets zeros automatically.
        g_cols_dict: Dict[EntityID, List[Matrix]] = {b_id: [] for b_id in assembly.boundaries}

        for constraint in assembly.constraints:
            k_mats = {
                t_id: h_mat @ self.transfer_along_plan(plan, omega)
                for t_id, (plan, h_mat) in constraint.tip_terms.items()
            }
            for ref_tips, other_tips in constraint.pairs:
                for b_id in assembly.boundaries:
                    if b_id in ref_tips:
                        g_cols_dict[b_id].append(k_mats[b_id])
                    elif b_id in other_tips:
                        g_cols_dict[b_id].append(-k_mats[b_id])
                    else:
                        g_cols_dict[b_id].append(_ZERO_6x12)

        return [np.vstack(g_cols_dict[b_id], dtype=np.float64) for b_id in assembly.boundaries]

    @staticmethod
    def _apply_merges(blocks: List[Matrix], merges: Sequence[Tuple[int, int, Matrix]]) -> List[Matrix]:
        """
        Combine column blocks according to the cut-point relations.

        Parameters
        ----------
        blocks : List[Matrix]
            One block per boundary, in boundary order. Consumed in place.
        merges : Sequence[tuple]
            ``(keep_index, drop_index, relation_matrix)`` triples, applied in order.

        Returns
        -------
        List[Matrix]
            The surviving blocks.
        """
        for idx1, idx2, mat in merges:
            blocks[idx1] = blocks[idx1] + blocks[idx2] @ mat
            blocks.pop(idx2)
        return blocks

    def overall_transfer_mat(self, omega: float) -> Tuple[Matrix, Vector, Vector, List[EntityID]]:
        """
        Assemble the reduced overall transfer equation at one frequency.

        Parameters
        ----------
        omega : float
            Vibration frequency [rad/s].

        Returns
        -------
        tuple
            ``(u_red, f, z_merged, remaining_boundaries)``. The reduced system to solve
            is ``u_red @ z_red = f``; note the sign of ``f`` is already folded in.

        Raises
        ------
        ValueError
            If the boundary conditions leave the reduced system non-square, meaning the
            system is under- or over-constrained.
        """
        assembly = self._get_assembly()

        # --- TRANSFER MATRIX COMPUTATION ---
        t_mats: List[Matrix] = [-_I12]  # init with root
        t_mats.extend(self.transfer_along_plan(plan, omega) for plan in assembly.tip_plans)

        # --- GEOMETRIC MATRIX COMPUTATION ---
        g_cols = self._geometric_constraint_columns(assembly, omega)

        # --- OVERALL MATRIX ASSEMBLY ---
        # Condense columns using the cut-point relations
        t_mats = self._apply_merges(t_mats, assembly.merges)
        if g_cols:
            g_cols = self._apply_merges(g_cols, assembly.merges)
            u_all = np.block([t_mats, g_cols])
        else:
            u_all = np.block(t_mats)

        # Eliminate columns corresponding to known boundary conditions...
        u_red = u_all[:, ~assembly.known_mask]
        # ... and move columns corresponding to known non-zero boundary conditions into a load vector
        u_nz = u_all[:, assembly.nonzero_mask]
        f = -u_nz @ assembly.z_nz  # CAREFUL: the (-) is included here, so we solve Uz=f, not Uz+f=0

        if u_red.shape[0] != u_red.shape[1]:
            err_msg = "Invalid boundary conditions: The system is either under-constrained or over-constrained"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return (
            u_red.astype(np.float64),
            f.astype(np.float64),
            assembly.z_merged,
            assembly.merged_boundaries,
        )

    # ------------------------------------------------------------------------------
    # State reconstruction
    # ------------------------------------------------------------------------------

    def reconstruct_boundary_states(self, z_red: Vector, z_merged: Vector,
                                    rem_boundary_ids: List[EntityID]) -> Dict[EntityID, Vector]:
        """
        Rebuild every boundary's full state vector from the reduced solution.

        Parameters
        ----------
        z_red : Vector
            The solution of the reduced overall transfer equation.
        z_merged : Vector
            The merged boundary state vector, with ``None`` for each unknown.
        rem_boundary_ids : List[EntityID]
            The boundaries surviving the cut-point merges, in order.

        Returns
        -------
        Dict[EntityID, Vector]
            Each boundary's 12-component state, in topology order.
        """
        # Fill in the merged overall state vector with the computed values from the reduced state vector
        z_red_full = np.empty(z_merged.shape, dtype=np.float64)
        z_red_iter = iter(z_red)
        for i, x in enumerate(z_merged):
            if x is None:
                z_red_full[i] = next(z_red_iter)
            else:
                z_red_full[i] = x
        # Decompose z_red_full into its component state vectors
        z_red_full = z_red_full.reshape((len(rem_boundary_ids), 12))
        rem_boundaries = {rem_boundary_ids[i]: sv for i, sv in enumerate(z_red_full)}
        # Expand the reduced vector back to full size by reintroducing eliminated virtual boundaries
        eliminated_boundaries = {}
        for c in self.topology.cut_points:
            eliminated_boundaries[c.b_id2] = c.mat @ rem_boundaries[c.b_id1]
        # We need to build the final dict in the correct order
        bound_svs = {}
        for b_id in self.topology.boundaries:
            if b_id in rem_boundary_ids:
                bound_svs[b_id] = rem_boundaries[b_id]
            else:
                bound_svs[b_id] = eliminated_boundaries[b_id]

        return bound_svs

    def propagate_state(
            self,
            omega: float,
            z_red: Vector,
            z_merged: Vector,
            rem_boundary_ids: List[EntityID],
            rtol: float = 1e-4,
    ) -> Dict[EntityID, Vector]:
        """
        Propagate the state from the tips, through every element, up to the root.

        Also verifies that the propagated state satisfies the root boundary condition,
        which is the arrival check on the whole assembly.

        Parameters
        ----------
        omega : float
            Vibration frequency [rad/s].
        z_red : Vector
            The vector solved for in the reduced overall transfer equation.
        z_merged : Vector
            Boundary conditions remaining after merging cut points.
        rem_boundary_ids : List[EntityID]
            IDs of those remaining boundaries, in order.
        rtol : float
            Relative tolerance of the root boundary check.

        Returns
        -------
        Dict[EntityID, Vector]
            The state vector at every node of the tree.

        Raises
        ------
        RuntimeError
            If any element was not reached by the sweep.
        ValueError
            If the propagated state does not match the root boundary condition.
        """
        logger.debug(f"Computing internal states at {omega:.3e} rad/s")

        topology = self.topology
        boundary_svs = self.reconstruct_boundary_states(z_red, z_merged, rem_boundary_ids)
        root_sv = boundary_svs.pop(topology.root.b_id)
        tip_svs = boundary_svs

        # Initialize search heads, traversal tracking, and state vectors for each element
        search_heads = deque(topology.tips)
        searched = {k: False for k in topology.element_ids}
        state_vecs = defaultdict(lambda: np.zeros(12), tip_svs)

        while search_heads:
            head_id = search_heads.popleft()
            if topology.is_tip(head_id):
                head_info = topology.get_tip_info(head_id)
            else:
                head_info = topology.get_element_info(head_id)
            next_id = head_info.downstream

            if not topology.has_element(next_id):
                # Also synonymous with the next node being the root
                continue
            # Info about the next element
            next_elem_info = topology.get_element_info(next_id)
            # I/O for next_elem
            next_input_port_idx: int = topology.tree[head_id][next_id]['input_port']
            next_input_pos = next_elem_info.ports[next_input_port_idx].pos
            next_output_pos = next_elem_info.output_port.pos
            # Add to the output state vector of the next element
            if next_input_port_idx == next_elem_info.main_input_idx:
                sv = next_elem_info.obj.u(next_input_pos, next_output_pos, omega) @ state_vecs[head_id]
            else:
                sv = next_elem_info.obj.u_extract(next_input_pos, next_output_pos) @ state_vecs[head_id]
            state_vecs[next_id] += sv
            # If the next element has already been used as a head, we don't need to add it again
            if not searched[next_id]:
                search_heads.append(next_id)
                searched[next_id] = True

        # Verify that all elements have been swept
        searched_count = sum(searched.values())
        msg = f"Swept through ({searched_count}/{len(searched)}) elements."
        if searched_count != len(searched):
            logger.error(msg)
            raise RuntimeError(msg)
        logger.debug(msg)

        # Verify that the propagated state vector satisfies the boundary conditions at the root
        last_elem_id = next(topology.tree.predecessors(topology.root.b_id))

        # Use np.allclose with an absolute tolerance to handle mixed-unit zero crossings
        if np.allclose(state_vecs[last_elem_id], root_sv, rtol=rtol, atol=1e-4):
            logger.debug("Propagated state matches root boundary state.")
        else:
            # If it fails, manually calculate the max absolute difference for logging
            abs_diff = float(np.max(np.abs(state_vecs[last_elem_id] - root_sv)))
            err_msg = f"Propagated state doesn't match root boundary state. Max absolute error = {abs_diff:.3e}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Add the root to the end of the dict
        state_vecs[topology.root.b_id] = root_sv

        logger.debug(f"Successfully computed internal states at {omega:.3e} rad/s.")

        return state_vecs

    # ------------------------------------------------------------------------------
    # Eigenvalue problem
    # ------------------------------------------------------------------------------

    def _sigma_min(self, omega: float) -> float:
        """
        Smallest singular value of the reduced overall transfer matrix.

        Parameters
        ----------
        omega : float
            Vibration frequency [rad/s].

        Returns
        -------
        float
            ``sigma_min(U_all(omega))``, which touches zero at an eigenfrequency.
        """
        u = self.overall_transfer_mat(omega)[0]
        return np.linalg.svd(u, compute_uv=False)[-1]

    def natural_modes(
            self,
            n_modes: int,
            omega_min: float = 0.0,
            omega_max: float = 1000.0,
            search_res: int = 10000,
            rtol: float = 1e-5,
            mtol: float = 1e-5,
            max_multiplicity: int = 6,
    ) -> List[Mode]:
        """
        Find the system's natural modes, resolving repeated eigenfrequencies in full.

        An eigenfrequency is repeated whenever the overall transfer matrix has a null
        space of dimension greater than one there, which symmetric systems produce
        routinely. E.g. a body on springs of equal stiffness in y and z has a
        two-dimensional eigenspace at one frequency. Every basis vector of that
        eigenspace is a genuine, physically distinct mode, so all of them are returned.
        Taking only one would silently discard part of the system's response without any
        visible symptom.

        Note that ``n_modes`` counts *distinct eigenfrequencies*: the returned list is
        longer than ``n_modes`` when some of them are repeated.

        Parameters
        ----------
        n_modes : int
            The number of distinct natural frequencies to retain.
        omega_min : float
            The minimum frequency for the search.
        omega_max : float
            The maximum frequency for the search.
        search_res : int
            The number of points to search between ``omega_min`` and ``omega_max``.
        rtol : float
            Relative tolerance on the reciprocal condition number of the transfer matrix.
            A candidate is accepted when the smallest singular value is less than ``rtol``
            times the largest.
        mtol : float
            Relative tolerance for merging near-identical frequencies.
        max_multiplicity : int
            Cap on the multiplicity detected at any one eigenfrequency, guarding against
            a badly conditioned transfer matrix being read as a large degenerate
            eigenspace.

        Returns
        -------
        List[Mode]
            The modes, sorted by frequency. Repeated eigenfrequencies contribute one
            entry per eigenvector, so ``len(result)`` may exceed ``n_modes``.

        Notes
        -----
        The sweep covers strictly positive frequencies only. This is a modeling choice,
        not a numerical one: the element transfer matrices are perfectly well-defined at
        ``omega = 0``, where they reduce to their static form, but a root there is a
        rigid-body freedom rather than a vibration mode, and a peak sitting exactly on
        the boundary of the sweep cannot be bracketed for refinement. A system with a
        genuine rigid-body mode should query it directly with
        ``eigenvectors_at(0.0)``.
        """
        omega = np.linspace(omega_min, omega_max, search_res)
        omega = omega[omega > 0]
        sigma = np.array([self._sigma_min(w) for w in omega])
        # Find rough peaks corresponding to the smallest singular values
        prominence = 5e-2 * np.max(sigma)
        candidates, _ = find_peaks(-sigma, prominence=prominence)

        modes: List[Mode] = []
        found_frequencies: List[float] = []
        # Refine candidates
        for idx in candidates:
            if idx == 0 or idx == len(omega) - 1:
                continue
            # Minimize the singular values that approach zero
            res = minimize_scalar(
                self._sigma_min,
                bounds=(omega[idx - 1], omega[idx + 1]),
                method="bounded",
            )
            refined_omega: float = res.x
            # It shouldn't happen if the sweep is fine enough, but we should avoid duplicating modes
            if any(np.isclose(refined_omega, f, rtol=mtol) for f in found_frequencies):
                continue

            # Reciprocal condition number of the transfer matrix at the refined frequency
            s = np.linalg.svd(self.overall_transfer_mat(refined_omega)[0], compute_uv=False)
            rcond = s[-1] / s[0]

            logger.debug(f"Mode candidate at {float(refined_omega):.3e} rad/s.")
            logger.debug(f"sigma_min = {s[-1]:.6e}, sigma_max = {s[0]:.6e}, rcond = {rcond:.6e}")
            # If rcond passes the tolerance, keep every eigenvector at this frequency --
            # the null space may be more than one-dimensional (a repeated eigenfrequency).
            if rcond < rtol:
                found_frequencies.append(refined_omega)
                modes.extend(self.eigenvectors_at(refined_omega, max_multiplicity))

        modes.sort(key=lambda x: x.frequency)
        found_frequencies.sort()

        logger.info(
            f"Found {len(modes)} natural modes across {len(found_frequencies)} distinct "
            f"eigenfrequencies between {omega_min:.3e} rad/s and {omega_max:.3e} rad/s."
        )
        for mode in modes:
            logger.debug(f"Mode at {mode.frequency:.3e} rad/s.")

        # n_modes counts distinct eigenfrequencies, so truncate on frequency rather than on
        # position -- slicing the list directly would cut a repeated cluster in half and
        # hand back an incomplete eigenspace.
        if len(found_frequencies) > n_modes:
            cutoff = found_frequencies[n_modes - 1]
            modes = [m for m in modes if m.frequency <= cutoff]

        return modes

    def eigenvectors_at(self, omega: float, max_multiplicity: int = 6) -> List[Mode]:
        """
        Every eigenvector of the system at an already-known eigenfrequency.

        :meth:`natural_modes` calls this once it has refined each frequency in its
        sweep; it is exposed separately for a frequency obtained some other way --
        measured, read off a chart, or known a priori such as the ``omega = 0`` of a
        rigid-body freedom -- without repeating the search.

        The number of eigenvectors is the multiplicity of the frequency, read off the
        singular-value spectrum by :func:`null_space_dimension`. They are orthonormal in
        R^n but not yet orthogonal in the augmented inner product; see
        :func:`dynaramp.mstmm.response.augmented_modes` for that step.

        Parameters
        ----------
        omega : float
            A known eigenfrequency [rad/s].
        max_multiplicity : int
            Cap on the detected multiplicity.

        Returns
        -------
        List[Mode]
            One mode per null-space direction.
        """
        u, _, z_merged, rem_bounds = self.overall_transfer_mat(omega)
        _, sigma, vh = np.linalg.svd(u)
        dim = min(null_space_dimension(sigma, max_multiplicity), vh.shape[0])

        # Homogeneous problem: external forcing and non-zero boundary values are off.
        z_homogeneous = np.array([None if v is None else 0.0 for v in z_merged])

        if dim > 1:
            logger.info(f"Eigenfrequency {omega:.6e} rad/s has multiplicity {dim}.")

        return [
            Mode(
                frequency=omega,
                internal_states=dict(self.propagate_state(
                    omega=omega, z_red=vh[-(i + 1)],
                    z_merged=z_homogeneous, rem_boundary_ids=rem_bounds,
                )),
            )
            for i in range(dim)
        ]

    # ------------------------------------------------------------------------------
    # Mode shapes and the augmented inner product
    # ------------------------------------------------------------------------------

    def mode_shapes_at(
            self,
            modes: Sequence[Mode],
            e_id: EntityID,
            position: Vector,
    ) -> Matrix:
        """
        The shapes of several modes at one material point.

        A mode stores the state at each element's main input; pushing it across the
        element's own transfer matrix up to ``position`` gives the state anywhere inside
        it -- along a beam, or at any point of a rigid body. This is the quantity that a
        generalized coordinate weights to give physical motion, and the quantity a point
        load is dotted into to give a modal force.

        The batch form is the primitive: it resolves the element and its main input once
        for the whole set, and it is what the modal basis and the guide modal field are
        both built on.

        Parameters
        ----------
        modes : Sequence[Mode]
            The modes to evaluate.
        e_id : EntityID
            Element the point belongs to.
        position : Vector
            Point coordinates in the element's local frame.

        Returns
        -------
        Matrix
            An ``(n_modes, 6)`` array of ``[x, y, z, theta_x, theta_y, theta_z]``, in
            the global frame.
        """
        pred_id, input_pos = self.topology.main_input_of(e_id)
        elem = self.topology.get_element(e_id)
        pos = np.asarray(position, dtype=np.float64)
        return np.array(
            [
                (elem.u(input_pos, pos, m.frequency) @ m.internal_states[pred_id])[0:6]
                for m in modes
            ],
            dtype=np.float64,
        )

    def mode_shape(self, mode: Mode, e_id: EntityID, position: Vector) -> Vector:
        """
        A mode's shape at an arbitrary material point.

        The 6x1 kinematic state ``[x, y, z, theta_x, theta_y, theta_z]``, in the global
        frame. Single-mode form of :meth:`mode_shapes_at`.

        Parameters
        ----------
        mode : Mode
            A mode from :meth:`natural_modes`.
        e_id : EntityID
            Element the point belongs to.
        position : Vector
            Point coordinates in the element's local frame.

        Returns
        -------
        Vector
            The 6-component mode shape at that point.
        """
        return self.mode_shapes_at((mode,), e_id, position)[0]

    def _element_extent(self, e_id: EntityID) -> Tuple[Vector, Vector]:
        """
        The pair of positions spanning an element, for the purpose of integrating its mass.

        Parameters
        ----------
        e_id : EntityID
            The element.

        Returns
        -------
        tuple of Vector
            From its main input to its center of mass (rigid body) or its output port
            (beam).
        """
        info = self.topology.get_element_info(e_id)
        output_pos = info.obj.com_pos if isinstance(info.obj, RigidBody) else info.output_port.pos
        return info.main_input_pos, output_pos

    def modal_product(self, mode_k: Mode, mode_p: Mode) -> float:
        """
        The augmented inner product ``<M V^k, V^p>`` over the whole system (Rui 3.48).

        For ``mode_k is mode_p`` this is the modal mass ``M_p``. For distinct modes at
        *different* eigenfrequencies orthogonality makes it vanish; for distinct modes
        sharing a repeated eigenfrequency it generally does not, which is why an
        orthogonalization step is needed before the modal equations decouple (see
        :func:`dynaramp.mstmm.response.augmented_modes`).

        Parameters
        ----------
        mode_k : Mode
            First mode.
        mode_p : Mode
            Second mode. Must have been computed at the same frequency as ``mode_k`` for
            an off-diagonal result to mean anything.

        Returns
        -------
        float
            The scalar inner product.
        """
        total = 0.0
        for e_id in self.topology.element_ids:
            info = self.topology.get_element_info(e_id)
            pred_id = info.main_input_pred
            input_pos, output_pos = self._element_extent(e_id)
            total += float(info.obj.modal_product(
                input_pos, output_pos, mode_k.frequency,
                mode_k.internal_states[pred_id], mode_p.internal_states[pred_id],
            ))
        return total

    def calc_system_modal_masses(self, modes: List[Mode]) -> Vector:
        """
        The modal mass of the whole system for each mode.

        The diagonal of :meth:`modal_product`.

        Parameters
        ----------
        modes : List[Mode]
            The modes to evaluate.

        Returns
        -------
        Vector
            One modal mass per mode.
        """
        return np.array([self.modal_product(m, m) for m in modes], dtype=np.float64)

    def get_system_modal_matrices(
            self,
            modes: List[Mode],
            rayleigh: Tuple[float, float] | None = None,
    ) -> Tuple[Matrix, Matrix] | Tuple[Matrix, Matrix, Matrix]:
        """
        The diagonal modal mass, stiffness and (optionally) damping matrices.

        Parameters
        ----------
        modes : List[Mode]
            The modes forming the basis.
        rayleigh : tuple of float | None
            Rayleigh coefficients ``(alpha, beta)``. When given, a damping matrix is
            returned as well.

        Returns
        -------
        tuple of Matrix
            ``(M, K)``, or ``(M, K, C)`` when ``rayleigh`` is supplied.

        Notes
        -----
        The damping matrix is built from
        :func:`dynaramp.common.modal.rayleigh_modal_damping` as
        ``C = diag(c_p M_p)``, which expands to exactly ``alpha M + beta K`` and stays
        finite for a rigid-body mode.
        """
        masses = self.calc_system_modal_masses(modes)
        frequencies = np.array([mode.frequency for mode in modes], dtype=np.float64)

        m_mat = np.diag(masses).astype(np.float64)
        k_mat = np.diag(frequencies ** 2 * masses).astype(np.float64)
        if rayleigh is None:
            return m_mat, k_mat

        alpha, beta = rayleigh
        c_mat = np.diag(rayleigh_modal_damping(frequencies, alpha, beta) * masses).astype(np.float64)
        return m_mat, k_mat, c_mat
