from __future__ import annotations
import logging
from dataclasses import dataclass

from typing import Tuple, List, Dict, Sequence
from collections import deque, defaultdict

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

from ..common.types import EntityID, Vector, Matrix
from .topology import TopologyHandler
from .element_lib import RigidBody

logger = logging.getLogger(__name__)

# TODO: better error raising, custom exceptions?
# TODO: Clean up z_rem, z_red, rem_boundaries clutter for state propagation
# TODO: Add rich results classes


@dataclass
class Mode:
    frequency: float
    internal_states: Dict[EntityID, Vector]


def null_space_dimension(sigma: Vector, max_dim: int = 6) -> int:
    """
    Estimate the dimension of the null space of the overall transfer matrix from its
    singular-value spectrum -- that is, the multiplicity of the eigenfrequency.

    A simple absolute threshold is unreliable here: at an eigenfrequency the smallest
    singular value can land anywhere from 1e-5 to 1e-13 depending on conditioning. What
    *is* robust is the gap. The null-space singular values sit orders of magnitude below
    the rest, so the spectrum is cut at its largest multiplicative jump.

    :param sigma: Singular values in descending order (as returned by ``np.linalg.svd``).
    :param max_dim: Largest multiplicity to consider. Guards against a badly
        rank-deficient matrix being read as a hugely degenerate eigenspace.
    :return: The number of trailing singular values belonging to the null space.
    """
    tail = np.asarray(sigma[-(max_dim + 1):], dtype=np.float64)
    ascending = tail[::-1]                      # walk up from the smallest
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(ascending[:-1] > 0, ascending[1:] / ascending[:-1], np.inf)
    if ratios.size == 0:
        return 1
    return int(np.argmax(ratios)) + 1


class System:

    def __init__(self, system_topo: TopologyHandler):
        self.topology = system_topo

    def transfer_mat_along_path(self, path: Sequence[EntityID], omega: float) -> Matrix:
        # Get transfer matrix from the output state vector of the path's origin to the output vector of the tail.
        u_chain = np.identity(12, dtype=np.float64)

        for i in range(len(path) - 1):
            e1, e2 = path[i], path[i + 1]
            # Because of the way we use the path, e2 can never be a boundary (root here)
            e2_info = self.topology.get_element_info(e2)
            e2_input_port_idx: int = self.topology.tree[e1][e2]['input_port']
            # Retrieve the port positions
            e2_in_pos: Vector = e2_info.ports[e2_input_port_idx].pos
            e2_out_pos: Vector = e2_info.output_port.pos
            # Here we apply the transfer matrix for the element e2 based on its input and output ports
            if e2_input_port_idx == e2_info.main_input_idx:
                u_chain = e2_info.obj.u(e2_in_pos, e2_out_pos, omega) @ u_chain
            else:
                u_chain = e2_info.obj.u_extract(e2_in_pos, e2_out_pos) @ u_chain

        return u_chain.astype(np.float64)

    def _get_geometric_constraints(self, boundaries: List[EntityID], omega: float) -> List[Matrix]:
        """

        :param boundaries:
        :param omega:
        :return:
        """
        multi_input_elems = [e_id for e_id in self.topology.elements if self.topology.tree.in_degree(e_id) > 1]
        # Initialize dictionary columns for every boundary (guarantees Root gets zeros automatically)
        g_cols_dict = {b_id: [] for b_id in boundaries}

        for m_id in multi_input_elems:
            m_info = self.topology.get_element_info(m_id)
            # Retrieve the N incoming branches
            preds = list(self.topology.tree.predecessors(m_id))
            if len(preds) <= 1:
                continue

            # Group tips by the immediate incoming branch they pass through
            tips_by_branch = {p: [] for p in preds}
            k_mats = {}

            for t_id in m_info.upstream_tips:
                path = self.topology.resolve_branch_up_to(src=t_id, tgt=m_id)
                branch_node = path[-1]
                if branch_node in tips_by_branch:
                    tips_by_branch[branch_node].append(t_id)

                # Extract the pure kinematics transformation matrix (no negative signs!)
                u_chain = self.transfer_mat_along_path(path, omega)
                input_port_idx: int = self.topology.tree[branch_node][m_id]['input_port']
                input_port_pos: Vector = m_info.ports[input_port_idx].pos
                main_port_pos: Vector = m_info.ports[m_info.main_input_idx].pos

                k_mat = m_info.obj.h(ref_pos=main_port_pos, input_pos=input_port_pos) @ u_chain

                k_mats[t_id] = k_mat

            # Generate exactly (N - 1) sets of geometric constraints
            ref_pred = preds[0]
            for other_pred in preds[1:]:
                for b_id in boundaries:
                    if b_id in tips_by_branch[ref_pred]:
                        g_cols_dict[b_id].append(k_mats[b_id])
                    elif b_id in tips_by_branch[other_pred]:
                        g_cols_dict[b_id].append(-k_mats[b_id])
                    else:
                        g_cols_dict[b_id].append(np.zeros((6, 12), dtype=np.float64))

        # Convert dictionary to ordered list of column blocks
        has_g_eqs = any(len(blocks) > 0 for blocks in g_cols_dict.values())
        if has_g_eqs:
            g_cols = [np.vstack(g_cols_dict[b_id], dtype=np.float64) for b_id in boundaries]
        else:
            g_cols = []

        return g_cols

    def _merge_columns(
            self,
            boundaries: List[EntityID],
            z_all: Vector,
            t_mats: List[Matrix],
            g_cols: List[Matrix] | None = None
    ) -> Tuple[List[EntityID], Vector, List[Matrix], List[Matrix]]:
        """
        Combine columns and boundary state vectors using the cutting point equation.
        :param boundaries:
        :param z_all:
        :param t_mats:
        :param g_cols:
        :return:
        """
        merged_boundaries = boundaries.copy()
        merged_z_all = z_all.copy()
        merged_t_mats = t_mats.copy()
        merged_g_cols = g_cols.copy() if g_cols is not None else []

        for cut in self.topology.cut_points:
            # Find the index of the cut's virtual boundaries.
            try:
                idx1 = next(i for i, b_id in enumerate(merged_boundaries) if b_id == cut.b_id1)
                idx2 = next(i for i, b_id in enumerate(merged_boundaries) if b_id == cut.b_id2)
            except StopIteration:
                err_msg = f"Cut point boundaries [{cut.b_id1}] or [{cut.b_id2}] not found in tips."
                logger.error(err_msg)
                raise ValueError(err_msg)

            merged_t_mats[idx1] += merged_t_mats[idx2] @ cut.mat
            merged_t_mats.pop(idx2)

            if merged_g_cols:
                merged_g_cols[idx1] += merged_g_cols[idx2] @ cut.mat
                merged_g_cols.pop(idx2)

            merged_boundaries.pop(idx2)
            mask = np.ones_like(merged_z_all, dtype=bool)
            mask[(12 * idx2):(12 * (idx2 + 1))] = False
            merged_z_all = merged_z_all[mask]

        return merged_boundaries, merged_z_all, merged_t_mats, merged_g_cols

    def overall_transfer_mat(self, omega: float) -> Tuple[Matrix, Vector, Vector, List[EntityID]]:
        """

        :param omega:
        :return:
        """
        remaining_boundaries = self.topology.boundaries  # Tracks what boundary vectors are actually still in z_red
        # --- TRANSFER MATRIX COMPUTATION ---
        t_mats = [- np.identity(12)]  # init with root
        for t_id in self.topology.tips:
            path = self.topology.resolve_branch_up_to(src=t_id, tgt=self.topology.root.b_id)
            t_mats.append(self.transfer_mat_along_path(path, omega))

        # --- GEOMETRIC MATRIX COMPUTATION ---
        g_cols = self._get_geometric_constraints(remaining_boundaries, omega)

        # --- OVERALL MATRIX ASSEMBLY ---
        # Condense columns using the cut-point relations
        remaining_boundaries, z_merged, t_mats, g_cols = self._merge_columns(
            remaining_boundaries, self.topology.z_all, t_mats, g_cols
        )
        # Assemble full-sized overall transfer matrix
        if g_cols:
            u_all = np.block([
                t_mats,
                g_cols
            ])
        else:
            u_all = np.block(t_mats)

        # Handle known boundary conditions
        known_mask = np.array([x is not None for x in z_merged])
        nonzero_mask = np.array([x != 0 for x in z_merged]) & known_mask
        # Eliminate columns corresponding to known boundary conditions...
        u_red = u_all[:, ~ known_mask]
        # ... and move columns corresponding to known non-zero boundary conditions into a load vector
        u_nz = u_all[:, nonzero_mask]
        z_nz = z_merged[nonzero_mask].astype(np.float64)
        f = - u_nz @ z_nz  # CAREFUL: the (-) is included here, so we need to solve Uz=f, not Uz+f=0

        if u_red.shape[0] != u_red.shape[1]:
            err_msg = "Invalid boundary conditions: The system is either under-constrained or over-constrained"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return u_red.astype(np.float64), f.astype(np.float64), z_merged, remaining_boundaries

    def reconstruct_boundary_states(self, z_red: Vector, z_merged: Vector,
                                    rem_boundary_ids: List[EntityID]) -> Dict[EntityID, Vector]:
        # Fill in the merged overall state vector with the computed values from the reduced state vector
        z_red_full = np.empty(z_merged.shape, dtype=np.float64)
        z_red_iter = iter(z_red)
        for i, x in enumerate(z_merged):
            if x is None:
                z_red_full[i] = next(z_red_iter)
            else:
                z_red_full[i] = x
        # Decompose z_red_full into it's component state vectors
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
            rtol: float = 1e-4
    ) -> Dict[EntityID, Vector]:
        """
        Propagate state from the tips of the system, through the elements
        (bodies and hinges) and up to the root.
        Verify that the propagated state vector satisfies the boundary conditions at the root.
        :param omega:
        :param z_red: Vector you solve for in the reduced overall transfer equation
        :param z_merged: Vector of remaining boundary conditions after merging cutting points
        :param rem_boundary_ids: Indices of remaining boundary conditions
        :param rtol:
        :return:
        """

        logger.debug(f"Computing internal states at {omega:.3e} rad/s")

        boundary_svs = self.reconstruct_boundary_states(z_red, z_merged, rem_boundary_ids)
        root_sv = boundary_svs.pop(self.topology.root.b_id)
        tip_svs = boundary_svs

        # Initialize search heads, traversal tracking, and state vectors for each element
        search_heads = deque([t for t in self.topology.tips])
        searched = {k: False for k in self.topology.elements}
        state_vecs = defaultdict(
            lambda: np.zeros(12),
            tip_svs
        )

        while search_heads:
            head_id = search_heads.popleft()
            if head_id in self.topology.tips:
                head_info = self.topology.get_tip_info(head_id)
            else:
                head_info = self.topology.get_element_info(head_id)
            next_id = head_info.downstream

            if next_id not in self.topology.elements:
                # Also synonymous with the next node being the root
                continue
            # Info about the next element
            next_elem_info = self.topology.get_element_info(next_id)
            # I/O for next_elem
            next_input_port_idx: int = self.topology.tree[head_id][next_id]['input_port']
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

        # Verify that all elements have been sweeped
        searched_count = list(searched.values()).count(True)
        msg = f"Swept through ({searched_count}/{len(searched)}) elements."
        if searched_count != len(searched):
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.debug(msg)

        # Verify that the propagated state vector satisfies the boundary conditions at the root
        last_elem_id = next(self.topology.tree.predecessors(self.topology.root.b_id))

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
        state_vecs[self.topology.root.b_id] = root_sv

        logger.info(f"Successfully computed internal states at {omega:.3e} rad/s.")

        return state_vecs

    def _sigma_min(self, omega: float) -> float:
        """
        Helper for retrieving the smallest singular value of an SVD of U_all(w).
        :param omega:
        :return:
        """
        u = self.overall_transfer_mat(omega)[0]
        return np.linalg.svd(u, compute_uv=False)[-1]

    def natural_modes(
            self,
            n_modes: int, omega_min=0,
            omega_max: int = 1000,
            search_res: int = 10000,
            rtol: float = 1e-5,
            mtol: float = 1e-5,
            max_multiplicity: int = 6,
    ) -> List[Mode]:
        """
        Find the system's natural modes, resolving repeated eigenfrequencies in full.

        An eigenfrequency is repeated whenever the overall transfer matrix has a null
        space of dimension greater than one there, which symmetric systems produce
        routinely -- a body on springs of equal stiffness in y and z has a two-dimensional
        eigenspace at one frequency. Every basis vector of that eigenspace is a genuine,
        physically distinct mode, so all of them are returned. Taking only one would
        silently discard part of the system's response without any visible symptom.

        Note that ``n_modes`` counts *distinct eigenfrequencies*: the returned list is
        longer than ``n_modes`` when some of them are repeated.

        Parameters
        ----------
        n_modes: int
            The number of distinct natural frequencies to retain.
        omega_min: float
            The minimum frequency for the search.
        omega_max: float
            The maximum frequency for the search.
        search_res: int
            The number of points to search between omega_min and omega_max.
        rtol: float
            Relative tolerance for the reciprocal condition number of the transfer matrix.
            If the smallest singular value is less than rtol times the largest singular value,
            the mode is considered valid.
        mtol: float
            Absolute tolerance for merging modes that are very close in frequency.
            If two modes are within mtol of each other, they will be considered the same mode and only one will be kept.
        max_multiplicity: int
            Cap on the multiplicity detected at any one eigenfrequency, guarding against a
            badly conditioned transfer matrix being read as a large degenerate eigenspace.

        Returns
        -------
        List[Mode]
            The modes, sorted by frequency. Repeated eigenfrequencies contribute one
            entry per eigenvector, so ``len(result)`` may exceed ``n_modes``.
        """

        omega = np.linspace(omega_min, omega_max, search_res)
        # Only retain strictly positive frequencies
        # We want to ignore rigid modes (and avoid dividing by zero) and negative frequencies
        omega = omega[omega > 0]
        sigma = np.array([self._sigma_min(w) for w in omega])
        # Find rough peaks corresponding to the smallest singular values
        prominence = 5e-2 * np.max(sigma)
        candidates, _ = find_peaks(-sigma, prominence=prominence)

        modes = []
        found_frequencies: List[float] = []
        # Refine candidates
        for idx in candidates:
            if idx == 0 or idx == len(omega) - 1:
                continue
            # Minimize the singular values that approach zero
            res = minimize_scalar(
                self._sigma_min,
                bounds=(omega[idx - 1], omega[idx + 1]),
                method="bounded"
            )
            refined_omega: float = res.x
            # It shouldn't happen if the sweep is fine enough, but we should avoid duplicating modes
            if any(np.isclose(refined_omega, f, rtol=mtol) for f in found_frequencies):
                continue

            # Reciprocal condition number of the transfer matrix at the refined frequency
            s = np.linalg.svd(self.overall_transfer_mat(refined_omega)[0], compute_uv=False)
            rcond = s[-1] / s[0]

            logger.info(f"Mode candidate at {float(refined_omega):.3e} rad/s.")
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

    def solve(self):
        # TODO: move some of the logic here
        raise NotImplementedError

    def mode_shape(self, mode: Mode, e_id: EntityID, position: Vector) -> Vector:
        """
        A mode's shape at an arbitrary material point: the 6x1 kinematic state
        ``[x, y, z, theta_x, theta_y, theta_z]``, in the global frame.

        A mode stores the state at each element's main input; pushing it across the
        element's own transfer matrix up to ``position`` gives the state anywhere inside
        it -- along a beam, or at any point of a rigid body. This is the quantity that a
        generalized coordinate weights to give physical motion, and the quantity a point
        load is dotted into to give a modal force.

        :param mode: A mode from :meth:`natural_modes`.
        :param e_id: Element the point belongs to.
        :param position: Point coordinates in the element's local frame.
        :return: The 6-component mode shape at that point.
        """
        pred_id, input_pos = self.topology.main_input_of(e_id)
        elem = self.topology.get_element(e_id)
        u = elem.u(input_pos, np.asarray(position, dtype=np.float64), mode.frequency)
        return (u @ mode.internal_states[pred_id])[0:6].astype(np.float64)

    def _element_extent(self, e_id: EntityID) -> Tuple[Vector, Vector]:
        """
        The pair of positions spanning an element for the purpose of integrating its mass:
        from its main input to its centre of mass (rigid body) or its output port (beam).
        """
        info = self.topology.get_element_info(e_id)
        _, input_pos = self.topology.main_input_of(e_id)
        output_pos = info.obj.com_pos if isinstance(info.obj, RigidBody) else info.output_port.pos
        return input_pos, output_pos

    def modal_product(self, mode_k: Mode, mode_p: Mode) -> float:
        """
        The augmented inner product ``<M V^k, V^p>`` over the whole system (Rui 3.48).

        For ``mode_k is mode_p`` this is the modal mass ``M_p``. For distinct modes at
        *different* eigenfrequencies orthogonality makes it vanish; for distinct modes
        sharing a repeated eigenfrequency it generally does not, which is why an
        orthogonalisation step is needed before the modal equations decouple (see
        :func:`dynaramp.mstmm.response.augmented_modes`).

        Both modes must have been computed at the same frequency for an off-diagonal
        result to mean anything.

        :param mode_k: First mode.
        :param mode_p: Second mode.
        :return: The scalar inner product.
        """
        total = 0.0
        for e_id in self.topology.elements:
            elem = self.topology.get_element(e_id)
            pred_id, _ = self.topology.main_input_of(e_id)
            input_pos, output_pos = self._element_extent(e_id)
            total += float(elem.modal_product(
                input_pos, output_pos, mode_k.frequency,
                mode_k.internal_states[pred_id], mode_p.internal_states[pred_id],
            ))
        return total

    def calc_system_modal_masses(self, modes: List[Mode]) -> Vector:
        """
        The modal mass of the whole system for each mode -- the diagonal of
        :meth:`modal_product`.
        """
        return np.array([self.modal_product(m, m) for m in modes], dtype=np.float64)

    def eigenvectors_at(self, omega: float, max_multiplicity: int = 6) -> List[Mode]:
        """
        Every eigenvector of the system at an already-known eigenfrequency.

        :meth:`natural_modes` calls this once it has refined each frequency in its sweep;
        it is exposed separately for a frequency obtained some other way -- measured, or
        read off a chart -- without repeating the search.

        The number of eigenvectors is the multiplicity of the frequency, read off the
        singular-value spectrum by :func:`null_space_dimension`. They are orthonormal in
        R^n but not yet orthogonal in the augmented inner product; see
        :func:`dynaramp.mstmm.response.augmented_modes` for that step.

        :param omega: A known eigenfrequency [rad/s].
        :param max_multiplicity: Cap on the detected multiplicity.
        :return: One :class:`Mode` per null-space direction.
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

    def get_system_modal_matrices(
            self,
            modes: List[Mode],
            rayleigh: Tuple[float, float] | None = None) -> Tuple[Matrix, Matrix] | Tuple[Matrix, Matrix, Matrix]:
        m_mat = np.diag(self.calc_system_modal_masses(modes)).astype(np.float64)
        k_mat = np.diag([mode.frequency**2 * m_mat[i, i] for i, mode in enumerate(modes)]).astype(np.float64)
        if rayleigh is not None:
            alpha, beta = rayleigh
            c_mat = alpha * m_mat + beta * k_mat
            return m_mat, k_mat, c_mat
        return m_mat, k_mat
