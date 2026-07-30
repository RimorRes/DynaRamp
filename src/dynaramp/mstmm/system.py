from __future__ import annotations
import logging

from typing import Tuple, List, Dict, Sequence
from collections import deque, defaultdict

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

from common.types import EntityID, Vector, VectorLike, Matrix
from .topology import TopologyHandler

logger = logging.getLogger(__name__)

# TODO: better error raising, custom exceptions?
# TODO: Clean up z_rem, z_red, rem_boundaries clutter for state propagation
# TODO: Add rich results classes


class System:

    def __init__(self, system_topo: TopologyHandler):
        self.topology = system_topo

    def transfer_mat_along_path(self, path: Sequence[EntityID], omega: float) -> Matrix:
        # Get transfer matrix from the output state vector of the path's origin to the output vector of the tail.
        u_chain = np.identity(13)

        for i in range(len(path) - 1):
            e1, e2 = path[i], path[i + 1]
            # Because of the way we use the path, e2 can never be a boundary (root here)
            e2_info = self.topology.get_element_info(e2)
            e2_input_port_idx: int = self.topology.tree[e1][e2]['input_port']
            # Retrieve the port positions
            e2_in_pos: VectorLike = e2_info.input_ports[e2_input_port_idx].pos
            e2_out_pos: VectorLike = e2_info.output_port.pos
            # Here we apply the transfer matrix for the element e2 based on its input and output ports
            if e2_input_port_idx == e2_info.main_input_idx:
                u_chain = e2_info.obj.u(e2_in_pos, e2_out_pos, omega) @ u_chain
            else:
                u_chain = e2_info.obj.u_extract(e2_in_pos, e2_out_pos) @ u_chain

        return u_chain

    def _get_geometric_constraints(
            self,
            boundaries: List[EntityID],
            omega: float
    ) -> List[Matrix]:
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
                input_port_pos: VectorLike = m_info.input_ports[input_port_idx].pos
                main_port_pos: VectorLike = m_info.input_ports[m_info.main_input_idx].pos

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
                        g_cols_dict[b_id].append(np.zeros((6, 13)))

        # Convert dictionary to ordered list of column blocks
        has_g_eqs = any(len(blocks) > 0 for blocks in g_cols_dict.values())
        if has_g_eqs:
            g_cols = [np.vstack(g_cols_dict[b_id], dtype=np.float64) for b_id in boundaries]
        else:
            g_cols = []

        return g_cols

    def _reduce_cut_points(
            self,
            boundaries: List[EntityID],
            z_all: Vector,
            t_mats: List[Matrix],
            g_cols: List[Matrix] | None = None
    ) -> Tuple[List[EntityID], Vector, List[Matrix], List[Matrix]]:
        reduced_boundaries = boundaries.copy()
        reduced_z_all = z_all.copy()
        reduced_t_mats = t_mats.copy()
        reduced_g_cols = g_cols.copy() if g_cols is not None else []

        for cut in self.topology.cut_points:
            # Find the index of the cut's virtual boundaries.
            try:
                idx1 = next(i for i, b_id in enumerate(reduced_boundaries) if b_id == cut.b_id1)
                idx2 = next(i for i, b_id in enumerate(reduced_boundaries) if b_id == cut.b_id2)
            except StopIteration:
                err_msg = f"Cut point boundaries [{cut.b_id1}] or [{cut.b_id2}] not found in tips."
                logger.error(err_msg)
                raise ValueError(err_msg)

            reduced_t_mats[idx1] += reduced_t_mats[idx2] @ cut.mat
            reduced_t_mats.pop(idx2)

            if reduced_g_cols:
                reduced_g_cols[idx1] += reduced_g_cols[idx2] @ cut.mat
                reduced_g_cols.pop(idx2)

            reduced_boundaries.pop(idx2)
            mask = np.ones_like(reduced_z_all, dtype=np.bool)
            mask[(13 * idx2):(13 * (idx2 + 1))] = False
            reduced_z_all = reduced_z_all[mask]

        return reduced_boundaries, reduced_z_all, reduced_t_mats, reduced_g_cols

    def overall_transfer(self, omega) -> Tuple[Matrix, Vector, Vector, List[EntityID]]:
        """

        :param omega:
        :return:
        """
        remaining_boundaries = self.topology.boundaries  # Tracks what boundary vectors are actually still in z_red
        # --- TRANSFER MATRIX COMPUTATION ---
        t_mats = [- np.identity(13)]  # init with root
        for t_id in self.topology.tips:
            path = self.topology.resolve_branch_up_to(src=t_id, tgt=self.topology.root.b_id)
            t_mats.append(self.transfer_mat_along_path(path, omega))

        # --- GEOMETRIC MATRIX COMPUTATION ---
        g_cols = self._get_geometric_constraints(remaining_boundaries, omega)

        # --- OVERALL MATRIX ASSEMBLY ---
        # Condense columns using the cut-point relations
        remaining_boundaries, z_rem, t_mats, g_cols = self._reduce_cut_points(
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
        known_mask = np.array([x is not None for x in z_rem])
        nonzero_mask = np.array([x != 0 for x in z_rem]) & known_mask
        # Eliminate columns corresponding to known boundary conditions...
        u_red = u_all[:, ~ known_mask]
        # ... and move columns corresponding to known non-zero boundary conditions into a load vector
        u_nz = u_all[:, nonzero_mask]
        z_nz = z_rem[nonzero_mask].astype(np.float64)
        f = - u_nz @ z_nz
        # Remove the trivial 13th row
        triv_mask = np.ones_like(f, dtype=bool)
        triv_mask[12] = False
        f = f[triv_mask]
        u_red = u_red[triv_mask, :]

        if u_red.shape[0] != u_red.shape[1]:
            err_msg = "Invalid boundary conditions: The system is either under-constrained or over-constrained"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return u_red, f, z_rem, remaining_boundaries

    def reconstruct_boundary_states(self, z_red: Vector, z_rem: Vector,
                                    rem_boundary_ids: List[EntityID]) -> Dict[EntityID, Vector]:
        # Fill in the reduced overall state vector with the computed values
        z_red_full = np.empty(z_rem.shape, dtype=np.float64)
        z_red_iter = iter(z_red)
        for i, x in enumerate(z_rem):
            if x is None:
                z_red_full[i] = next(z_red_iter)
            else:
                z_red_full[i] = x
        # Decompose z_red_full into it's component state vectors
        z_red_full = z_red_full.reshape((len(rem_boundary_ids), 13))
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
            z_rem: Vector,
            rem_boundary_ids: List[EntityID],
            rtol: float = 1e-4
    ) -> Dict[EntityID, Vector]:
        """
        Propagate state from the tips of the system, through the elements
        (bodies and hinges) and up to the root.
        Verify that the propagated state vector satisfies the boundary conditions at the root.
        :param omega:
        :param z_red:
        :param z_rem:
        :param rem_boundary_ids:
        :param rtol:
        :return:
        """

        logger.debug(f"Computing internal states at {omega:.3e} rad/s")

        boundary_svs = self.reconstruct_boundary_states(z_red, z_rem, rem_boundary_ids)
        root_sv = boundary_svs.pop(self.topology.root.b_id)
        tip_svs = boundary_svs

        # Initialize search heads, traversal tracking, and state vectors for each element
        search_heads = deque([t for t in self.topology.tips])
        searched = {k: False for k in self.topology.elements}
        state_vecs = defaultdict(
            lambda: np.zeros(13),
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
            next_input_pos = next_elem_info.input_ports[next_input_port_idx].pos
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

        rerr = np.linalg.norm(state_vecs[last_elem_id] - root_sv)/np.linalg.norm(root_sv)
        if rerr < rtol:
            logger.debug(f"Propagated state matches root boundary state. Relative error = {rerr:.3e}")
        else:
            err_msg = f"Propagated state doesn't match root boundary state. Relative error = {rerr:.3e}"
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
        u = self.overall_transfer(omega)[0]
        return np.linalg.svd(u, compute_uv=False)[-1]

    def natural_modes(
            self,
            n_modes: int, omega_min=0,
            omega_max: int = 1000,
            search_res: int = 10000,
            rtol: float = 1e-5
    ) -> List[Tuple[float, Vector]]:

        omega = np.linspace(omega_min, omega_max, search_res)
        # Only retain strictly positive frequencies
        # We want to ignore rigid modes (and avoid dividing by zero) and negative frequencies
        omega = omega[omega > 0]
        sigma = np.array([self._sigma_min(w) for w in omega])
        # Find rough peaks corresponding to the smallest singular values
        prominence = 5e-2 * np.max(sigma)
        candidates, _ = find_peaks(-sigma, prominence=prominence)

        modes = []
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
            # Apply SVD to the transfer matrix at the refined frequency
            u, _, z_rem, rem_bounds = self.overall_transfer(res.x)
            _, s, vh = np.linalg.svd(u)
            # Reciprocal condition number
            rcond = s[-1] / s[0]

            logger.info(f"Mode candidate at {res.x:.3e} rad/s.")
            logger.debug(f"sigma_min = {s[-1]:.6e}, sigma_max = {s[0]:.6e}, rcond = {rcond:.6e}")
            # If rcond passes the tolerance, save the frequency and mode states
            if rcond < rtol:
                all_state_vecs = self.propagate_state(omega=res.x, z_red=vh[-1], z_rem=z_rem,
                                                      rem_boundary_ids=rem_bounds)  # or Vh[-1].T for the mode shape
                modes.append((res.x, all_state_vecs))

        modes.sort(key=lambda x: x[0])

        logger.info(f"Found {len(modes)} natural modes between {omega_min:.3e} rad/s and {omega_max:.3e} rad/s.")
        for w, _ in modes:
            logger.debug(f"Mode at {w:.3e} rad/s.")

        return modes[:n_modes]

    def solve(self):
        # TODO: move some of the logic here
        raise NotImplementedError
