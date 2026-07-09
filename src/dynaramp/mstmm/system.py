from __future__ import annotations
import logging

from typing import Tuple, List, Dict, Iterable, Sequence
from collections import deque, defaultdict

import networkx as nx
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

from ..common_types import EntityID, is_entity_id, Vector, VectorLike, Matrix
from .structs import NULL_SV, Element, ElemLike, Boundary, CutPoint

logger = logging.getLogger(__name__)

# TODO: better error raising, custom exceptions?


class MBS:

    def __init__(self):
        self._elements: Dict[EntityID, Element] = {}
        self._slot_occupancy: Dict[EntityID, Dict[EntityID | None, str | None]] = {}  # 'input', 'output' or None

        self._root: Boundary | None = None
        self._tips: Dict[EntityID, Boundary] = {}
        self._cut_points: List[CutPoint] = []

        self._z_all = np.array([], dtype=np.float64)  # Overall state vector (concatenated boundary vectors)

        self._internal_graph: nx.DiGraph = nx.DiGraph()
        self._user_graph: nx.DiGraph = nx.DiGraph()  # Graph entirely defined by user, for visualization and analysis

        # For internal use only!
        self._z_all_cache_invalid = True  # Flag to indicate if the overall state vector needs to be recomputed
        self._tree_generated = False
        self._successor_in_tree = {}
        self._upstream_tips = {}

    # Read-only attributes
    @property
    def elements(self) -> Dict[EntityID, Element]:
        # Return a copy to avoid accidental mutation of internal state
        return self._elements.copy()

    @property
    def graph(self) -> nx.DiGraph[EntityID]:
        if self._tree_generated:
            return self._user_graph.copy(as_view=True)
        else:
            return self._internal_graph.copy(as_view=True)

    @property
    def tree(self):
        if self._tree_generated:
            return self._internal_graph.copy(as_view=True)
        else:
            err_msg = "Tree not generated."
            logger.error(err_msg)
            raise ValueError(err_msg)

    @property
    def root(self) -> Boundary:
        if self._root is None:
            err_msg = "Root element not identified."
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            return self._root

    @property
    def tips(self) -> Dict[EntityID, Boundary]:
        # Return a copy to avoid accidental mutation of internal state
        return self._tips.copy()

    @property
    def boundaries(self) -> List[EntityID]:
        # Sort the boundaries -> [root, tip1, tip2, ...]
        # Purposefully not cached. Every call generates a new list to avoid accidental mutations/side effects.
        # Avoid calling `self.root` here because the property logs and raises; check internal `_root` directly.
        if self._root is None:
            logger.warning("Root boundary not defined. Returning only tip boundaries.")
            bound = [t_id for t_id in self._tips]
        else:
            bound = [self._root.b_id] + [t_id for t_id in self._tips]
        return bound

    @property
    def z_all(self) -> Vector:
        # Should only be called once all boundaries are defined (especially root as this would otherwise throw an error)
        if self._z_all_cache_invalid:  # Recompute cached value as needed
            z = np.hstack([self.root.state_vector] + [b.state_vector for b in self._tips.values()])
            z.setflags(write=False)  # Make the aray read-only
            self._z_all = z
            self._z_all_cache_invalid = False
        return self._z_all

    @staticmethod
    def _resolve_elem_id(elem_or_eid: ElemLike) -> EntityID:
        # Accept raw IDs directly but prefer an object's explicit `e_id` attribute.
        if hasattr(elem_or_eid, "e_id"):
            e_id_attr = elem_or_eid.e_id
            if is_entity_id(e_id_attr):
                return e_id_attr
            err_msg = f"Unsupported type {type(e_id_attr).__name__} for the provided element's `e_id`."
            logger.error(err_msg)
            raise TypeError(err_msg)

        if is_entity_id(elem_or_eid):
            return elem_or_eid

        err_msg = (
            f"Expected an entity ID or Element-like object with an `e_id` attribute. "
            f"Got {type(elem_or_eid).__name__}."
        )
        logger.error(err_msg)
        raise TypeError(err_msg)

    def add_elements(self, elems: Element | Iterable[Element]) -> MBS:
        if isinstance(elems, Iterable):
            elem_iterable = elems
        else:
            elem_iterable = (elems,)

        for elem in elem_iterable:
            # Strictly typed interface. Check that the element is an instance of Element or its subclasses
            if not isinstance(elem, Element):
                err_msg = (f"Provided element must be an instance of Element or its subclasses. "
                           f"Got {type(elem).__name__}.")
                logger.error(err_msg)
                raise TypeError(err_msg)

            self._elements[elem.e_id] = elem
            self._internal_graph.add_node(elem.e_id)
            # Initializing slot status for the new element
            occupancy_init: Dict[EntityID | None, str | None] = {None: None}  # init `None` a.k.a `MAIN` slot
            for s in elem.slots_pos:
                occupancy_init[s] = None
            self._slot_occupancy[elem.e_id] = occupancy_init
            logger.debug("Adding element %r to the system.", elem.e_id)

        return self

    def add_root(self, target_elem: ElemLike, output_slot: EntityID, boundary_sv: VectorLike) -> EntityID:
        # NO SLOT OVERWRITE PROTECTION
        if self._root is not None:
            err_msg = f"Root boundary is already defined as [{self._root.b_id}]."
            logger.error(err_msg)
            raise ValueError(err_msg)

        tgt_id = self._resolve_elem_id(target_elem)
        if tgt_id not in self._elements:
            err_msg = f"Cannot add root: target element [{tgt_id}] is missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        root_boundary = Boundary(b_id=f"{tgt_id}.{output_slot}, 0", state_vector=boundary_sv)
        # Cache the new root
        self._root = root_boundary
        self._slot_occupancy[tgt_id][output_slot] = 'output'
        self._z_all_cache_invalid = True
        # Add it to the graph
        self._internal_graph.add_node(root_boundary.b_id)
        self._internal_graph.add_edge(tgt_id, root_boundary.b_id, output_slot=output_slot, input_slot=None)

        logger.debug(f"Added [{root_boundary.b_id}] as the root boundary to element [{tgt_id}].")

        return root_boundary.b_id

    def add_tip(self, target_element: ElemLike, input_slot: EntityID | None, boundary_sv: VectorLike) -> EntityID:
        # NO SLOT OVERWRITE PROTECTION
        tgt_id = self._resolve_elem_id(target_element)
        if tgt_id not in self._elements:
            err_msg = f"Cannot add tip: target element [{tgt_id}] is missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        tip_boundary = Boundary(b_id=f"{tgt_id}.{input_slot}, 0", state_vector=boundary_sv)
        # Cache the new tip boundary
        self._tips[tip_boundary.b_id] = tip_boundary
        self._slot_occupancy[tgt_id][input_slot] = 'input'
        self._z_all_cache_invalid = True
        # Add to graph
        self._internal_graph.add_node(tip_boundary.b_id)
        self._internal_graph.add_edge(tip_boundary.b_id, tgt_id, output_slot=None, input_slot=input_slot)

        logger.debug(f"Added [{tip_boundary.b_id}] as a tip boundary to element [{tgt_id}].")

        return tip_boundary.b_id

    def connect_elements(
            self,
            src: ElemLike,
            dst: ElemLike,
            src_slot: EntityID,
            dst_slot: EntityID | None = None
    ) -> MBS:

        src_id = self._resolve_elem_id(src)
        dst_id = self._resolve_elem_id(dst)
        log_prefix = f"[{src_id}] -> [{dst_id}]:"

        # Prevent creating an invalid link referencing non-existent elements
        if src_id not in self._elements or dst_id not in self._elements:
            err_msg = f"{log_prefix} element(s) missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        # Prevent self-loops
        if src_id == dst_id:
            logger.warning(f"{log_prefix} would create a self-loop.")
            return self

        # Check if the directed edge already exists (regardless of slot)
        if (src_id, dst_id) in self._internal_graph.edges:
            logger.warning(f"{log_prefix} parallel edges between the same elements are not allowed.")
            return self

        # Specific dispatching based on selected slot
        # SOURCE ELEMENT
        if src_slot is None:
            err_msg = f"{log_prefix} slot [MAIN] cannot be used as an output."
            logger.error(err_msg)
            raise ValueError(err_msg)
        try:
            if self._slot_occupancy[src_id][src_slot] is not None:
                err_msg = f"{log_prefix} slot [{src_slot}] of element [{src_id}] is already populated."
                logger.error(err_msg)
                raise ValueError(err_msg)
        except KeyError:  # Undefined slots end up here
            err_msg = (f"{log_prefix} slot [{src_slot}] is not defined for the source element "
                       f"or cannot be used as an output.")
            logger.error(err_msg)
            raise KeyError(err_msg)

        # DESTINATION ELEMENT
        try:
            if self._slot_occupancy[dst_id][dst_slot] is not None:
                err_msg = (f"{log_prefix} slot [{"MAIN" if dst_slot is None else dst_slot}] "
                           f"of element [{dst_id}] is already populated.")
                logger.error(err_msg)
                raise ValueError(err_msg)
        except KeyError:
            err_msg = f"{log_prefix} slot [{dst_slot}] is not defined for the destination element."
            logger.error(err_msg)
            raise KeyError(err_msg)

        # After both elements are validated, they can be linked
        logger.debug(f"{log_prefix} linking from source slot [{src_slot}] "
                     f"to destination slot [{"MAIN" if dst_slot is None else dst_slot}].")
        self._slot_occupancy[src_id][src_slot] = 'output'
        self._slot_occupancy[dst_id][dst_slot] = 'input'
        self._internal_graph.add_edge(src_id, dst_id, output_slot=src_slot, input_slot=dst_slot)

        return self

    def cut_connection(self, connection: Iterable[ElemLike]) -> MBS:
        """
        "Cutting the hinge" i.e., generating virtual tips to account for multi-output or looping elements
        :param connection:
        :return:
        """

        src_id, dst_id = map(self._resolve_elem_id, connection)
        log_prefix = f"Cutting [{src_id}] -/> [{dst_id}]:"

        try:
            src_slot = self._internal_graph[src_id][dst_id]['output_slot']
            dst_slot = self._internal_graph[src_id][dst_id]['input_slot']
        except KeyError:
            err_msg = f"{log_prefix} specified connection does not exist in the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        if src_id in self.boundaries or dst_id in self.boundaries:
            err_msg = f"{log_prefix} cannot cut a connection with a tip boundary."
            logger.error(err_msg)
            raise ValueError(err_msg)

        if self._internal_graph.out_degree(src_id) == 1:
            # Check if node is in a directed cycle
            in_cycle = False
            for scc in nx.strongly_connected_components(self._internal_graph):
                if src_id in scc and len(scc) > 1:
                    in_cycle = True
            if in_cycle:
                try:
                    # It would be mathematically equivalent to have root in b_id2,
                    # but it makes more sense to conserve the root during reduction
                    b_id1 = self.add_root(self._elements[src_id], src_slot, NULL_SV)
                    b_id2 = self.add_tip(self._elements[dst_id], dst_slot, NULL_SV)
                    # Cutting a connection in this case generates a new virtual INPUT/OUTPUT pair.
                    # No C sign matrix will be needed. Both state vectors are equal
                    cut = CutPoint(b_id1, b_id2, False)
                    logger.info(f"{log_prefix} cut closed loop. Created new root [{b_id1}] and tip [{b_id2}].")
                except ValueError as exc:
                    err_msg = f"{log_prefix} failed to cut the connection. Could not create new root."
                    logger.error(err_msg)
                    raise ValueError(err_msg) from exc
            else:
                err_msg = (f"{log_prefix} source element [{src_id}] has only one output. Cutting this connection "
                           f"would isolate the downstream elements.")
                logger.error(err_msg)
                raise ValueError(err_msg)

        else:
            b_id1 = self.add_tip(self._elements[src_id], src_slot, NULL_SV)
            b_id2 = self.add_tip(self._elements[dst_id], dst_slot, NULL_SV)
            # Cutting a connection in this case generates two new virtual INPUT tips/boundaries.
            # C sign matrix will be needed
            cut = CutPoint(b_id1, b_id2)
            logger.debug(f"{log_prefix} cut connection. Created new tips [{b_id1}] and [{b_id2}].")

        # clean up the old edge and save the cutting point relation
        self._cut_points.append(cut)
        self._internal_graph.remove_edge(src_id, dst_id)
        logger.debug(f"{log_prefix}: old edge removed.")
        logger.info(f"{log_prefix} done.")

        return self

    def find_cuts(self) -> List[Tuple[EntityID, EntityID]]:
        # Identify which connections between elements need to be removed to get a tree system
        # At this step a valid and unique root must be selected

        connections_to_cut = []
        # Handle diverging nodes (out_degree > 1)
        diverging_nodes = [n for n in self._internal_graph if self._internal_graph.out_degree(n) > 1]
        for dnode in diverging_nodes:
            succs = iter(self._internal_graph[dnode])
            # Each element can only have one output
            # We can choose to only keep the output with the shortest path to root
            preserved = nx.shortest_path(self._internal_graph, source=dnode, target=self.root.b_id)[1]
            for n in succs:
                if n != preserved:
                    connections_to_cut.append((dnode, n))
        # Handle the special closed loop case
        # Collect strongly connected components, in our case this is analogous to directed cycles
        cycles = nx.strongly_connected_components(self._internal_graph)
        root_cyc = None
        for cyc in cycles:
            if len(cyc) > 1:
                if all([self._internal_graph.out_degree(node) == 1 for node in cyc]):
                    root_cyc = cyc
                    break
        if root_cyc:
            # We now need to choose where to cut
            # We can try cutting at anentry point of the loop
            for node in root_cyc:
                if self._internal_graph.in_degree(node) > 1:
                    pred_in_cycle = next(x for x in self._internal_graph.predecessors(node) if x in root_cyc)
                    connections_to_cut.append((pred_in_cycle, node))
                    break
            # If this fails, we can just cut anywhere
            node = root_cyc.pop()
            root_cyc.add(node)
            pred_in_cycle = next(x for x in self._internal_graph.predecessors(node) if x in root_cyc)
            logger.debug((pred_in_cycle, node))
            connections_to_cut.append((pred_in_cycle, node))

        # Return edges to be cut
        logger.info(f"Found {len(connections_to_cut)} cuts to be made.")
        for i, (src, dst) in enumerate(connections_to_cut):
            logger.debug(f"Cut {i}: {src}->{dst}")

        return connections_to_cut

    def make_tree(self) -> MBS:
        logger.info("Transforming system into a tree structure.")
        self._user_graph = self._internal_graph.copy()

        logger.debug(f"Auto resolving edges to cut.")
        c2c = self.find_cuts()  # Connections to cut
        for connection in c2c:
            self.cut_connection(connection)

        # Check that we have a correct tree system
        # The actual check needs to be run on a reversed view of the graph since the root is the sink, not the source.
        # is_arborescence allows for an in_degree <= 1,
        # but with the current algorithm any element node with in_degree == 0 would be disconnected
        # (only tips can verify this condition and be connected as they are not internal nodes);
        # ergo this effectively validates the tree structure where every element has exactly one ouput.
        if not nx.is_arborescence(nx.reverse_view(self._internal_graph)):
            err_msg = f"Invalid topology. Could not transform system into a tree."
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.debug(f"Validated topology.")

        # Buffer the output slot of all element/boundary nodes to speed up the transfer matrix calculations
        for e in self._elements:
            successor = next(self._internal_graph.successors(e))
            out_slot = next(
                slot
                for slot, value in self._slot_occupancy[e].items()
                if value == "output"
            )
            self._successor_in_tree[e] = {'output_slot': out_slot, 'next': successor}
        # Same operation on the tips (we exclude the root as it has no output)
        for t_id in self._tips:
            successor = next(self._internal_graph.successors(t_id))
            self._successor_in_tree[t_id] = {'output_slot': None, 'next': successor}

        # Next, we buffer what elements have what upstream tips. We don't want to keep testing unreachable elements.
        for e in self._elements:
            self._upstream_tips[e] = []
            for t_id in self._tips:
                try:
                    self.resolve_branch_up_to(src=t_id, tgt=e)
                    self._upstream_tips[e].append(t_id)
                except ValueError:
                    continue

        logger.info("Successfully built valid tree system.")
        self._tree_generated = True

        return self

    def resolve_branch_up_to(self, src: EntityID, tgt: EntityID) -> List[EntityID]:
        """
        Return path from src to tgt (excluding tgt)
        :param src:
        :param tgt:
        :return:
        """
        path = []
        node = src
        while node != tgt:
            path.append(node)
            try:
                node = self._successor_in_tree[node]['next']
            except KeyError:
                err_msg = f"No path found from [{src}] to [{tgt}]."
                logger.warning(err_msg)
                raise ValueError(err_msg)

        return path

    def transfer_mat_along_path(self, path: Sequence[EntityID], omega: float) -> Matrix:
        # Get transfer matrix from the output state vector of the path's origin to the output vector of the tail.
        u_chain = np.identity(13)

        for i in range(len(path) - 1):
            e1, e2 = path[i], path[i + 1]

            e2_elem = self._elements[e2]
            input_slot = self.tree[e1][e2]['input_slot']
            # Retrieve the position of the output
            output_slot = self._successor_in_tree[e2]['output_slot']
            out_pos = e2_elem.slots_pos[output_slot]
            # Here we apply the transfer matrix for the element e2 based on its input slot (input_slot)
            if input_slot is None:
                u_chain = e2_elem.u(out_pos, omega) @ u_chain
            else:
                u_chain = e2_elem.u_ext(out_pos, input_slot) @ u_chain

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
        multi_input_elems = [e_id for e_id in self._elements if self.tree.in_degree(e_id) > 1]
        # Initialize dictionary columns for every boundary (guarantees Root gets zeros automatically)
        g_cols_dict = {b_id: [] for b_id in boundaries}

        for m_id in multi_input_elems:
            # Retrieve the N incoming branches
            preds = list(self.tree.predecessors(m_id))
            if len(preds) <= 1:
                continue

            # Group tips by the immediate incoming branch they pass through
            tips_by_branch = {p: [] for p in preds}
            k_mats = {}

            for t_id in self._upstream_tips[m_id]:
                path = self.resolve_branch_up_to(src=t_id, tgt=m_id)
                branch_node = path[-1]
                if branch_node in tips_by_branch:
                    tips_by_branch[branch_node].append(t_id)

                # Extract the pure kinematics transformation matrix (no negative signs!)
                u_chain = self.transfer_mat_along_path(path, omega)
                input_slot = self.tree[branch_node][m_id]['input_slot']
                if input_slot is None:
                    k_mat = self._elements[m_id].h_ext @ u_chain
                else:
                    k_mat = self._elements[m_id].h_incs[input_slot] @ u_chain
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

        for cut in self._cut_points:
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
        remaining_boundaries = self.boundaries  # Tracks what boundary vectors are actually still in z_red
        # --- TRANSFER MATRIX COMPUTATION ---
        t_mats = [- np.identity(13)]  # init with root
        for t_id in self._tips:
            path = self.resolve_branch_up_to(src=t_id, tgt=self.root.b_id)
            t_mats.append(self.transfer_mat_along_path(path, omega))

        # --- GEOMETRIC MATRIX COMPUTATION ---
        g_cols = self._get_geometric_constraints(remaining_boundaries, omega)

        # --- OVERALL MATRIX ASSEMBLY ---
        # Condense columns using the cut-point relations
        remaining_boundaries, z_rem, t_mats, g_cols = self._reduce_cut_points(
            remaining_boundaries, self.z_all, t_mats, g_cols
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
        f = u_nz @ z_nz
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
        for c in self._cut_points:
            eliminated_boundaries[c.b_id2] = c.mat @ rem_boundaries[c.b_id1]
        # We need to build the final dict in the correct order
        bound_svs = {}
        for b_id in self.boundaries:
            if b_id in rem_boundary_ids:
                bound_svs[b_id] = rem_boundaries[b_id]
            else:
                bound_svs[b_id] = eliminated_boundaries[b_id]

        return bound_svs

    def _sigma_min(self, omega: float) -> float:
        """
        Helper for retrieving the smallest singular value of an SVD of U_all(w).
        :param omega:
        :return:
        """
        u = self.overall_transfer(omega)[0]
        return np.linalg.svd(u, compute_uv=False)[-1]

    def solve(self):
        pass

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

            logger.debug(f"Mode candidate at {res.x:.3e} rad/s: "
                         f"sigma_min = {s[-1]:.6e}, sigma_max = {s[0]:.6e}, rcond = {rcond:.6e}")
            # If rcond passes the tolerance, save the frequency and mode shape
            if rcond < rtol:
                all_state_vecs = self.propagate_state(omega=res.x, z_red=vh[-1], z_rem=z_rem,
                                                      rem_boundary_ids=rem_bounds)  # or Vh[-1].T for the mode shape
                modes.append((res.x, all_state_vecs))

        modes.sort(key=lambda x: x[0])

        logger.info(f"Found {len(modes)} natural modes between {omega_min:.3e} rad/s and {omega_max:.3e} rad/s.")
        for w, _ in modes:
            logger.debug(f"Mode at {w:.3e} rad/s.")

        return modes[:n_modes]

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
        log_prefix = f"Computing internal states at {omega:.3e} rad/s:"

        boundary_svs = self.reconstruct_boundary_states(z_red, z_rem, rem_boundary_ids)
        root_sv = boundary_svs.pop(self.root.b_id)
        tip_svs = boundary_svs

        # Initialize search heads, traversal tracking, and state vectors for each element
        search_heads = deque([t for t in self._tips])
        searched = {k: False for k in self._elements}
        state_vecs = defaultdict(
            lambda: np.zeros(13),
            tip_svs
        )

        while search_heads:
            head_id = search_heads.popleft()
            next_id = self._successor_in_tree[head_id]['next']

            if next_id not in self._elements:
                # Also synonymous with the next node being the root
                continue
            # Info about the next element
            next_elem = self._elements[next_id]
            input_slot = self.tree[head_id][next_id]['input_slot']
            output_slot = self._successor_in_tree[next_id]['output_slot']
            output_pos = next_elem.slots_pos[output_slot]
            # Add to the output state vector of the next element
            if input_slot is None:
                sv = next_elem.u(output_pos, omega) @ state_vecs[head_id]
            else:
                sv = next_elem.u_ext(output_pos, input_slot) @ state_vecs[head_id]
            state_vecs[next_id] += sv
            # If the next element has already been used as a head, we don't need to add it again
            if not searched[next_id]:
                search_heads.append(next_id)
                searched[next_id] = True

        # Verify that all elements have been sweeped
        searched_count = list(searched.values()).count(True)
        msg = f"{log_prefix} swept through ({searched_count}/{len(searched)}) elements."
        if searched_count != len(searched):
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.debug(msg)

        # Verify that the propagated state vector satisfies the boundary conditions at the root
        last_elem_id = next(self.tree.predecessors(self.root.b_id))

        rerr = np.linalg.norm(state_vecs[last_elem_id] - root_sv)/np.linalg.norm(root_sv)
        if rerr < rtol:
            logger.debug(f"{log_prefix} propagated state matches root boundary state. Relative error = {rerr}")
        else:
            err_msg = f"{log_prefix} propagated state doesn't match root boundary state. Relative error = {rerr}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Add the root to the end of the dict
        state_vecs[self.root.b_id] = root_sv

        logger.info(f"{log_prefix} success.")

        return state_vecs
