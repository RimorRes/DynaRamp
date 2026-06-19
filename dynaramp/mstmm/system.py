from __future__ import annotations
import logging

from typing import Tuple, List, Dict, Iterable, Sequence

import networkx as nx
import numpy as np

from ..common_types import EntityID, is_entity_id, Vector, Matrix
from .structs import Element, ElemLike, Boundary, CutPoint

logger = logging.getLogger(__name__)


class MBS:

    def __init__(self):
        self.elements: Dict[EntityID, Element] = {}
        self.slot_occupancy: Dict[EntityID, Dict[EntityID | None, str | None]] = {}  # 'input', 'output' or None

        self._root: Boundary | None = None
        self._boundaries: Dict[EntityID, Boundary] = {}
        self._cut_points: List[CutPoint] = []

        self.graph: nx.DiGraph = nx.DiGraph()
        self._successor_in_tree = {}

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

            self.elements[elem.e_id] = elem
            self.graph.add_node(elem.e_id)
            # Initializing slot status for the new element
            occupancy_init: Dict[EntityID | None, str | None] = {None: None}  # init `None` a.k.a `MAIN` slot
            for s in elem.slots_pos:
                occupancy_init[s] = None
            self.slot_occupancy[elem.e_id] = occupancy_init
            logger.debug("Adding element %r to the system.", elem.e_id)

        return self

    def add_root(self, boundary_sv: Vector, target_elem: ElemLike, output_slot: int) -> EntityID:
        # NO SLOT OVERWRITE PROTECTION
        if self._root is not None:
            err_msg = f"Root boundary is already defined as [{self._root.b_id}]."
            logger.error(err_msg)
            raise ValueError(err_msg)

        tgt_id = self._resolve_elem_id(target_elem)
        if tgt_id not in self.elements:
            err_msg = f"Cannot add root: target element [{tgt_id}] is missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        root_boundary = Boundary(b_id=f"{tgt_id}.{output_slot},0", state_vector=boundary_sv)
        self._root = root_boundary
        self._boundaries[root_boundary.b_id] = root_boundary
        self.graph.add_node(root_boundary.b_id)
        self.graph.add_edge(tgt_id, root_boundary.b_id, src_slot=output_slot, dst_slot=None)
        self.slot_occupancy[tgt_id][output_slot] = 'output'

        logger.debug(f"Added [{root_boundary.b_id}] as the root boundary to element [{tgt_id}].")

        return root_boundary.b_id

    def add_tip(self, boundary_sv: Vector, target_element: ElemLike, input_slot: int) -> EntityID:
        # NO SLOT OVERWRITE PROTECTION
        tgt_id = self._resolve_elem_id(target_element)
        if tgt_id not in self.elements:
            err_msg = f"Cannot add tip: target element [{tgt_id}] is missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        tip_boundary = Boundary(b_id=f"{tgt_id}.{input_slot},0", state_vector=boundary_sv)
        self._boundaries[tip_boundary.b_id] = tip_boundary
        self.graph.add_node(tip_boundary.b_id)
        self.graph.add_edge(tip_boundary.b_id, tgt_id, src_slot=None, dst_slot=input_slot)
        self.slot_occupancy[tgt_id][input_slot] = 'input'

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
        if src_id not in self.elements or dst_id not in self.elements:
            err_msg = f"{log_prefix} element(s) missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        # Prevent self-loops
        if src_id == dst_id:
            logger.warning(f"{log_prefix} would create a self-loop.")
            return self

        # Check if the directed edge already exists (regardless of slot)
        if (src_id, dst_id) in self.graph.edges:
            logger.warning(f"{log_prefix} parallel edges between the same elements are not allowed.")
            return self

        # Specific dispatching based on selected slot
        # SOURCE ELEMENT
        if src_slot is None:
            err_msg = f"{log_prefix} slot [MAIN] cannot be used as an output."
            logger.error(err_msg)
            raise ValueError(err_msg)
        try:
            if self.slot_occupancy[src_id][src_slot] is not None:
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
            if self.slot_occupancy[dst_id][dst_slot] is not None:
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
        self.slot_occupancy[src_id][src_slot] = 'output'
        self.slot_occupancy[dst_id][dst_slot] = 'input'
        self.graph.add_edge(src_id, dst_id, src_slot=src_slot, dst_slot=dst_slot)

        return self

    def cut_connection(self, connection: Iterable[ElemLike]) -> MBS:

        src_id, dst_id = map(self._resolve_elem_id, connection)
        log_prefix = f"Cutting [{src_id}] -/> [{dst_id}]:"

        try:
            src_slot = self.graph[src_id][dst_id]['src_slot']
            dst_slot = self.graph[src_id][dst_id]['dst_slot']
        except KeyError:
            err_msg = f"{log_prefix} specified connection does not exist in the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        if src_id in self._boundaries or dst_id in self._boundaries:
            err_msg = f"{log_prefix} cannot cut a connection with a boundary."
            logger.error(err_msg)
            raise ValueError(err_msg)

        if self.graph.out_degree(src_id) == 1:
            # Check if node is in a directed cycle
            in_cycle = False
            for scc in nx.strongly_connected_components(self.graph):
                if src_id in scc and len(scc) > 1:
                    in_cycle = True
            if in_cycle:
                try:
                    b_id1 = self.add_root(np.full(12, None), self.elements[src_id], src_slot)
                    b_id2 = self.add_tip(np.full(12, None), self.elements[dst_id], dst_slot)
                    # Cutting a connection in this case generates a new INPUT and OUTPUT.
                    # No C sign matrix will be needed. Both state vectors are equal
                    cut = CutPoint(b_id1, b_id2, False)
                    logger.debug(f"{log_prefix} cut closed loop. Created new root [{b_id1}] and tip [{b_id2}].")
                except ValueError:
                    err_msg = f"{log_prefix} failed to cut the connection. Could not create new root."
                    logger.error(err_msg)
                    raise ValueError(err_msg)
            else:
                err_msg = (f"{log_prefix} source element [{src_id}] has only one output. Cutting this connection "
                           f"would isolate the downstream elements.")
                logger.error(err_msg)
                raise ValueError(err_msg)

        else:
            b_id1 = self.add_tip(np.full(12, None), self.elements[src_id], src_slot)
            b_id2 = self.add_tip(np.full(12, None), self.elements[dst_id], dst_slot)
            # Cutting a connection in this case generates two new INPUT tips/boundaries.
            # C sign matrix will be needed
            cut = CutPoint(b_id1, b_id2)
            logger.debug(f"{log_prefix} cut connection. Created new tips [{b_id1}] and [{b_id2}].")

        # clean up the old edge and save the cutting point relation
        self._cut_points.append(cut)
        self.graph.remove_edge(src_id, dst_id)
        logger.debug(f"{log_prefix}: old edge removed.")
        logger.info(f"{log_prefix} done.")

        return self

    def _find_cuts(self, graph: nx.DiGraph) -> List[Tuple[EntityID, EntityID]]:
        # Identify and cut connections between elements to get a tree system

        # At this step a valid and unique root must be selected
        if self._root is None:
            err_msg = "Root element not identified."
            logger.error(err_msg)
            raise ValueError(err_msg)

        connections_to_cut = []
        # Handle diverging nodes (out_degree > 1)
        diverging_nodes = [n for n in graph if graph.out_degree(n) > 1]
        for dnode in diverging_nodes:
            succs = iter(graph[dnode])
            # Each element can only have one output
            # We can choose to only keep the output with the shortest path to root
            preserved = nx.shortest_path(graph, source=dnode, target=self._root.b_id)[1]
            for n in succs:
                if n != preserved:
                    connections_to_cut.append((dnode, n))

        # Return edges to be cut
        logger.info(f"Found {len(connections_to_cut)} cuts to be made.")
        for i, (src, dst) in enumerate(connections_to_cut):
            logger.debug(f"Cut {i}: {src}->{dst}")

        return connections_to_cut

    def make_tree(self) -> MBS:
        log_prefix = "Making system tree:"

        logger.debug(f"{log_prefix}: auto resolving edges to cut.")
        c2c = self._find_cuts(self.graph)
        logger.debug(f"{log_prefix}: cutting edges.")
        for connection in c2c:
            self.cut_connection(connection)

        # Check that we have a correct tree system
        # The actual check needs to be run on a reversed view of the graph since the root is the sink, not the source.
        # is_arborescence allows for an in_degree <= 1,
        # but with the current algorithm any element node with in_degree == 0 would be disconnected
        # (only tips can verify this condition and be connected as they are not internal nodes);
        # ergo this effectively validates the tree structure where every element has exactly one ouput.
        if not nx.is_arborescence(nx.reverse_view(self.graph)):
            err_msg = f"{log_prefix} invalid topology. Could not transform system into a tree."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Buffer the output slot of all element nodes to speed up the transfer matrix calculations
        for e in self.elements:
            successor = next(iter(self.graph[e]))
            out_slot = next(
                slot
                for slot, value in self.slot_occupancy[e].items()
                if value == "output"
            )
            self._successor_in_tree[e] = {'output_slot': out_slot, 'next': successor}

        logger.info(f"{log_prefix}: validated topology.")

        return self

    def _resolve_branch_up_to(self, src: EntityID, tgt: EntityID) -> List[EntityID]:
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
                logger.error(err_msg)
                raise ValueError(err_msg)

        return path

    def _transfer_mat_along_path(self, path: Sequence[EntityID], omega: float) -> Matrix:
        # Get transfer matrix from the output state vector of the path's origin to the output vector of the tail.
        u_chain = np.identity(12)

        for i in range(len(path) - 1):
            e1, e2 = path[i], path[i + 1]

            e2_elem = self.elements[e2]
            dst_slot = self.graph[e1][e2]['dst_slot']
            # Here we apply the transfer matrix for the element e2 based on its input slot (dst_slot)
            if dst_slot is None:
                output_slot = self._successor_in_tree[e2]['output_slot']
                out_pos = e2_elem.slots_pos[output_slot]
                u_chain = e2_elem.u(omega, out_pos) @ u_chain
            else:
                u_chain = e2_elem.u_exts[dst_slot] @ u_chain

        return u_chain

    def _geometric_mat(self, tip_id: EntityID, mult_in_e_id: EntityID, omega: float) -> Matrix:
        try:
            path = self._resolve_branch_up_to(src=tip_id, tgt=mult_in_e_id)
            u_chain = self._transfer_mat_along_path(path, omega)
            dst_slot = self.graph[path[-1]][mult_in_e_id]['dst_slot']
            if dst_slot is None:
                g_mat = - self.elements[mult_in_e_id].h_ext @ u_chain
            else:
                g_mat = self.elements[mult_in_e_id].h_incs[dst_slot] @ u_chain
            return g_mat
        except nx.NetworkXNoPath:
            return np.zeros((6, 12))

    def overall_transfer(self, omega) -> Tuple[Matrix, Vector, Vector]:
        if self._root is None:
            err_msg = "Root element not identified."
            logger.error(err_msg)
            raise ValueError(err_msg)
        # Sort the boundaries -> [root, tip1, tip2, ...]
        tips = [b for b in self._boundaries.values() if b is not self._root]

        t_mats = []
        for tip in tips:
            try:
                path = self._resolve_branch_up_to(src=tip.b_id, tgt=self._root.b_id)
                t_mats.append(self._transfer_mat_along_path(path, omega))
            except nx.NetworkXNoPath:
                raise ValueError(f"No path found from tip [{tip.b_id}] to root [{self._root.b_id}].")

        multi_input_elems = [e_id for e_id in self.elements if self.graph.in_degree(e_id) > 1]
        g_cols = []
        for tip in tips:
            col = []
            for mult_in_e_id in multi_input_elems:
                col.append(self._geometric_mat(tip.b_id, mult_in_e_id, omega))
            g_cols.append(np.vstack(col))

        # Condense columns at cut points
        for cut in self._cut_points:
            # Find the index of the cut's tip boundaries in the tip list
            try:
                idx1 = next(i for i, b in enumerate(tips) if b.b_id == cut.b_id1)
                idx2 = next(i for i, b in enumerate(tips) if b.b_id == cut.b_id2)
            except StopIteration:
                raise ValueError(f"Cut point boundaries [{cut.b_id1}] or [{cut.b_id2}] not found in tips.")

            t_mats[idx1] += t_mats[idx2] @ cut.mat
            t_mats.pop(idx2)

            g_cols[idx1] += g_cols[idx2] @ cut.mat
            g_cols.pop(idx2)

            tips.pop(idx2)

        t_block = np.hstack(t_mats)
        g_block = np.hstack(g_cols)

        u_all = np.block([
            [- np.identity(12), t_block],
            [np.zeros((g_block.shape[0], 12)), g_block],
        ])

        z_all = np.hstack([self._root.state_vector] + [b.state_vector for b in tips])

        # Handle known boundary conditions
        known_mask = np.array([x is not None for x in z_all])
        nonzero_mask = np.array([x != 0 for x in z_all]) & known_mask
        # Eliminate columns corresponding to known zero boundary conditions...
        u_red = u_all[:, ~ known_mask]
        # ... and move columns corresponding to known non-zero boundary conditions into a load vector
        u_nz = u_all[:, nonzero_mask]
        z_nz = z_all[nonzero_mask]
        f = u_nz @ z_nz

        return u_red, f, known_mask
