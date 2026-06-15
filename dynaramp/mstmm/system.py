from __future__ import annotations
import logging

from typing import Tuple, List, Dict, Iterable, Hashable

import networkx as nx
import numpy as np

from .data_struct import Element, ElemLike, Boundary, CutPoint

logger = logging.getLogger(__name__)


class MBS:

    def __init__(self):
        self.elements: Dict[Hashable, Element] = {}
        self._populated_slots: Dict[Hashable, List[int | None]] = {}

        self._root: Boundary | None = None
        self._boundaries: Dict[Hashable, Boundary] = {}
        self._cut_points: List[CutPoint] = []

        self.graph: nx.DiGraph = nx.DiGraph()

    @staticmethod
    def _resolve_elem_id(elem_or_eid: ElemLike) -> Hashable:
        # Accept raw hashable IDs directly but prefer an object's explicit `e_id` attribute.
        if hasattr(elem_or_eid, "e_id"):
            e_id = elem_or_eid.e_id
            if isinstance(e_id, Hashable):
                return e_id
            err_msg = "Provided element has an unhashable `e_id`."
            logger.error(err_msg)
            raise TypeError(err_msg)

        if isinstance(elem_or_eid, Hashable):
            return elem_or_eid

        err_msg = (
            f"Expected a hashable element ID or Element-like object with an `e_id` attribute. "
            f"Got {type(elem_or_eid).__name__}."
        )
        logger.error(err_msg)
        raise TypeError(err_msg)

    def add_elements(self, elems: Element | Iterable[Element]) -> MBS:
        if not isinstance(elems, Iterable):
            elems = (elems,)

        for elem in elems:
            # Strictly typed interface. Check that the element is an instance of Element or its subclasses
            if not isinstance(elem, Element):
                err_msg = (f"Provided element must be an instance of Element or its subclasses. "
                           f"Got {type(elem).__name__}.")
                logger.error(err_msg)
                raise TypeError(err_msg)

            self.elements[elem.e_id] = elem
            self.graph.add_node(elem.e_id)
            logger.debug(f"Adding element {elem.e_id} to the system.")

        return self

    def add_root(self, boundary_sv: np.ndarray, target_elem: ElemLike, output_slot: int) -> Hashable:
        # NO SLOT OVERWRITE PROTECTION
        if self._root is not None:
            err_msg = f"Root boundary is already defined as [{self._root.b_id}]."
            logger.error(err_msg)
            raise ValueError(err_msg)

        tgt_id = self._resolve_elem_id(target_elem)
        if tgt_id not in self.elements:
            err_msg = f"Cannot add root: target element [{tgt_id}] is missing from the system. Add it."
            logger.error(err_msg)
            raise ValueError(err_msg)

        root_boundary = Boundary(b_id=f"{tgt_id}.{output_slot},0", state_vector=boundary_sv)
        self._root = root_boundary
        self._boundaries[root_boundary.b_id] = root_boundary
        self.graph.add_node(root_boundary.b_id)
        self.graph.add_edge(tgt_id, root_boundary.b_id, src_slot=output_slot, dst_slot=None)

        logger.debug(f"Added [{root_boundary.b_id}] as the root boundary to element [{tgt_id}].")

        return root_boundary.b_id

    def add_tip(self, boundary_sv: np.ndarray, target_element: ElemLike, input_slot: int) -> Hashable:
        # NO SLOT OVERWRITE PROTECTION
        tgt_id = self._resolve_elem_id(target_element)
        if tgt_id not in self.elements:
            err_msg = f"Cannot add tip: target element [{tgt_id}] is missing from the system. Add it."
            logger.error(err_msg)
            raise ValueError(err_msg)

        tip_boundary = Boundary(b_id=f"{tgt_id}.{input_slot},0", state_vector=boundary_sv)
        self._boundaries[tip_boundary.b_id] = tip_boundary
        self.graph.add_node(tip_boundary.b_id)
        self.graph.add_edge(tip_boundary.b_id, tgt_id, src_slot=None, dst_slot=input_slot)

        logger.debug(f"Added [{tip_boundary.b_id}] as a tip boundary to element [{tgt_id}].")

        return tip_boundary.b_id

    def connect_elements(
            self,
            src: ElemLike,
            dst: ElemLike,
            src_slot: int,
            dst_slot: int | None = None,
    ) -> MBS:

        src_id = self._resolve_elem_id(src)
        dst_id = self._resolve_elem_id(dst)
        log_prefix = f"[{src_id}] -> [{dst_id}]:"

        # Prevent creating an invalid link referencing non-existent elements
        if src_id not in self.elements:
            err_msg = f"{log_prefix} source element [{src_id}] is missing from the system. Add it."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if dst_id not in self.elements:
            err_msg = f"{log_prefix} target element [{dst_id}] is missing from the system. Add it."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Prevent self-loops
        if src_id == dst_id:
            logger.warning(f"{log_prefix} would create a self-loop.")
            return self

        # Check if the directed edge already exists (regardless of slot)
        if (src_id, dst_id) in self.graph.edges:
            logger.warning(f"{log_prefix} parallel edges between the same elements are not allowed.")
            return self

        # For type-dispatch, retrieve the stored source and target elements
        # We now know the keys exist
        src_elem = self.elements[src_id]
        dst_elem = self.elements[dst_id]
        # Specific dispatching based on selected slot

        # SOURCE ELEMENT
        if src_slot in self._populated_slots[src_id]:
            err_msg = f"{log_prefix} slot [{src_slot}] of element [{src_id}] is already populated."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if not src_slot in src_elem.slots:  # Also handles the invalid case of using the None slot as an output
            err_msg = (f"{log_prefix} slot [{src_slot}] is not defined for the source element "
                       f"or cannot be used as an output.")
            logger.error(err_msg)
            raise ValueError(err_msg)

        # DESTINATION ELEMENT
        if dst_slot in self._populated_slots[dst_id]:
            err_msg = (f"{log_prefix} slot [{"MAIN" if dst_slot is None else dst_slot}] "
                         f"of element [{dst_id}] is already populated.")
            logger.error(err_msg)
            raise ValueError(err_msg)
        if dst_slot is not None and dst_slot not in dst_elem.slots:
            err_msg = f"{log_prefix} slot [{dst_slot}] is not defined for the destination element."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # After both elements are validated, they can be linked
        logger.debug(f"{log_prefix} linking from source slot [{src_slot}] "
                     f"to destination slot [{"MAIN" if dst_slot is None else dst_slot}].")
        self._populated_slots[src_id].append(src_slot)
        self._populated_slots[dst_id].append(dst_slot)
        self.graph.add_edge(src_id, dst_id, src_slot=src_slot, dst_slot=dst_slot)

        return self

    def cut_connection(self, connection: Iterable[ElemLike]) -> MBS:

        src_id, dst_id = map(self._resolve_elem_id, connection)
        log_prefix = f"Cutting [{src_id}] -/> [{dst_id}]:"

        if (src_id, dst_id) not in self.graph.edges:
            err_msg = f"{log_prefix} specified connection does not exist in the system."
            logger.error(err_msg)
            raise ValueError(err_msg)

        if src_id in self._boundaries or dst_id in self._boundaries:
            err_msg = f"{log_prefix} cannot cut a connection with a boundary."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Grab the edge data now that we know that the edge exists
        src_slot = self.graph[src_id][dst_id]['src_slot']
        dst_slot = self.graph[src_id][dst_id]['dst_slot']

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

    def _find_cuts(self, graph: nx.DiGraph) -> List[Tuple[Hashable, Hashable]]:
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
            succs = list(graph.successors(dnode))
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
        # The actual check needs to be run on a reversed view of the graph since the root is the sink, not the source
        if not nx.is_arborescence(nx.reverse_view(self.graph)):
            err_msg = f"{log_prefix} invalid topology. Could not transform system into a tree."
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.info(f"{log_prefix}: validated topology.")

        return self

    def _transfer_path(self, tree: nx.DiGraph, source_id: Hashable, target_id: Hashable) -> np.ndarray:
        # Get transfer matrix from the output state vector of the source to the input vec
        path = nx.shortest_path(tree, source=source_id, target=target_id)
        segments = {}
        for i in range(len(path)-1):
            n1, n2 = path[i], path[i+1]
            if n2 in self.elements:
                segments[(n1, n2)] = tree[n1][n2]['dst_slot']

        # Walk through the branches and pre-multiply along the way
        transfer_matrix = np.empty((12,12))

        return transfer_matrix

    def _geometric_relation(self):
        pass

