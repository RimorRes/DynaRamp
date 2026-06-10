from __future__ import annotations
import logging

from typing import Tuple, List, Dict, Iterable, Hashable

import networkx as nx
import numpy as np
from networkx.algorithms import boundary

from .data_struct import Element, ElemLike, Boundary

logger = logging.getLogger(__name__)


class MBS:

    def __init__(self):
        self.elements: Dict[Hashable, Element] = {}
        self._populated_slots: Dict[Hashable, List[int | None]] = {}

        self._root: Boundary | None = None
        self._boundaries: Dict[Hashable, Boundary] = {}

        self.graph: nx.DiGraph = nx.DiGraph()

    @staticmethod
    def _resolve_eid(elem_or_eid: ElemLike | Hashable) -> Hashable:
        # Accept raw hashable IDs directly but prefer an object's explicit `eid` attribute.
        if hasattr(elem_or_eid, "eid"):
            eid = elem_or_eid.e_id
            if isinstance(eid, Hashable):
                return eid
            err_msg = "Provided element has an unhashable `eid`."
            logger.error(err_msg)
            raise TypeError(err_msg)

        if isinstance(elem_or_eid, Hashable):
            return elem_or_eid

        err_msg = (
            f"Expected a hashable element ID or Element-like object with an `eid` attribute. "
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

    def add_root(self, root_boundary: Boundary, target_elem: ElemLike, output_slot: int) -> MBS:
        if self._root is not None:
            logger.warning(f"Root boundary is already defined as [{self._root.b_id}]. No action taken.")
            return self

        tgt_id = self._resolve_eid(target_elem)
        if tgt_id not in self.elements:
            logger.error(f"Cannot add root: target element [{tgt_id}] is missing from the system. Add it.")
            return self

        self._root = root_boundary
        self._boundaries[root_boundary.b_id] = root_boundary
        self.graph.add_node(root_boundary.b_id)
        self.graph.add_edge(tgt_id, root_boundary.b_id, src_slot=output_slot)

        logger.debug(f"Added [{root_boundary.b_id}] as the root boundary to element [{tgt_id}].")

        return self

    def add_tip(self, tip_boundary: Boundary, target_element: ElemLike, input_slot: int) -> MBS:
        tgt_id = self._resolve_eid(target_element)
        if tgt_id not in self.elements:
            logger.error(f"Cannot add tip: target element [{tgt_id}] is missing from the system. Add it.")
            return self

        if tip_boundary in self._boundaries:
            logger.warning(f"Boundary [{tip_boundary.b_id}] is already defined. No action taken.")
            return self

        self._boundaries[tip_boundary.b_id] = tip_boundary
        self.graph.add_node(tip_boundary.b_id)
        self.graph.add_edge(tip_boundary.b_id, tgt_id, dst_slot=input_slot)

        logger.debug(f"Added [{tip_boundary.b_id}] as a tip boundary to element [{tgt_id}].")

        return self

    def connect_elements(
            self,
            src: ElemLike,
            dst: ElemLike,
            src_slot: int,
            dst_slot: int | None = None,
    ) -> MBS:

        src_id = self._resolve_eid(src)
        dst_id = self._resolve_eid(dst)
        log_prefix = f"[{src_id}] -> [{dst_id}]:"

        # Prevent creating an invalid link referencing non-existent elements
        if src_id not in self.elements:
            logger.error(f"{log_prefix} source element [{src_id}] is missing from the system. Add it.")
            return self
        if dst_id not in self.elements:
            logger.error(f"{log_prefix} target element [{dst_id}] is missing from the system. Add it.")
            return self

        # Prevent self-loops
        if src_id == dst_id:
            logger.error(f"{log_prefix} would create a self-loop.")
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
            logger.error(f"{log_prefix} slot [{src_slot}] of element [{src_id}] is already populated.")
            return self
        if not src_slot in src_elem.slots:  # Also handles the invalid case of using the None slot as an output
            logger.error(f"{log_prefix} slot [{src_slot}] is not defined for the source element "
                         f"or cannot be used as an output.")
            return self

        # DESTINATION ELEMENT
        if dst_slot in self._populated_slots[dst_id]:
            logger.error(f"{log_prefix} slot [{"MAIN" if dst_slot is None else dst_slot}] "
                         f"of element [{dst_id}] is already populated.")
            return self
        if dst_slot is not None and dst_slot not in dst_elem.slots:
            logger.error(f"{log_prefix} slot [{dst_slot}] is not defined for the destination element.")
            return self

        # After both elements are validated, they can be linked
        logger.debug(f"{log_prefix} linking from source slot [{src_slot}] "
                     f"to destination slot [{"MAIN" if dst_slot is None else dst_slot}].")
        self._populated_slots[src_id].append(src_slot)
        self._populated_slots[dst_id].append(dst_slot)
        self.graph.add_edge(src_id, dst_id, src_slot=src_slot, dst_slot=dst_slot)

        return self

    def _validate_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        # TODO: fix this
        # Check that the graph is connected
        if not nx.is_connected(graph.to_undirected()):
            err_msg = "Invalid topology. There are disconnected components. Check your elements and links."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Check that all elements can flow to the root
        pass
        # Should probably detect if the root is looped
        pass

        logger.info(f"Validated system graph.")

        return graph

    def _find_cuts(self, graph: nx.DiGraph) -> List[Tuple[Hashable, Hashable]]:
        # Identify and cut connections between elements to get a tree system
        # At this step a SINGLE desired root is assumed to be selected
        cut_pairs = []

        # Handle the case of DIVERGING NODES (out_degree > 1)
        #   Cutting a connection in this case generates two new INPUT tips/boundaries.
        #   C sign matrix will be needed
        #   Branching nodes are body nodes with out_degree > 1
        diverging_nodes = [n for n in graph if graph.out_degree(n) > 1]
        for dnode in diverging_nodes:
            neighbors = list(graph.successors(dnode))
            # Each element can only have one output
            # We can choose to only keep the output with the shortest path to root
            preserved = nx.shortest_path(graph, source=dnode, target=self._root.b_id)[1]
            for n in neighbors:
                if n != preserved:
                    cut_pairs.append((dnode, n))

        # Handle the potential closed-loop containing the ROOT
        #   Cutting a connection in this case generates a new INPUT and OUTPUT.
        #   No C sign matrix will be needed. Both state vectors are equal
        if graph.out_degree(self._root_eid) > 0:
            neighbors = list(graph.successors(self._root_eid))
            for n in neighbors:
                cut_pairs.append((self._root_eid, n))

        # Return edges to be cut
        logger.info(f"Found {len(cut_pairs)} cuts to be made.")
        for i, (src, dst) in enumerate(cut_pairs):
            logger.debug(f"Cut {i}: {src}->{dst}")

        return cut_pairs

    def _execute_cuts(
            self,
            graph: nx.DiGraph,
            cut_pairs: Iterable[Tuple[Hashable, Hashable]]
    ) -> nx.DiGraph:
        new_graph = graph.copy()

        for (src, dst) in cut_pairs:
            if new_graph.out_degree(src) == 1:
                new_graph.remove_edge(src, dst)




        return new_graph


    def make_tree(self, cut_connections: List[Tuple[Hashable, Hashable]]) -> MBS:
        # Check that expected tips and root have been defined
        # TODO: offer auto root and tip resolution
        if self._root_eid is None:
            err_msg = "Root element not identified."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # First, execute user-defined cuts
        # Then cut the remaining connections to get a tree system

    def _transfer_path(self, tree: nx.DiGraph, source_eid: Hashable, target_eid: Hashable) -> np.ndarray:
        # Get the path from the tip to the root in the tree
        path = nx.shortest_path(tree, source=source_eid, target=target_eid)

        # Walk through the branches and pre-multiply along the way
        transfer_matrix = self.elements[source_eid].U
        prev_node = source_eid
        for node in path[1:]:
            pass

        return transfer_matrix

    def _geometric_relation(self):
        pass

