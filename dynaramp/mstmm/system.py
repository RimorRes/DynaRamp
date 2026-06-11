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

    def add_root(self, root_boundary: Boundary, target_elem: ElemLike, output_slot: int) -> MBS:
        if self._root is not None:
            err_msg = f"Root boundary is already defined as [{self._root.b_id}]."
            logger.error(err_msg)
            raise ValueError(err_msg)

        tgt_id = self._resolve_elem_id(target_elem)
        if tgt_id not in self.elements:
            err_msg = f"Cannot add root: target element [{tgt_id}] is missing from the system. Add it."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._root = root_boundary
        self._boundaries[root_boundary.b_id] = root_boundary
        self.graph.add_node(root_boundary.b_id)
        self.graph.add_edge(tgt_id, root_boundary.b_id, src_slot=output_slot)

        logger.debug(f"Added [{root_boundary.b_id}] as the root boundary to element [{tgt_id}].")

        return self

    def add_tip(self, tip_boundary: Boundary, target_element: ElemLike, input_slot: int) -> MBS:
        tgt_id = self._resolve_elem_id(target_element)
        if tgt_id not in self.elements:
            err_msg = f"Cannot add tip: target element [{tgt_id}] is missing from the system. Add it."
            logger.error(err_msg)
            raise ValueError(err_msg)

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

    def cut_connection(self, connection: Iterable[ElemLike]) -> nx.DiGraph:

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

        if self.graph.out_degree(src_id) == 1:
            logger.warning(f"{log_prefix} source element [{src_id}] has only one output. Cutting this connection "
                           f"will isolate the source element and its upstream branches.")

            input_bound = Boundary()
            cut = CutPoint()

        else:
            pass

        # Cutting the specified connections and return the resulting graph
        new_graph = self._execute_cuts(self.graph, cut_pairs)
        logger.info(f"Cut {len(cut_pairs)} connections as specified by the user.")
        return new_graph

    def cut_closed_loop(self, connection: Tuple[ElemLike, ElemLike]) -> nx.DiGraph:
        """
        Cutting a closed loop at the designated connection. Creating 2 new boundaries of which on is the new root.
        :param connection:
        :return:
        """
        src_id, dst_id = map(self._resolve_elem_id, connection)
        log_prefix = f"Cutting connection [{src_id}] -> [{dst_id}]:"

        if (src_id, dst_id) not in self.graph.edges:
            logger.error(f"{log_prefix} specified connection does not exist in the system.")
            return self.graph

        # Create new root boundary
        new_root_id = f"{src_id}_to_{dst_id}_cut"
        new_root = Boundary(b_id=new_root_id, free_dofs=list(range(6)), fixed_dofs=[])
        self._root = new_root
        self._boundaries[new_root_id] = new_root
        self.graph.add_node(new_root_id)

        # Cut the connection and link the new root to the source and destination elements
        cut_pairs = [(src_id, dst_id), (dst_id, new_root_id), (new_root_id, src_id)]
        new_graph = self._execute_cuts(self.graph, cut_pairs)

        logger.info(f"Cut closed loop at connection [{src_id}] -> [{dst_id}]. "
                    f"Created new root boundary [{new_root_id}].")

        return new_graph


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

    @staticmethod
    def _execute_cuts(
            graph: nx.DiGraph,
            cut_pairs: Iterable[Tuple[Hashable, Hashable]]
    ) -> nx.DiGraph:
        new_graph = graph.copy()

        for (src, dst) in cut_pairs:
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

