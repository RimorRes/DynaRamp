from __future__ import annotations
import logging

from typing import Tuple, List, Dict

import networkx as nx
import numpy as np

from .data_struct import Link, Element, MultiInputElement, ElemLike


logger = logging.getLogger(__name__)

class MBS:

    def __init__(self):
        self.elements: Dict[int, Element] = {}
        self.links: List[Link] = []
        self.root_eid: int | None = None
        self.tip_eids: List[int] = []

        self.graph: nx.DiGraph | None = None

    @staticmethod
    def _resolve_eid(elem_or_eid: ElemLike) -> int:
        if isinstance(elem_or_eid, int):
            return elem_or_eid
        # duck typing: accept anything with an `eid` attribute
        if hasattr(elem_or_eid, "eid"):
            try:
                return int(elem_or_eid.eid)
            except Exception as exc:
                err_msg = "Provided element has a non-integer `eid`."
                logger.error(err_msg)
                raise ValueError(err_msg) from exc
        err_msg = "Expected int or Element-like object with an `eid` attribute"
        logger.error(err_msg)
        raise TypeError(err_msg)

    def add_element(self, element: Element) -> MBS:
        # Strictly typed interface. Check that the element is an instance of Element or its subclasses
        if not isinstance(element, Element):
            raise TypeError("Provided element has a non-integer `eid`.")

        self.elements[element.eid] = element
        logger.debug(f"Adding element {element} to the system.")

        return self

    def mark_root(self, root_elem: ElemLike) -> MBS:
        root_eid = self._resolve_eid(root_elem)

        if root_eid not in self.elements:
            logger.error(f"Cannot mark root: element {root_eid} is missing from the system. Add it.")
        else:
            self.root_eid = root_eid
            logger.debug(f"Marked element {root_eid} as root.")

        return self

    def mark_tip(self, tip_eid: int) -> MBS:
        pass

    def link_elements(self, source_elem: ElemLike,
                      target_elem: ElemLike, slot_num: int | None = None) -> MBS:
        src_id = self._resolve_eid(source_elem)
        tgt_id = self._resolve_eid(target_elem)
        log_prefix = f"Linking {src_id} -> {tgt_id}:"
        # Sanity check for element existence
        if src_id not in self.elements:
            logger.error(f"{log_prefix} source element {src_id} is missing from the system. Add it.")
        if tgt_id not in self.elements:
            logger.error(f"{log_prefix} target element {tgt_id} is missing from the system. Add it.")

        # Prevent duplicates
        if any(l.source == src_id and l.target == tgt_id for l in self.links):
            logger.warning(f"{log_prefix} duplicate link not created.")
            return self

        # Prevent self-loops
        if src_id == tgt_id:
            logger.error(f"{log_prefix} would create a self-loop.")
            return self

        # For type-dispatch, retrieve the stored target element
        target_elem = self.elements.get(tgt_id)
        if target_elem is None:
            return self
        # Specific dispatching for mulit-element types

        match target_elem:

            case MultiInputElement():
                # ...slot number must be provided...
                if slot_num is None:
                    logger.error(f"{log_prefix} slot number must be provided for multi-input element {tgt_id}.")
                    return self
                # ...and it must be valid
                if slot_num not in target_elem.Us:
                    logger.error(f"{log_prefix} slot number {slot_num} has not been defined for element {tgt_id}.")
                    return self
                if any((c.target == tgt_id and c.slot == slot_num) for c in self.links):
                    logger.warning(f"{log_prefix} slot number {slot_num} of element {tgt_id} is already populated.")
                    return self

                self.links.append(Link(src_id, tgt_id, slot_num))

            case Element():
                self.links.append(Link(src_id, tgt_id))

            case _:
                # Fallback
                logger.error(f"{log_prefix} element {tgt_id} is of unsupported type {type(target_elem).__name__}.")

        return self

    def _build_graph(self) -> nx.DiGraph:
        # Bodies and Hinges are BOTH nodes in this graph instead of Hinges being edges
        graph = nx.DiGraph()
        logger.debug(f"Building system graph with elements {self.elements}.")

        for eid in self.elements:
            graph.add_node(eid)

        for link in self.links:
            # Validate that all connections reference existing elements.
            # This is necessary because networkx would silently add the missing node automatically,
            # and we would then be missing the necessary element definitions
            if not (link.source in self.elements):
                err_msg = f"Link source element {link.source} does not exist in the system elements."
                logger.error(err_msg)
                raise ValueError(err_msg)
            if not (link.target in self.elements):
                err_msg = f"Link target element {link.target} does not exist in the system elements."
                logger.error(err_msg)
                raise ValueError(err_msg)

            match link:
                case Link(source=src, target=tgt, slot=None):
                    logger.debug(f"Adding single-input link {src} -> {tgt} to the graph.")
                    graph.add_edge(src, tgt)
                case Link(source=src, target=tgt, slot=slot):
                    logger.debug(f"Adding multi-input link {src} -> {tgt} (slot {slot}) to the graph.")
                    # Carry over the input slot number data for multi-input elements
                    graph.add_edge(src, tgt, slot=slot)

        # Check that the graph is connected
        if not nx.is_connected(graph.to_undirected()):
            err_msg = "System graph invalid: there are disconnected components. Check your elements and links."
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.debug(f"Successfully built system graph.")

        return graph

    def _auto_cut_hinges(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, List[Tuple[int, int]]]:
        new_tree = graph.copy()
        cut_pairs = [] # Cut pairs generate new tip boundaries that need to be returned
        # Identify and cut hinges to get a tree system
        # Branching nodes are body nodes with out_degree > 1
        # TODO: as is a hinge could also be multi-output and it would get cut. Potential issue?
        branching_nodes = [n for n in graph if graph.out_degree(n) > 1]
        for bnode in branching_nodes:
            neighbors = list(graph.successors(bnode))
            # Each element can only have one output
            # We can choose to only keep the output with the shortest path to root
            preserved = nx.shortest_path(graph, source=bnode, target=self.root_eid)[1]

            for n in neighbors:
                if n != preserved:
                    new_tree.remove_edge(bnode, n)
                    cut_pairs.append((bnode, n))

        return new_tree, cut_pairs

    def _transfer_path(self, tree: nx.DiGraph, source_eid: int, target_eid: int) -> np.ndarray:
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

    def make_tree(self):
        self.graph = self._build_graph()
