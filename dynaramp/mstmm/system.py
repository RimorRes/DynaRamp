from __future__ import annotations
import logging

from typing import Tuple, List, Dict

import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.unweighted import predecessor

from .data_struct import Link, Element, MultiInputElement, ElemLike


logger = logging.getLogger(__name__)

class MBS:

    def __init__(self):
        self.elements: Dict[int, Element] = {}

        self._links: List[Link] = []
        self._root_eid: int | None = None
        self._tip_eids: List[int] = []

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
        err_msg = f"Expected int or Element-like object with an `eid` attribute. Got {type(elem_or_eid).__name__}."
        logger.error(err_msg)
        raise TypeError(err_msg)

    def add_element(self, elem: Element) -> MBS:
        # Strictly typed interface. Check that the element is an instance of Element or its subclasses
        if not isinstance(elem, Element):
            err_msg = f"Provided element must be an instance of Element or its subclasses. Got {type(elem).__name__}."
            logger.error(err_msg)
            raise TypeError(err_msg)

        self.elements[elem.eid] = elem
        logger.debug(f"Adding element {elem} to the system.")

        return self

    def mark_root(self, root_elem: ElemLike) -> MBS:
        root_eid = self._resolve_eid(root_elem)

        if root_eid not in self.elements:
            logger.error(f"Cannot mark root: element {root_eid} is missing from the system. Add it.")
        else:
            if self._root_eid is not None:
                logger.warning(f"Overwriting previously marked root element [{self._root_eid}] "
                               f"with new root [{root_eid}].")

            self._root_eid = root_eid
            logger.debug(f"Marked element {root_eid} as root.")

        return self


    def mark_tip(self, tip_elem: ElemLike) -> MBS:
        tip_eid = self._resolve_eid(tip_elem)

        if tip_eid not in self.elements:
            logger.error(f"Cannot mark root: element {tip_eid} is missing from the system. Add it.")
        else:
            if tip_eid in self._tip_eids:
                logger.info(f"Element [{tip_eid}] is already marked as a tip. No action taken.")
            else:
                self._tip_eids.append(tip_eid)
                logger.debug(f"Marked element {tip_eid} as tip.")

        return self


    def auto_mark_root(self) -> MBS:
        # Attempts to find the root element automatically
        # Loop over links and remove elements who have outputs
        potential_roots = set(self.elements.keys())
        for link in self._links:
            if link.target in potential_roots:
                potential_roots.remove(link.target)

        if not potential_roots:
            logger.warning("Cannot find any root in the system. Consider marking one manually.")
        else:
            try:
                root, = potential_roots  # Only works for a singleton
                logger.info(f"Auto-identified root element: [{root}]. Marking it as root.")
                self._root_eid = root
            except ValueError as exc:
                logger.error(f"Auto-identified multiple root elements: {potential_roots}. Invalid system structure.")

        return self


    def auto_mark_tips(self) -> MBS:
        # Attempts to find the tip elements automatically
        # Loop over links and remove elements who have inputs
        potential_tips = set(self.elements.keys())
        for link in self._links:
            if link.target in potential_tips:
                potential_tips.remove(link.target)

        if not potential_tips:
            logger.warning("Cannot find any tips in the system. Consider marking them manually.")
        else:
            logger.info(f"Auto-identified tip elements: {potential_tips}. Marking them as tips.")
            self._tip_eids.extend(potential_tips)

        return self



    def link_elements(self, source_elem: ElemLike,
                      target_elem: ElemLike, slot: int | None = None) -> MBS:

        src_id = self._resolve_eid(source_elem)
        tgt_id = self._resolve_eid(target_elem)
        log_prefix = f"Linking [{src_id}] -> [{tgt_id}]:"

        # Prevent creating an invalid link referencing non-existent elements
        if src_id not in self.elements:
            logger.error(f"{log_prefix} source element [{src_id}] is missing from the system. Add it.")
            return self
        if tgt_id not in self.elements:
            logger.error(f"{log_prefix} target element [{tgt_id}] is missing from the system. Add it.")
            return self

        # Prevent self-loops
        if src_id == tgt_id:
            logger.error(f"{log_prefix} would create a self-loop.")
            return self

        # Index previously created links which have target_elem as the target
        # Sets for membership tests
        sources = set()
        occupied_slots = set()
        for link in self._links:
            if link.target == tgt_id:
                sources.add(link.source)
                occupied_slots.add(link.slot)

        # Elements (body or hinges) can only have a single link between them (regardless of slot)
        if src_id in sources:
            logger.warning(f"{log_prefix} multiple links between the same elements are not allowed.")
            return self

        # For type-dispatch, retrieve the stored target element
        # We know the key exists
        tgt_elem_obj = self.elements[tgt_id]
        # Specific dispatching for multi-element types

        match tgt_elem_obj:

            case MultiInputElement():
                # Assigning the main input. The primary input occupies the special None slot.
                if slot is None:
                    # Check if main slot is populated already
                    if None in occupied_slots:
                        logger.error(f"{log_prefix} main slot of element [{tgt_id}] is already populated. Assign to "
                                       f"a free slot to link to an auxiliary input.")
                        return self
                    # If not, proceed to populate it
                    logger.debug(f"{log_prefix} populating main input.")
                    self._links.append(Link(src_id, tgt_id))  # Behave like a single-input element for the primary input
                # Assigning an auxillary input
                else:
                    # Check if slot is valid
                    if slot not in tgt_elem_obj.U_exts:
                        logger.error(f"{log_prefix} slot [{slot}] has not been defined for element [{tgt_id}].")
                        return self
                    # Check if slot is populated already
                    if slot in occupied_slots:
                        logger.error(f"{log_prefix} slot [{slot}] of element [{tgt_id}] is already populated.")
                        return self
                    # If all checks are passed, link the source to the target at the designated slot
                    logger.debug(f"{log_prefix} linking to auxiliary input slot [{slot}].")
                    self._links.append(Link(src_id, tgt_id, slot))

            case Element():
                # Check if previously linked
                if sources:
                    logger.error(f"{log_prefix} element [{tgt_id}] is single-input "
                                   f"but already has a link from source {sources}.")
                    return self

                logger.debug(f"{log_prefix} linking single-input element.")
                self._links.append(Link(src_id, tgt_id))

            case _:
                # Fallback
                logger.error(f"{log_prefix} element {tgt_id} is of unsupported type {type(tgt_elem_obj).__name__}.")

        return self


    def _build_graph(self) -> nx.DiGraph:
        # Bodies and Hinges are BOTH nodes in this graph instead of Hinges being edges
        graph = nx.DiGraph()
        logger.info(f"Building system graph with elements {self.elements}.")

        for eid in self.elements:
            graph.add_node(eid)

        for link in self._links:
            # Validate that all connections reference existing elements.
            # This is necessary because networkx would silently add the missing node automatically,
            # and we would then be missing the necessary element definitions
            if not (link.source in self.elements):
                err_msg = f"Link source element [{link.source}] does not exist in the system elements."
                logger.error(err_msg)
                raise ValueError(err_msg)
            if not (link.target in self.elements):
                err_msg = f"Link target element [{link.target}] does not exist in the system elements."
                logger.error(err_msg)
                raise ValueError(err_msg)

            match link:
                case Link(source=src, target=tgt, slot=None):
                    logger.debug(f"Adding single-input edge [{src}] -> [{tgt}] to the graph.")
                    graph.add_edge(src, tgt)
                case Link(source=src, target=tgt, slot=slot):
                    logger.debug(f"Adding multi-input edge [{src}] -> [{tgt}] (slot [{slot}]) to the graph.")
                    # Carry over the input slot number data for multi-input elements
                    graph.add_edge(src, tgt, slot=slot)

        # Check that the graph is connected
        if not nx.is_connected(graph.to_undirected()):
            err_msg = "System graph invalid: there are disconnected components. Check your elements and links."
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.info(f"Successfully built system graph.")

        return graph


    def _auto_cut_hinges(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, List[Tuple[int, int]]]:
        if self._root_eid is None:
            err_msg = "Root element not identified. Mark it manually or run auto_mark_root."
            logger.error(err_msg)
            raise ValueError(err_msg)

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
            preserved = nx.shortest_path(graph, source=bnode, target=self._root_eid)[1]

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
