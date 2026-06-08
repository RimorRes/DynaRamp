from __future__ import annotations
import logging

from typing import Tuple, List, Dict, Iterable

import networkx as nx
import numpy as np

from .data_struct import Link, Element, ComplexElement, ElemLike

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

    def mark_tips(self, tip_elems: ElemLike | Iterable[ElemLike]) -> MBS:
        if not isinstance(tip_elems, Iterable):
            tip_elems = (tip_elems,)

        for tip_elem in tip_elems:
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

    def auto_resolve_root(self) -> MBS:
        # TODO: fix this
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

    def auto_resolve_tips(self) -> MBS:
        # TODO: fix this
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

    def link_elements(self, source_elem: ElemLike, target_elem: ElemLike,
                      output_slot: int | None = None, input_slot: int | None = None) -> MBS:

        src_id = self._resolve_eid(source_elem)
        tgt_id = self._resolve_eid(target_elem)
        log_prefix = f"[{src_id}] -> [{tgt_id}]:"

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

        # For type-dispatch, retrieve the stored source and target elements
        # We now know the keys exist
        src_elem_obj = self.elements[src_id]
        tgt_elem_obj = self.elements[tgt_id]
        # Specific dispatching for multi-element types

        match tgt_elem_obj:

            case ComplexElement():
                # Assigning the main input. The primary input occupies the special None slot.
                if input_slot is None:
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
                    if input_slot not in tgt_elem_obj.U_exts:
                        logger.error(f"{log_prefix} slot [{input_slot}] has not been defined for element [{tgt_id}].")
                        return self
                    # Check if slot is populated already
                    if input_slot in occupied_slots:
                        logger.error(f"{log_prefix} slot [{input_slot}] of element [{tgt_id}] is already populated.")
                        return self
                    # If all checks are passed, link the source to the target at the designated slot
                    logger.debug(f"{log_prefix} linking to auxiliary input slot [{input_slot}].")
                    self._links.append(Link(src_id, tgt_id, input_slot))

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
            graph.add_node(eid, etype=self.elements[eid].etype)

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

        logger.info(f"Built system graph.")

        return graph

    def _validate_graph(self, graph: nx.DiGraph) -> nx.DiGraph:

        # Check that the graph is connected
        if not nx.is_connected(graph.to_undirected()):
            err_msg = "Invalid topology. There are disconnected components. Check your elements and links."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Count root-like elements
        root_like = [n for n in graph if graph.out_degree(n) == 0]
        match root_like:
            case []:
                logger.warning("No explicitly root-like element found. Consider checking your links. "
                               "This may be the result of a looping system, "
                               "in which case consider cutting the appropriate hinges.")
            case [root]:
                if root == self._root_eid:
                    logger.info(f"Found single root-like element [{root}]. Matches designated root element.")
                else:
                    err_msg = f"Found single root-like element [{root}] that does not match designated root element."
                    logger.error(err_msg)
                    raise ValueError(err_msg)
            case _:
                err_msg = f"Invalid topology. Multiple root-like elements found: {root_like}. " \
                          f"Consider checking your links and topology."
                logger.error(err_msg)
                raise ValueError(err_msg)

        logger.info(f"Validated system graph.")

        return graph

    def _find_cuts(self, graph: nx.DiGraph) -> List[Tuple[int, int]]:
        # Identify and cut connections between elements to get a tree system
        # At this step a SINGLE desired root is assumed to be selected
        cut_pairs = []

        # Handle the case of DIVERGING NODES (out_degree > 1)
        #   Cutting a connection in this case generates two new INPUT tips/boundaries.
        #   C sign matrix needed
        #   Branching nodes are body nodes with out_degree > 1
        diverging_nodes = [n for n in graph if graph.out_degree(n) > 1]
        for bnode in diverging_nodes:
            neighbors = list(graph.successors(bnode))
            # Each element can only have one output
            # We can choose to only keep the output with the shortest path to root

            preserved = nx.shortest_path(graph, source=bnode, target=self._root_eid)[1]

            for n in neighbors:
                if n != preserved:
                    cut_pairs.append((bnode, n))

        # Handle the potential closed-loop containing the ROOT
        #   Cutting a connection in this case generates a new INPUT and OUTPUT.
        #   No C sign matrix needed. Both state vectors are equal
        if graph.out_degree(self._root_eid) > 0:
            neighbors = list(graph.successors(self._root_eid))
            for n in neighbors:
                cut_pairs.append((self._root_eid, n))

        # Return edges to be cut
        return cut_pairs

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

    def make_tree(self, cut_connections: List[Tuple[int, int]]) -> MBS:
        # Check that expected tips and root have been defined
        # TODO: offer auto root and tip resolution
        if self._root_eid is None:
            err_msg = "Root element not identified."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if not self._tip_eids:
            err_msg = "At least one tip element must be identified."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.graph = self._build_graph()
