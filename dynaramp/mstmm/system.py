from __future__ import annotations
import logging

from typing import Tuple, List, Dict, Iterable, Hashable

import networkx as nx
import numpy as np

from .data_struct import Element, ElemLike

logger = logging.getLogger(__name__)


class MBS:

    def __init__(self):
        self.elements: Dict[Hashable, Element] = {}
        self._populated_slots: Dict[Hashable, List[int | None]] = {}

        self._root_eid: Hashable | None = None
        self._tip_eids: List[Hashable] = []

        self.graph: nx.DiGraph = nx.DiGraph()

    @staticmethod
    def _resolve_eid(elem_or_eid: ElemLike | Hashable) -> Hashable:
        # Accept raw hashable IDs directly, but prefer an object's explicit `eid` attribute.
        if hasattr(elem_or_eid, "eid"):
            eid = elem_or_eid.eid
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

            self.elements[elem.eid] = elem
            self.graph.add_node(elem.eid)
            logger.debug(f"Adding element {elem.eid} to the system.")

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
            logger.warning(f"{log_prefix} multiple directed edges between the same elements are not allowed.")
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
            preserved = nx.shortest_path(graph, source=dnode, target=self._root_eid)[1]
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
        return cut_pairs

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

    def make_tree(self, cut_connections: List[Tuple[Hashable, Hashable]]) -> MBS:
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
