from __future__ import annotations

from typing import Tuple, List, Dict, Optional

import networkx as nx
import numpy as np

from .data_struct import Element, Connection


class MBS:

    def __init__(self):
        self.elements: Dict[int, Element] = {}
        self.connections: List[Connection] = []
        self.root_eid: int = 0
        self.tip_eids: List[int] = []

        self.graph: Optional[nx.DiGraph] = None


    def add_element(self, element: Element) -> MBS:
        self.elements[element.id] = element
        # TODO: distinguish between body and hinge elements
        return self

    def add_connection(self, from_elem: int, to_elem: int, input_slot: int = 1) -> MBS:
        self.connections.append(Connection(from_elem, to_elem, input_slot))
        # TODO: add checks for valid input slots
        return self


    def _build_graph(self) -> nx.Graph:
        # Bodies and Hinges are BOTH nodes in this graph instead of Hinges being edges
        graph = nx.DiGraph()
        for eid in self.elements:
            graph.add_node(eid)
        for c in self.connections:
            # Carry over the input number data
            graph.add_edge(c.source, c.target, input_slot=c.input_slot)
        return graph

    def _cut_hinges(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, List[int]]:
        new_tree = graph.copy()
        cut_pairs = []
        # Identify and cut hinges to get a tree system
        # Offenders are body nodes with out_degree > 1
        # TODO: as is a hinge could also be multi-output and it would get cut. Potential issue?
        offenders = [n for n in list(graph.nodes) if graph.out_degree(n) > 2]
        for off in offenders:
            neighbors = list(graph.successors(off))
            # Each element can only have one output
            # We can choose to only keep the output with the shortest path to root
            preserved = nx.shortest_path(graph, source=off, target=self.root_eid)[1]
            # The offender in the cut pair generates a new tip boundary
            self.tip_eids.append(off)
            for n in neighbors:
                if n != preserved:
                    new_tree.remove_edge(off, n)
                    cut_pairs.append((off, n))
                    # The neighbor in the cut pair generates a new tip boundary
                    self.tip_eids.append(n)

        return new_tree, cut_pairs

    def _transfer_path(self, tree: nx.DiGraph, tip_eid: int, node_eid: int) -> np.ndarray:
        # Get the path from the tip to the root in the tree
        path = nx.shortest_path(tree, source=tip_eid, target=node_eid)

        # Walk through the branches and pre-multiply along the way
        transfer_matrix = self.elements[tip_eid].U
        prev_node = tip_eid
        for node in path[1:]:
            input_slot = tree.get_edge_data(prev_node, node)["input_slot"]

            if input_slot == 1:
                transfer_matrix = self.elements[node].U @ transfer_matrix
            else:
                transfer_matrix = self.elements[node].U_extract @ transfer_matrix

        return transfer_matrix

    def _geometric_relation(self):
        pass