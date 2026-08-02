from __future__ import annotations
from typing import Tuple, List, Dict, Iterable

import logging

from functools import wraps

from dataclasses import dataclass

import networkx as nx
import numpy as np
from pygments.styles import default

from ..common.types import EntityID, is_entity_id, Vector, VectorLike
from .structs import NULL_SV, PortType, Element, ElemLike, Boundary, CutPoint

logger = logging.getLogger(__name__)


def requires_tree_generated(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._tree_generated:
            err_msg = "Invalid state for the requested operation: tree must be generated."
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        return func(self, *args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class ElementInfo:
    obj: Element
    main_input_idx: int
    ports: List[PortInfo]
    output_port: PortInfo
    upstream_tips: List[EntityID]
    downstream: EntityID


@dataclass(frozen=True)
class TipInfo:
    obj: Boundary
    downstream: EntityID


@dataclass(frozen=True)
class PortInfo:
    pos: Vector
    ptype: PortType


class TopologyHandler:

    def __init__(self):
        self._elements: Dict[EntityID, Element] = {}
        self._elem_port_pos: Dict[EntityID, List[Vector]] = {}
        self._elem_port_type: Dict[EntityID, List[PortType]] = {}
        self._elem_main_port: Dict[EntityID, int] = {}

        self._root: Boundary | None = None
        self._tips: Dict[EntityID, Boundary] = {}
        self._cut_points: List[CutPoint] = []

        self._internal_graph: nx.DiGraph = nx.DiGraph()
        self._user_graph: nx.DiGraph = nx.DiGraph()  # Graph entirely defined by user, for visualization and analysis
        # This flag controls access to some attribs that would be nonsensical at this stage (e.g., z_all, etc.)
        self._tree_generated = False
        # Caches
        self._z_all = np.array([])  # Overall state vector (concatenated boundary vectors)
        self._einfo_cache: Dict[EntityID, ElementInfo] = {}
        self._tinfo_cache: Dict[EntityID, TipInfo] = {}

    # Read-only and public-facing attributes
    @property
    def elements(self) -> Dict[EntityID, Element]:
        # Return a copy to avoid accidental mutation of internal state
        return self._elements.copy()

    def get_element(self, eid: EntityID) -> Element:
        return self._elements[eid]

    @requires_tree_generated
    def get_element_info(self, e_id: EntityID) -> ElementInfo:
        return self._einfo_cache[e_id]

    @property
    def graph(self) -> nx.DiGraph[EntityID]:
        if self._tree_generated:
            return self._user_graph.copy(as_view=True)
        else:
            return self._internal_graph.copy(as_view=True)

    @property
    @requires_tree_generated
    def tree(self):
        return self._internal_graph.copy(as_view=True)

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

    @requires_tree_generated
    def get_tip_info(self, t_id: EntityID) -> TipInfo:
        return self._tinfo_cache[t_id]

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
    @requires_tree_generated
    def cut_points(self):
        return self._cut_points.copy()

    @property
    @requires_tree_generated
    def z_all(self) -> Vector:
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

    def _has_port_capacity(self, elem: Element) -> Tuple[bool, bool]:
        """
        Checks remaining capacity for input and output ports.
        Return whether the element can accept additional input and output connections.
        :param elem:
        :return: A tuple `(can_add_input, can_add_output)`
        """
        eid = self._resolve_elem_id(elem)

        ports = self._elem_port_type[eid]

        input_count = ports.count(PortType.INPUT)
        output_count = ports.count(PortType.OUTPUT)

        return input_count < elem.MAX_INPUTS, output_count < elem.MAX_OUTPUTS

    def add_elements(self, elems: Element | Iterable[Element]) -> TopologyHandler:
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
            # Initializing port entries for the new element
            self._elem_port_pos[elem.e_id] = []
            self._elem_port_type[elem.e_id] = []

            logger.debug("Adding element %r to the system.", elem.e_id)

        return self

    def add_root(self, target_elem: ElemLike, boundary_sv: VectorLike, output_pos: VectorLike | None = None,
                 output_port: int | None = None) -> EntityID:

        if output_pos is None and output_port is None:
            err_msg = "Either output_pos or output_port must be provided to add a root."
            logger.error(err_msg)
            raise ValueError(err_msg)
        elif output_pos is not None and output_port is not None:
            err_msg = "Only one of output_pos or output_port can be provided to add a root."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # NO PORT OVERWRITE PROTECTION
        if self._root is not None:
            err_msg = f"Root boundary is already defined as [{self._root.b_id}]."
            logger.error(err_msg)
            raise ValueError(err_msg)

        tgt_id = self._resolve_elem_id(target_elem)
        if tgt_id not in self._elements:
            err_msg = f"Cannot add root: target element [{tgt_id}] is missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        # Create a new port if needed, otherwise use the provided port index
        port_idx = None
        if output_pos is not None:
            # Check if the target can accept an additional output
            if not self._has_port_capacity(self._elements[tgt_id])[1]:
                err_msg = f"Cannot add root: target element [{tgt_id}] has reached its maximum output capacity."
                logger.error(err_msg)
                raise ValueError(err_msg)

            port_idx = len(self._elem_port_pos[tgt_id])
            self._elem_port_pos[tgt_id].append(np.array(output_pos, dtype=np.float64))
            self._elem_port_type[tgt_id].append(PortType.OUTPUT)

        elif output_port is not None:
            port_idx = output_port
            self._elem_port_type[tgt_id][port_idx] = PortType.OUTPUT

        # Create the root
        root_boundary = Boundary(b_id=f"{tgt_id}.{port_idx}_root", state_vector=boundary_sv)
        # Cache the new root
        self._root = root_boundary
        # Add it to the graph
        self._internal_graph.add_node(root_boundary.b_id)
        self._internal_graph.add_edge(tgt_id, root_boundary.b_id, output_port=port_idx)

        logger.debug(f"Added [{root_boundary.b_id}] as the root boundary to element [{tgt_id}] at port [{port_idx}].")

        return root_boundary.b_id

    def add_tip(self, target_element: ElemLike, boundary_sv: VectorLike, input_pos: VectorLike | None = None,
                input_port: int | None = None) -> EntityID:

        if input_pos is None and input_port is None:
            err_msg = "Either input_pos or input_port must be provided to add a tip."
            logger.error(err_msg)
            raise ValueError(err_msg)
        elif input_pos is not None and input_port is not None:
            err_msg = "Only one of input_pos or input_port can be provided to add a tip."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # NO PORT OVERWRITE PROTECTION
        tgt_id = self._resolve_elem_id(target_element)
        if tgt_id not in self._elements:
            err_msg = f"Cannot add tip: target element [{tgt_id}] is missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        # Create a new port if needed, otherwise use the provided port index
        port_idx = None

        if input_pos is not None:
            # Check if the target can accept an additional input
            if not self._has_port_capacity(self._elements[tgt_id])[0]:
                err_msg = f"Cannot add tip: target element [{tgt_id}] has reached its maximum input capacity."
                logger.error(err_msg)
                raise ValueError(err_msg)

            port_idx = len(self._elem_port_pos[tgt_id])
            self._elem_port_pos[tgt_id].append(np.array(input_pos, dtype=np.float64))
            self._elem_port_type[tgt_id].append(PortType.INPUT)
            # Set port as main input if needed
            if tgt_id not in self._elem_main_port:
                self._elem_main_port[tgt_id] = port_idx

        elif input_port is not None:
            port_idx = input_port
            self._elem_port_type[tgt_id][port_idx] = PortType.INPUT

        # Create the tip boundary
        tip_boundary = Boundary(b_id=f"tip_{tgt_id}.{port_idx}", state_vector=boundary_sv)
        # Cache the new tip boundary
        self._tips[tip_boundary.b_id] = tip_boundary
        # Add to graph
        self._internal_graph.add_node(tip_boundary.b_id)
        self._internal_graph.add_edge(tip_boundary.b_id, tgt_id, input_port=port_idx)

        logger.debug(f"Added [{tip_boundary.b_id}] as a tip boundary to element [{tgt_id}] at port [{port_idx}].")

        return tip_boundary.b_id

    def connect_elements(self, src: ElemLike, dst: ElemLike,
                         src_pos: VectorLike, dst_pos: VectorLike) -> TopologyHandler:

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

        # Check if the src (resp. dst) can accept an additional output (resp. input).
        if not self._has_port_capacity(self._elements[src_id])[1]:
            err_msg = f"{log_prefix} source element [{src_id}] has reached its maximum output capacity."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if not self._has_port_capacity(self._elements[dst_id])[0]:
            err_msg = f"{log_prefix} destination element [{dst_id}] has reached its maximum input capacity."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # After pre-checks are run, the elements can be linked
        # Creating ports
        out_port_idx = len(self._elem_port_pos[src_id])
        in_port_idx = len(self._elem_port_pos[dst_id])
        self._elem_port_pos[src_id].append(np.array(src_pos, dtype=np.float64))
        self._elem_port_type[src_id].append(PortType.OUTPUT)
        self._elem_port_pos[dst_id].append(np.array(dst_pos, dtype=np.float64))
        self._elem_port_type[dst_id].append(PortType.INPUT)
        # Set port as main input if needed
        if dst_id not in self._elem_main_port:
            self._elem_main_port[dst_id] = in_port_idx
        # Linking
        logger.debug(f"{log_prefix} linking from source port [{src_id}.{out_port_idx}] "
                     f"to destination port [{dst_id}.{in_port_idx}].")
        self._internal_graph.add_edge(src_id, dst_id, output_port=out_port_idx, input_port=in_port_idx)

        return self

    def cut_connection(self, connection: Iterable[ElemLike]) -> TopologyHandler:
        """
        "Cutting the hinge" i.e., generating virtual tips to account for multi-output or looping elements
        :param connection:
        :return:
        """

        src_id, dst_id = map(self._resolve_elem_id, connection)
        log_prefix = f"Cutting [{src_id}] -/> [{dst_id}]:"

        try:
            src_port_idx = self._internal_graph[src_id][dst_id]['output_port']
            dst_port_idx = self._internal_graph[src_id][dst_id]['input_port']
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
                    b_id1 = self.add_root(self._elements[src_id], NULL_SV, output_port=src_port_idx)
                    b_id2 = self.add_tip(self._elements[dst_id], NULL_SV, input_port=dst_port_idx)
                    # Cutting a connection in this case generates a new virtual INPUT/OUTPUT pair.
                    # No C sign matrix will be needed. Both state vectors are equal
                    cut = CutPoint(b_id1, b_id2, False)
                    logger.info(f"{log_prefix} cut closed loop")
                    logger.debug(f"Created new root [{b_id1}] and tip [{b_id2}].")
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
            b_id1 = self.add_tip(self._elements[src_id], NULL_SV, input_port=src_port_idx)
            b_id2 = self.add_tip(self._elements[dst_id], NULL_SV, input_port=dst_port_idx)
            # Cutting a connection in this case generates two new virtual INPUT tips/boundaries.
            # C sign matrix will be needed
            cut = CutPoint(b_id1, b_id2)
            logger.info(f"{log_prefix} cut connection.")
            logger.debug(f"Created new tips [{b_id1}] and [{b_id2}].")

        # clean up the old edge and save the cutting point relation
        self._cut_points.append(cut)
        self._internal_graph.remove_edge(src_id, dst_id)
        logger.debug(f"{log_prefix} old edge removed.")

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

    def make_tree(self) -> TopologyHandler:
        logger.info("(Re)generating tree from system topology.")
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
        self._tree_generated = True
        logger.info("Successfully built valid tree system.")
        self._recompute_caches()
        logger.debug(f"(Re)generated caches for {len(self._elements)} elements and {len(self._tips)} tips.")

        return self

    def _recompute_caches(self) -> None:
        # Z_all might change whenever the tree is (re)generated, so we need to (re)compute it here
        z = np.hstack([self.root.state_vector] + [b.state_vector for b in self._tips.values()])
        z.setflags(write=False)  # Make the array read-only
        self._z_all = z  # Overwrite cache

        # (Re)gen all ElementInfo objects
        self._einfo_cache = {}  # Clear invalid cache
        # Per element creation
        for e_id in self._elements:
            obj = self._elements[e_id]
            main_input_idx = self._elem_main_port[e_id]
            # We create all PortInfo objects for the element
            ports = []
            output_port = None
            downstream = None
            for port_pos, ptype in zip(self._elem_port_pos[e_id], self._elem_port_type[e_id]):
                pinfo = PortInfo(port_pos, ptype)
                if ptype == PortType.OUTPUT:  # the output is treated and stored separately for easy access
                    # Sanity check for multiple outputs even though this situation shouldn't be possible
                    if output_port is not None:
                        err_msg = f"Element {e_id} cannot have multiple output ports at this stage."
                        logger.critical(err_msg)
                        raise RuntimeError(err_msg)
                    # Buffer the successor ID of all element/boundary nodes to speed up the transfer matrix calculations
                    downstream = next(self._internal_graph.successors(e_id), None)
                    output_port = pinfo

                ports.append(pinfo)
            # Another sanity check for proper definition. An error at this point would indicate state corruption
            if output_port is None:
                err_msg = f"No output port found for element {e_id}."
                logger.critical(err_msg)
                raise RuntimeError(err_msg)
            if downstream is None:
                err_msg = f"Could not find downstream for element {e_id}."
                logger.critical(err_msg)
                raise RuntimeError(err_msg)

            # Next, we buffer each element's upstream tips. We don't want to keep testing unreachable branches.
            upstream_tips = [t for t in nx.ancestors(self._internal_graph, e_id) if t in self._tips]

            # Store the completed ElementInfo object
            einfo = ElementInfo(obj, main_input_idx, ports, output_port, upstream_tips, downstream)
            self._einfo_cache[e_id] = einfo

        # (Re)gen TipInfo objects
        self._tinfo_cache = {}  # Clear invalid cache
        for t_id in self._tips:
            obj = self._tips[t_id]
            downstream = next(self._internal_graph.successors(t_id))
            self._tinfo_cache[t_id] = TipInfo(obj, downstream)

    @requires_tree_generated
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
            # Move to the next node or terminate if we've reached the root
            if node == self.root.b_id:
                err_msg = f"No path found from [{src}] to [{tgt}]."
                logger.error(err_msg)
                raise ValueError(err_msg)
            elif node in self._tips:
                node = self._tinfo_cache[node].downstream
            else:  # Regular elements get caught here
                node = self._einfo_cache[node].downstream

        return path
