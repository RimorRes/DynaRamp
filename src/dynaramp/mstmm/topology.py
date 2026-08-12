from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps
from typing import Tuple, List, Dict, Iterable, KeysView, Sequence, cast

import networkx as nx
import numpy as np

from ..common.types import EntityID, is_entity_id, Vector, VectorLike
from .structs import NULL_SV, PortType, Element, ElemLike, Boundary, CutPoint

logger = logging.getLogger(__name__)


def requires_tree_generated(func):
    """Guard a method that is only meaningful once the tree has been generated."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._tree_generated:
            err_msg = "Invalid state for the requested operation: tree must be generated."
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        return func(self, *args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class PortInfo:
    """
    One port of an element.

    Attributes
    ----------
    pos : Vector
        Port position, in the element's local frame.
    ptype : PortType
        Whether the port is an input or the output.
    """
    pos: Vector
    ptype: PortType


@dataclass(frozen=True)
class ElementInfo:
    """
    Everything about one element that is fixed once the tree is generated.

    Assembled by :meth:`TopologyHandler._recompute_caches` so that the frequency sweep,
    which re-walks the whole tree at every frequency, never has to touch the graph.

    Attributes
    ----------
    obj : Element
        The element itself.
    main_input_idx : int
        Index into ``ports`` of the element's main input, the port its state vector is
        propagated from.
    ports : List[PortInfo]
        All ports, in creation order.
    output_port : PortInfo
        The element's single output port.
    upstream_tips : List[EntityID]
        Tip boundaries that feed this element (can be indirect).
    downstream : EntityID
        Successor node the output port feeds into.
    main_input_pred : EntityID
        Predecessor feeding the main input port. Resolved once here rather than by
        scanning predecessors on every query.
    """
    obj: Element
    main_input_idx: int
    ports: List[PortInfo]
    output_port: PortInfo
    upstream_tips: List[EntityID]
    downstream: EntityID
    main_input_pred: EntityID

    @property
    def main_input_pos(self) -> Vector:
        """Position of the main input port, in the element's local frame."""
        return self.ports[self.main_input_idx].pos


@dataclass(frozen=True)
class TipInfo:
    """
    A tip boundary and the element it feeds.

    Attributes
    ----------
    obj : Boundary
        The boundary itself.
    downstream : EntityID
        Successor node the tip feeds into.
    """
    obj: Boundary
    downstream: EntityID


@dataclass(frozen=True)
class PropagationStep:
    """
    One element traversal along a path, with all of its geometry already resolved.

    A path from a tip to the root is fixed by the topology and so is the port geometry
    of every element along it. Only the frequency changes between sweeps, so resolving
    the ports once turns the transfer-matrix assembly into a flat loop with no graph lookups.

    Attributes
    ----------
    elem : Element
        The element being traversed.
    input_pos : Vector
        Position of the port the path enters through, in the element's local frame.
    output_pos : Vector
        Position of the element's output port, in its local frame.
    is_main_input : bool
        Whether the path enters through the element's main input. If not, only the
        wrench is carried across, via :meth:`Element.u_extract`.
    """
    elem: Element
    input_pos: Vector
    output_pos: Vector
    is_main_input: bool


class TopologyHandler:
    """
    Builder and owner of a multibody system's connection topology.

    Elements are registered, wired together, and terminated with boundaries; then
    :meth:`make_tree` reduces the resulting graph to the tree that the transfer matrix
    method requires, cutting any extra connections and recording the cut relations.
    """

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
        self._path_cache: Dict[Tuple[EntityID, EntityID], List[EntityID]] = {}
        self._plan_cache: Dict[Tuple[EntityID, ...], Tuple[PropagationStep, ...]] = {}

    # Read-only and public-facing attributes
    @property
    def elements(self) -> Dict[EntityID, Element]:
        """A copy of the element registry, keyed by element ID."""
        # Return a copy to avoid accidental mutation of internal state
        return self._elements.copy()

    def get_element(self, eid: EntityID) -> Element:
        """
        Look up an element by ID.

        Parameters
        ----------
        eid : EntityID
            The element's identifier.

        Returns
        -------
        Element
            The registered element.
        """
        return self._elements[eid]

    @property
    def element_ids(self) -> KeysView[EntityID]:
        """
        A live view of the registered element IDs.

        Prefer this to iterating :attr:`elements` when only the identifiers are needed:
        that property copies the whole registry on every access, which is costly inside
        a traversal loop.
        """
        return self._elements.keys()

    def has_element(self, e_id: EntityID) -> bool:
        """
        Whether an ID refers to a registered element.

        Parameters
        ----------
        e_id : EntityID
            The identifier to test.

        Returns
        -------
        bool
            True if the ID is a registered element.
        """
        return e_id in self._elements

    def is_tip(self, b_id: EntityID) -> bool:
        """
        Whether an ID refers to a tip boundary.

        Parameters
        ----------
        b_id : EntityID
            The identifier to test.

        Returns
        -------
        bool
            True if the ID is a tip boundary.
        """
        return b_id in self._tips

    @requires_tree_generated
    def get_element_info(self, e_id: EntityID) -> ElementInfo:
        """
        The cached :class:`ElementInfo` for an element.

        Parameters
        ----------
        e_id : EntityID
            The element's identifier.

        Returns
        -------
        ElementInfo
            Its resolved topology information.
        """
        return self._einfo_cache[e_id]

    @requires_tree_generated
    def main_input_of(self, e_id: EntityID) -> Tuple[EntityID, Vector]:
        """
        The predecessor feeding an element's main input port, with that port's position.

        An element's state vector is propagated from its main input, so this pair is the
        anchor for anything evaluated inside the element: its transfer matrix, its modal
        mass, or a mode shape at some station along it.

        Parameters
        ----------
        e_id : EntityID
            The element to look up.

        Returns
        -------
        tuple
            ``(predecessor_id, main_input_position)``.
        """
        info = self._einfo_cache[e_id]
        return info.main_input_pred, info.main_input_pos

    @property
    def graph(self) -> nx.DiGraph[EntityID]:
        """A read-only view of the user-defined graph, or the working graph before reduction."""
        if self._tree_generated:
            return self._user_graph.copy(as_view=True)
        else:
            return self._internal_graph.copy(as_view=True)

    @property
    @requires_tree_generated
    def tree(self):
        """A read-only view of the reduced tree."""
        return self._internal_graph.copy(as_view=True)

    @property
    def root(self) -> Boundary:
        """
        The root boundary.

        Raises
        ------
        ValueError
            If no root has been defined yet.
        """
        if self._root is None:
            err_msg = "Root element not identified."
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            return self._root

    @property
    def tips(self) -> Dict[EntityID, Boundary]:
        """A copy of the tip-boundary registry."""
        # Return a copy to avoid accidental mutation of internal state
        return self._tips.copy()

    @requires_tree_generated
    def get_tip_info(self, t_id: EntityID) -> TipInfo:
        """
        The cached :class:`TipInfo` for a tip boundary.

        Parameters
        ----------
        t_id : EntityID
            The tip's identifier.

        Returns
        -------
        TipInfo
            Its resolved topology information.
        """
        return self._tinfo_cache[t_id]

    @property
    def boundaries(self) -> List[EntityID]:
        """
        All boundary IDs, ordered ``[root, tip1, tip2, ...]``.

        Purposefully not cached: every call returns a fresh list, so a caller cannot
        mutate the handler's state by accident.
        """
        # Avoid calling `self.root` here because the property logs and raises; check internal `_root` directly.
        if self._root is None:
            logger.warning("Root boundary not defined. Returning only tip boundaries.")
            bound = [t_id for t_id in self._tips]
        else:
            bound = [self._root.b_id] + [t_id for t_id in self._tips]
        return bound

    @property
    @requires_tree_generated
    def cut_points(self) -> List[CutPoint]:
        """A copy of the cut-point relations recorded during tree reduction."""
        return self._cut_points.copy()

    @property
    @requires_tree_generated
    def z_all(self) -> Vector:
        """The concatenated boundary state vectors, root first. Read-only."""
        return self._z_all

    @staticmethod
    def _resolve_elem_id(elem_or_eid: ElemLike) -> EntityID:
        """
        Accept either an element or a bare ID and return the ID.

        Parameters
        ----------
        elem_or_eid : ElemLike
            An element, or an identifier.

        Returns
        -------
        EntityID
            The resolved identifier.

        Raises
        ------
        TypeError
            If neither form is recognized.
        """
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
        Remaining input and output port capacity of an element.

        Parameters
        ----------
        elem : Element
            The element to check.

        Returns
        -------
        tuple of bool
            ``(can_add_input, can_add_output)``.
        """
        eid = self._resolve_elem_id(elem)

        ports = self._elem_port_type[eid]

        input_count = ports.count(PortType.INPUT)
        output_count = ports.count(PortType.OUTPUT)

        in_cond = True if elem.MAX_INPUTS is None else input_count < elem.MAX_INPUTS
        out_cond = True if elem.MAX_OUTPUTS is None else output_count < elem.MAX_OUTPUTS

        return in_cond, out_cond

    def _add_port(self, tgt_id: EntityID, pos: VectorLike, ptype: PortType) -> int:
        """
        Register a new port on an element and return its index.

        Parameters
        ----------
        tgt_id : EntityID
            The element to add the port to.
        pos : VectorLike
            Port position, in the element's local frame.
        ptype : PortType
            Input or output.

        Returns
        -------
        int
            Index of the new port.
        """
        port_idx = len(self._elem_port_pos[tgt_id])
        self._elem_port_pos[tgt_id].append(np.asarray(pos, dtype=np.float64))
        self._elem_port_type[tgt_id].append(ptype)
        # The first input registered on an element becomes its main input.
        if ptype == PortType.INPUT and tgt_id not in self._elem_main_port:
            self._elem_main_port[tgt_id] = port_idx
        return port_idx

    @staticmethod
    def _resolve_port_argument(pos: VectorLike | None, port: int | None, kind: str) -> None:
        """
        Validate that exactly one of a position / port-index pair was supplied.

        Parameters
        ----------
        pos : VectorLike | None
            The position argument.
        port : int | None
            The port-index argument.
        kind : str
            Argument name stem used in the error message, e.g. ``"output"``.

        Raises
        ------
        ValueError
            If both or neither were supplied.
        """
        if pos is None and port is None:
            err_msg = f"Either {kind}_pos or {kind}_port must be provided."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if pos is not None and port is not None:
            err_msg = f"Only one of {kind}_pos or {kind}_port can be provided."
            logger.error(err_msg)
            raise ValueError(err_msg)

    def add_elements(self, elems: Element | Iterable[Element]) -> TopologyHandler:
        """
        Register one or more elements with the system.

        Parameters
        ----------
        elems : Element | Iterable[Element]
            The element(s) to add.

        Returns
        -------
        TopologyHandler
            ``self``, for chaining.

        Raises
        ------
        TypeError
            If any argument is not an :class:`Element`.
        """
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
        """
        Terminate an element's output with the system's root boundary.

        Parameters
        ----------
        target_elem : ElemLike
            Element to attach the root to.
        boundary_sv : VectorLike
            The 12-component boundary state, ``None`` for each unknown component.
        output_pos : VectorLike | None
            Position at which to create a new output port. Mutually exclusive with
            ``output_port``.
        output_port : int | None
            Index of an existing port to convert to the output. Mutually exclusive with
            ``output_pos``.

        Returns
        -------
        EntityID
            The new root boundary's ID.

        Raises
        ------
        ValueError
            If a root already exists, the argument pair is invalid, or the element has
            no remaining output capacity.
        KeyError
            If the target element is not registered.
        """
        self._resolve_port_argument(output_pos, output_port, "output")

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
        if output_pos is not None:
            # Check if the target can accept an additional output
            if not self._has_port_capacity(self._elements[tgt_id])[1]:
                err_msg = f"Cannot add root: target element [{tgt_id}] has reached its maximum output capacity."
                logger.error(err_msg)
                raise ValueError(err_msg)
            port_idx = self._add_port(tgt_id, output_pos, PortType.OUTPUT)
        else:
            port_idx = cast(int, output_port)
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
        """
        Terminate an element's input with a tip boundary.

        Parameters
        ----------
        target_element : ElemLike
            Element to attach the tip to.
        boundary_sv : VectorLike
            The 12-component boundary state, ``None`` for each unknown component.
        input_pos : VectorLike | None
            Position at which to create a new input port. Mutually exclusive with
            ``input_port``.
        input_port : int | None
            Index of an existing port to convert to an input. Mutually exclusive with
            ``input_pos``.

        Returns
        -------
        EntityID
            The new tip boundary's ID.

        Raises
        ------
        ValueError
            If the argument pair is invalid or the element has no remaining input capacity.
        KeyError
            If the target element is not registered.
        """
        self._resolve_port_argument(input_pos, input_port, "input")

        # NO PORT OVERWRITE PROTECTION
        tgt_id = self._resolve_elem_id(target_element)
        if tgt_id not in self._elements:
            err_msg = f"Cannot add tip: target element [{tgt_id}] is missing from the system."
            logger.error(err_msg)
            raise KeyError(err_msg)

        # Create a new port if needed, otherwise use the provided port index
        if input_pos is not None:
            # Check if the target can accept an additional input
            if not self._has_port_capacity(self._elements[tgt_id])[0]:
                err_msg = f"Cannot add tip: target element [{tgt_id}] has reached its maximum input capacity."
                logger.error(err_msg)
                raise ValueError(err_msg)
            port_idx = self._add_port(tgt_id, input_pos, PortType.INPUT)
        else:
            port_idx = cast(int, input_port)
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
        """
        Connect one element's output to another element's input.

        Parameters
        ----------
        src, dst : ElemLike
            Source and destination elements.
        src_pos, dst_pos : VectorLike
            Port positions, each in its own element's local frame.

        Returns
        -------
        TopologyHandler
            ``self``, for chaining. Self-loops and parallel edges are refused with a
            warning rather than an exception.

        Raises
        ------
        KeyError
            If either element is not registered.
        ValueError
            If either element has no remaining port capacity.
        """
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
        out_port_idx = self._add_port(src_id, src_pos, PortType.OUTPUT)
        in_port_idx = self._add_port(dst_id, dst_pos, PortType.INPUT)

        logger.debug(f"{log_prefix} linking from source port [{src_id}.{out_port_idx}] "
                     f"to destination port [{dst_id}.{in_port_idx}].")
        self._internal_graph.add_edge(src_id, dst_id, output_port=out_port_idx, input_port=in_port_idx)

        return self

    def cut_connection(self, connection: Iterable[ElemLike]) -> TopologyHandler:
        """
        Cut a connection, replacing it with a pair of virtual boundaries.

        "Cutting the hinge": generates virtual tips to account for multi-output or
        looping elements and records the relation between the two new boundaries so
        the overall transfer equation can recombine them.

        Parameters
        ----------
        connection : Iterable[ElemLike]
            The ``(source, destination)`` pair identifying the edge to cut.

        Returns
        -------
        TopologyHandler
            ``self``, for chaining.

        Raises
        ------
        KeyError
            If the connection does not exist.
        ValueError
            If the connection touches a boundary, or if cutting it would isolate the
            downstream elements.
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
            in_cycle = any(
                src_id in scc and len(scc) > 1
                for scc in nx.strongly_connected_components(self._internal_graph)
            )
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
        """
        Identify the connections that must be removed to get a tree system.

        A valid and unique root must already have been selected.

        Returns
        -------
        List[tuple]
            The ``(source, destination)`` edges to cut.
        """
        connections_to_cut = []
        # Handle diverging nodes (out_degree > 1)
        diverging_nodes = [n for n in self._internal_graph if self._internal_graph.out_degree(n) > 1]
        for dnode in diverging_nodes:
            # Each element can only have one output.
            # We can choose to only keep the output with the shortest path to root
            preserved = nx.shortest_path(self._internal_graph, source=dnode, target=self.root.b_id)[1]
            for n in self._internal_graph[dnode]:
                if n != preserved:
                    connections_to_cut.append((dnode, n))
        # Handle the special closed loop case
        # Collect strongly connected components, in our case this is analogous to directed cycles
        root_cyc = None
        for cyc in nx.strongly_connected_components(self._internal_graph):
            if len(cyc) > 1 and all(self._internal_graph.out_degree(node) == 1 for node in cyc):
                root_cyc = cyc
                break
        if root_cyc:
            # We now need to choose where to cut.
            # Prefer cutting at an entry point of the loop (node with in_degree > 1). If none exists, cut anywhere.
            entry = next((n for n in root_cyc if self._internal_graph.in_degree(n) > 1), None)
            node = entry if entry is not None else next(iter(root_cyc))
            pred_in_cycle = next(x for x in self._internal_graph.predecessors(node) if x in root_cyc)
            connections_to_cut.append((pred_in_cycle, node))

        # Return edges to be cut
        logger.info(f"Found {len(connections_to_cut)} cuts to be made.")
        for i, (src, dst) in enumerate(connections_to_cut):
            logger.debug(f"Cut {i}: {src}->{dst}")

        return connections_to_cut

    def make_tree(self) -> TopologyHandler:
        """
        Reduce the topology to a tree and rebuild every cache.

        Returns
        -------
        TopologyHandler
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the topology cannot be reduced to a valid tree.
        """
        logger.info("(Re)generating tree from system topology.")
        self._user_graph = self._internal_graph.copy()

        logger.debug("Auto resolving edges to cut.")
        c2c = self.find_cuts()  # Connections to cut
        for connection in c2c:
            self.cut_connection(connection)

        # Check that we have a correct tree system.
        # The actual check needs to be run on a reversed view of the graph since the root is the sink, not the source.
        # is_arborescence allows for an in_degree <= 1,
        # but with the current algorithm any element node with in_degree == 0 would be disconnected
        # (only tips can verify this condition and be connected as they are not internal nodes);
        # ergo this effectively validates the tree structure where every element has exactly one output.
        if not nx.is_arborescence(nx.reverse_view(self._internal_graph)):
            err_msg = "Invalid topology. Could not transform system into a tree."
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.debug("Validated topology.")
        self._tree_generated = True
        logger.info("Successfully built valid tree system.")
        self._recompute_caches()
        logger.debug(f"(Re)generated caches for {len(self._elements)} elements and {len(self._tips)} tips.")

        return self

    def _recompute_caches(self) -> None:
        """Rebuild every derived cache after a change of topology."""
        # Z_all might change whenever the tree is (re)generated, so we need to (re)compute it here
        z = np.hstack([self.root.state_vector] + [b.state_vector for b in self._tips.values()])
        z.setflags(write=False)  # Make the array read-only
        self._z_all = z  # Overwrite cache

        # Topology keys path and plan caches, so they die with it.
        self._path_cache = {}
        self._plan_cache = {}

        # (Re)gen all ElementInfo objects
        self._einfo_cache = {}  # Clear invalid cache
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

            # Resolve the main input's predecessor a single time. This used to be a predecessor scan
            # performed on every mode-shape and modal-mass query.
            try:
                main_pred = next(
                    n for n in self._internal_graph.predecessors(e_id)
                    if self._internal_graph[n][e_id]["input_port"] == main_input_idx
                )
            except StopIteration as exc:
                err_msg = f"Element [{e_id}] has no predecessor on its main input port."
                logger.error(err_msg)
                raise ValueError(err_msg) from exc

            # Next, we buffer each element's upstream tips. We don't want to keep testing unreachable branches.
            upstream_tips = [t for t in nx.ancestors(self._internal_graph, e_id) if t in self._tips]

            # Store the completed ElementInfo object
            downstream = cast(EntityID, downstream)  # Downstream is guaranteed
            self._einfo_cache[e_id] = ElementInfo(
                obj, main_input_idx, ports, output_port, upstream_tips, downstream, main_pred,
            )

        # (Re)gen TipInfo objects
        self._tinfo_cache = {}  # Clear invalid cache
        for t_id in self._tips:
            obj = self._tips[t_id]
            downstream = next(self._internal_graph.successors(t_id))
            self._tinfo_cache[t_id] = TipInfo(obj, downstream)

    @requires_tree_generated
    def resolve_branch_up_to(self, src: EntityID, tgt: EntityID) -> List[EntityID]:
        """
        The path from ``src`` down to ``tgt``, excluding ``tgt``.

        Paths are fixed by the topology, so each one is resolved at most once and served
        from a cache thereafter.

        Parameters
        ----------
        src : EntityID
            Starting node.
        tgt : EntityID
            Node to stop at, itself excluded from the result.

        Returns
        -------
        List[EntityID]
            The ordered node IDs.

        Raises
        ------
        ValueError
            If the root is reached without meeting ``tgt``.
        """
        cached = self._path_cache.get((src, tgt))
        if cached is not None:
            return cached

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

        self._path_cache[(src, tgt)] = path
        return path

    @requires_tree_generated
    def propagation_plan(self, path: Sequence[EntityID]) -> Tuple[PropagationStep, ...]:
        """
        Resolve a path into the sequence of element traversals it implies.

        Everything a transfer-matrix product needs except the frequency: which element,
        entered at which port, leaving at which port, and whether that port is the
        element's main input. All of it is fixed by the topology, so it is resolved once
        per path and cached.

        Parameters
        ----------
        path : Sequence[EntityID]
            An ordered path, as returned by :meth:`resolve_branch_up_to`. The first node
            is the origin, whose own transfer matrix is not applied.

        Returns
        -------
        tuple of PropagationStep
            One step per element traversed, in order.
        """
        key = tuple(path)
        cached = self._plan_cache.get(key)
        if cached is not None:
            return cached

        steps: List[PropagationStep] = []
        for prev_id, e_id in zip(key, key[1:]):
            # Because of the way paths are used, e_id can never be a boundary.
            info = self._einfo_cache[e_id]
            in_port_idx: int = self._internal_graph[prev_id][e_id]['input_port']
            steps.append(PropagationStep(
                elem=info.obj,
                input_pos=info.ports[in_port_idx].pos,
                output_pos=info.output_port.pos,
                is_main_input=(in_port_idx == info.main_input_idx),
            ))

        plan = tuple(steps)
        self._plan_cache[key] = plan
        return plan
