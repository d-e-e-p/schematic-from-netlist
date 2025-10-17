import csv
import json
import logging as log
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jpype
import jpype.imports
import numpy as np
from jpype import JClass, JDouble, JInt
from shapely.affinity import rotate, scale, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree
from sklearn.decomposition import PCA
from tabulate import tabulate

from schematic_from_netlist.interfaces.elk_utils import ElkUtils
from schematic_from_netlist.interfaces.symbol_library import SymbolLibrary
from schematic_from_netlist.utils.config import setup_elk

setup_elk()  # Initialize ELK JVM

# ruff: noqa: E402
# pyright: reportMissingImports=false

from java.lang import Double, Integer
from org.eclipse.elk.alg.force.options import ForceMetaDataProvider, ForceModelStrategy, ForceOptions, StressOptions
from org.eclipse.elk.alg.layered.options import (
    CrossingMinimizationStrategy,
    EdgeStraighteningStrategy,
    FixedAlignment,
    LayeredMetaDataProvider,
    LayeredOptions,
    LayeringStrategy,
    NodePlacementStrategy,
    OrderingStrategy,
    SelfLoopDistributionStrategy,
    WrappingStrategy,
)
from org.eclipse.elk.alg.spore.options import SporeCompactionOptions

# ELK Core Engine
from org.eclipse.elk.core import RecursiveGraphLayoutEngine
from org.eclipse.elk.core.data import LayoutMetaDataService

# ELK Option classes for layout configuration
from org.eclipse.elk.core.options import CoreOptions, Direction, EdgeRouting, PortAlignment, PortConstraints, PortSide
from org.eclipse.elk.core.util import BasicProgressMonitor

# ELK IO Utility
from org.eclipse.elk.graph.properties import IProperty, Property

# ELK Model classes for programmatic graph construction
from org.eclipse.elk.graph.util import ElkGraphUtil

# Map LTSpice orientations to Shapely transform directives
ORIENTATIONS = {
    "R0": {"rotate": 0, "mirror_x": False, "mirror_y": False},
    "R90": {"rotate": 90, "mirror_x": False, "mirror_y": False},
    "R180": {"rotate": 180, "mirror_x": False, "mirror_y": False},
    "R270": {"rotate": 270, "mirror_x": False, "mirror_y": False},
    "M0": {"rotate": 0, "mirror_x": False, "mirror_y": True},
    "M90": {"rotate": 90, "mirror_x": False, "mirror_y": True},
    "M180": {"rotate": 180, "mirror_x": True, "mirror_y": False},
    "M270": {"rotate": 270, "mirror_x": False, "mirror_y": True},
}


class ElkInterface:
    def __init__(self, db):
        self.db = db
        self.output_dir = "data/json"
        self.size_node_base_w = 6
        self.size_node_base_h = 18
        self.size_factor_per_pin = 2  #  ie size_min_macro + size_factor_per_pin * degree

        self.symbol_outlines = SymbolLibrary().get_symbol_outlines()
        self.keep_port_loc_property = Property("keep_port_loc ", False)
        self.elk_utils = ElkUtils()

    # Main layout generation routines
    def generate_layout_figures(self):
        self.db._build_lookup_tables()
        sorted_modules = sorted(self.db.design.modules.values(), key=lambda m: m.depth, reverse=True)
        for module in sorted_modules:
            if not module.is_leaf:
                self.generate_module_layout(module)

    def generate_module_layout(self, module):
        log.info(f"Generating layout for module {module.name}")

        graph = self.create_graph(module, "pass1")
        log.info(f" nodes = {[str(node) for node in graph.getChildren()]}")
        log.info(f" edges = {[str(edge) for edge in graph.getContainedEdges()]}")
        self.layout_graph(graph, "pass1")
        self.evaluate_layout(module)

        # self.create_ranks_from_elk_graph(module, graph)
        # self.dump_elk_config_table(graph)
        # self.update_graph_with_symbols(module, graph)
        # graph = self.create_graph(module, "pass2")
        # self.layout_graph(graph, "pass2")

        # for edge in graph.getContainedEdges():
        #     edge.getSections().clear()
        #     edge.getProperties().clear()
        # self.layout_graph(graph, "pass3")

        self.extract_geometry(graph, module)

    def create_graph(self, module, stage):
        graph = ElkGraphUtil.createGraph()
        graph.setIdentifier(module.name)

        nodes, ports = self.create_nodes(module, graph, stage)
        self.create_edges(module, graph, nodes, ports, stage)
        log.info(f" nodes = {[str(node) for node in graph.getChildren()]}")
        log.info(f" edges = {[str(edge) for edge in graph.getContainedEdges()]}")
        # self.remove_floating_nodes(module, graph)
        return graph

    # Graph creation and layout
    def create_nodes(self, module, graph, stage):
        nodes = {}
        ports = {}
        port_size = 0

        # create top level node
        node = ElkGraphUtil.createNode(graph)
        node.setIdentifier(module.name)
        node.setWidth(1)
        node.setHeight(1)
        node.setX(1)
        node.setY(1)
        node.setProperty(CoreOptions.PORT_CONSTRAINTS, PortConstraints.FIXED_POS)
        nodes[module.name] = node

        for inst in module.instances.values():
            # leaf? get from gv run. module? get from previous elk run and assume pin locs are still valid
            (x1, y1, x2, y2) = inst.draw.efig
            if inst.module.is_leaf:
                (width, height) = (x2 - x1, y2 - y1)
            else:
                (width, height) = inst.module.draw.efig
            port = ElkGraphUtil.createPort(node)
            port.setWidth(width)
            port.setHeight(height)
            port.setX(x1)
            port.setY(y1)
            port.setIdentifier(inst.name)
            # NODE {id} {x1} {y1} {x2} {y2}
            log.info(f"R NODE {inst.name} {x1} {y1} {x2} {y2}")
            for pin in inst.pins.values():
                if pin.draw.efig:
                    port = ElkGraphUtil.createPort(node)
                    port.setIdentifier(pin.full_name)
                    port.setWidth(port_size)
                    port.setHeight(port_size)
                    (x, y) = pin.draw.efig
                    port.setX(x)
                    port.setY(y)
                    direction = self.get_port_direction(pin.draw.efig, inst.draw.efig)
                    # PORT {port id} {node id} {side} {x} {y}
                    log.info(f"R PORT {pin.full_name} {inst.name} {direction} {x - x1} {y - y1}")
                    port.setProperty(CoreOptions.PORT_SIDE, direction)
                    ports[pin.full_name] = port
        return nodes, ports

    def get_port_direction(self, pin_loc: Tuple[float, float], inst_loc: Tuple[float, float, float, float]) -> PortSide:
        """
        Return which side of the instance boundary the pin is closest to.
        """
        x1, y1, x2, y2 = inst_loc
        x, y = pin_loc

        distances = {
            PortSide.WEST: abs(x - x1),
            PortSide.EAST: abs(x2 - x),
            PortSide.NORTH: abs(y - y1),
            PortSide.SOUTH: abs(y2 - y),
        }

        # side with the smallest distance
        closest_side: PortSide = min(distances, key=lambda s: distances[s])
        return closest_side

    def create_edges(self, module, graph, nodes, ports, stage):
        """
        Create edges respecting rank-based hierarchy.
        For multi-rank nets: chain from lowest to highest rank.
        """
        edges = []

        for net in module.nets.values():
            if net.num_conn > self.db.fanout_threshold:
                continue

            pins = list(net.pins.values())
            if len(pins) <= 1:
                continue

            # Group pins by rank
            rank_to_pins = defaultdict(list)
            unranked_pins = []

            for pin in pins:
                rank = getattr(pin.instance, "rank", -1)
                if rank >= 0:
                    rank_to_pins[rank].append(pin)
                else:
                    unranked_pins.append(pin)

            # If no ranked pins, fall back to original behavior
            if not rank_to_pins:
                self._create_star_edges(net, pins, nodes, ports, graph)
                continue

            # Sort ranks low to high
            sorted_ranks = sorted(rank_to_pins.keys())

            # Create chained connections between ranks
            net_index = 0

            for i in range(len(sorted_ranks)):
                current_rank = sorted_ranks[i]
                current_pins = rank_to_pins[current_rank]

                if i == 0:
                    # First rank: pick driver (pin with most fanout)
                    current_pins_sorted = sorted(current_pins, key=lambda p: len(p.instance.pins.values()), reverse=True)
                    driver = current_pins_sorted[0]
                    current_sinks = current_pins_sorted[1:]  # Other pins in same rank
                else:
                    # Subsequent ranks: all pins are sinks from previous rank
                    driver = None
                    current_sinks = current_pins

                # Connect to next rank(s)
                if i < len(sorted_ranks) - 1:
                    next_rank = sorted_ranks[i + 1]
                    next_pins = rank_to_pins[next_rank]

                    # Chain from current rank to next rank
                    if driver:
                        # From driver in current rank to all pins in next rank
                        for dst_pin in next_pins:
                            edge = self._create_one_edge(net, driver, dst_pin, nodes, ports, graph, net_index)
                            if edge:
                                edges.append(edge)
                                net_index += 1
                    else:
                        # From first pin in current rank to all pins in next rank
                        src_pin = current_pins[0]
                        for dst_pin in next_pins:
                            edge = self._create_one_edge(net, src_pin, dst_pin, nodes, ports, graph, net_index)
                            if edge:
                                edges.append(edge)
                                net_index += 1

                # Connect sinks within same rank (if driver exists)
                if driver and current_sinks:
                    for sink_pin in current_sinks:
                        edge = self._create_one_edge(net, driver, sink_pin, nodes, ports, graph, net_index)
                        if edge:
                            edges.append(edge)
                            net_index += 1

            # Handle unranked pins: connect them to the lowest rank
            if unranked_pins and sorted_ranks:
                lowest_rank_pins = rank_to_pins[sorted_ranks[0]]
                src_pin = lowest_rank_pins[0]  # Pick first pin in lowest rank

                for unranked_pin in unranked_pins:
                    edge = self._create_one_edge(net, src_pin, unranked_pin, nodes, ports, graph, net_index)
                    if edge:
                        edges.append(edge)
                        net_index += 1

        return edges

    def _create_one_edge(self, net, src_pin, dst_pin, nodes, ports, graph, net_index):
        """Helper to create a single edge between two pins."""
        src_node = ports.get(src_pin.full_name)
        dst_node = ports.get(dst_pin.full_name)

        if not src_node or not dst_node or src_node == dst_node:
            return None

        net_id = f"{net.name}_idx{net_index}"
        edge = ElkGraphUtil.createEdge(graph)
        edge.setProperty(CoreOptions.EDGE_THICKNESS, 0.0)
        # edge.setProperty(CoreOptions.DIRECTION, Direction.UNDEFINED)
        edge.setIdentifier(net_id)
        edge.getSources().add(src_node)
        edge.getTargets().add(dst_node)
        # PEDGEP {edge id} {source node id} {target node id} {source port id} {target port id}
        log.info(f"R PEDGEP {net_id} {src_pin.instance.name} {dst_pin.instance.name} {src_pin.full_name} {dst_pin.full_name}")

        return edge

    def _create_star_edges(self, net, pins, nodes, ports, graph):
        """Fallback: create star topology (original behavior)."""
        net_index = 0
        sorted_pins = sorted(pins, key=lambda pin: len(pin.instance.pins.values()), reverse=True)
        src = sorted_pins[0]

        for dst in sorted_pins[1:]:
            self._create_one_edge(net, src, dst, nodes, ports, graph, net_index)
            net_index += 1

    def layout_graph(self, graph, step: str = ""):
        if step == "pass1":
            service = LayoutMetaDataService.getInstance()
            provider = LayeredMetaDataProvider()
            service.registerLayoutMetaDataProviders(provider)
            algorithms = service.getAlgorithmData()
            log.debug([algo.getId() for algo in algorithms])
            graph.setProperty(CoreOptions.ALGORITHM, "layered")
            graph.setProperty(LayeredOptions.PARTITIONING_ACTIVATE, False)
            graph.setProperty(CoreOptions.PORT_CONSTRAINTS, PortConstraints.FIXED_POS)
            graph.setProperty(LayeredOptions.NODE_PLACEMENT_STRATEGY, NodePlacementStrategy.INTERACTIVE)
            graph.setProperty(LayeredOptions.LAYERING_STRATEGY, LayeringStrategy.INTERACTIVE)
        elif step == "pass2":
            service = LayoutMetaDataService.getInstance()
            provider = ForceMetaDataProvider()
            service.registerLayoutMetaDataProviders(provider)
            algorithms = service.getAlgorithmData()
            log.debug([algo.getId() for algo in algorithms])
            graph.setProperty(CoreOptions.ALGORITHM, "force")
            graph.setProperty(ForceOptions.INTERACTIVE, True)

        # graph.setProperty(CoreOptions.ASPECT_RATIO, 2.0)

        graph.setProperty(CoreOptions.DEBUG_MODE, True)
        graph.setProperty(ForceOptions.ITERATIONS, Integer(3000))  # default 300
        # graph.setProperty(ForceOptions.SPACING_NODE_NODE, Double(80))  # default 80 ?

        graph.setProperty(LayeredOptions.MERGE_EDGES, True)
        """
        graph.setProperty(SporeCompactionOptions.UNDERLYING_LAYOUT_ALGORITHM, "layered")

        graph.setProperty(LayeredOptions.LAYERING_STRATEGY, LayeringStrategy.NETWORK_SIMPLEX)
        graph.setProperty(LayeredOptions.EDGE_ROUTING, EdgeRouting.ORTHOGONAL)
        graph.setProperty(LayeredOptions.ALLOW_NON_FLOW_PORTS_TO_SWITCH_SIDES, True)
        graph.setProperty(LayeredOptions.COMPACTION_CONNECTED_COMPONENTS, True)
        graph.setProperty(LayeredOptions.CONSIDER_MODEL_ORDER_CROSSING_COUNTER_NODE_INFLUENCE, 0.001)
        graph.setProperty(LayeredOptions.CROSSING_MINIMIZATION_GREEDY_SWITCH_ACTIVATION_THRESHOLD, Integer(0))  # always on
        graph.setProperty(LayeredOptions.EDGE_ROUTING_SELF_LOOP_DISTRIBUTION, SelfLoopDistributionStrategy.EQUALLY)  # always on
        graph.setProperty(
            LayeredOptions.NODE_PLACEMENT_STRATEGY, NodePlacementStrategy.NETWORK_SIMPLEX
        )  # better than default BRANDES_KOEPF
        graph.setProperty(LayeredOptions.SEPARATE_CONNECTED_COMPONENTS, False)  #
        graph.setProperty(LayeredOptions.THOROUGHNESS, Integer(1000))  #  num of minimizeCrossingsWithCounter loops
        graph.setProperty(LayeredOptions.WRAPPING_STRATEGY, WrappingStrategy.MULTI_EDGE)  #  bypoass chunks
        graph.setProperty(LayeredOptions.GENERATE_POSITION_AND_LAYER_IDS, True)
        graph.setProperty(LayeredOptions.CONSIDER_MODEL_ORDER_STRATEGY, OrderingStrategy.NONE)
        graph.setProperty(LayeredOptions.NODE_PLACEMENT_FAVOR_STRAIGHT_EDGES, True)
        graph.setProperty(LayeredOptions.NODE_PLACEMENT_BK_EDGE_STRAIGHTENING, EdgeStraighteningStrategy.IMPROVE_STRAIGHTNESS)
        # with unzipping enabled, ELK can insert an extra layer (or more),
        graph.setProperty(LayeredOptions.LAYER_UNZIPPING_MINIMIZE_EDGE_LENGTH, True)
        graph.setProperty(LayeredOptions.LAYER_UNZIPPING_LAYER_SPLIT, Integer(2))
        """

        # Force ELK to ignore old coordinates
        self.write_graph_to_file(graph, f"pre_{step}")

        layout_engine = RecursiveGraphLayoutEngine()
        monitor = BasicProgressMonitor()
        layout_engine.layout(graph, monitor)
        self.write_graph_to_file(graph, f"post_{step}")

    # Geometry extraction and evaluation
    def extract_geometry(self, graph, module):
        def walk_graph_and_extract_data(node, module):
            """
            is_root = node.getParent() is None
            if is_root:
                module.draw.efig = (node.getWidth(), node.getHeight())
            else:
                node_id = node.getIdentifier()
                instance = module.instances[node_id]
                instance.draw.efig = (
                    node.getX(),
                    node.getY(),
                    node.getX() + node.getWidth(),
                    node.getY() + node.getHeight(),
                )
                for port in node.getPorts():
                    port_id = port.getIdentifier()
                    pin = module.pins[port_id]
                    pin.draw.efig = (
                        port.getX(),
                        port.getY(),
               for node in graph.getChildren():     )
            """

            for edge in node.getContainedEdges():
                net_id = edge.getIdentifier()
                net = module.nets[self.id2name(net_id)]
                if not edge.isConnected():
                    log.warning(f"Net {net.name} is not connected")
                for section in edge.getSections():
                    # collect points in order
                    points = [(section.getStartX(), section.getStartY())]
                    for bp in section.getBendPoints():
                        points.append((bp.getX(), bp.getY()))
                    points.append((section.getEndX(), section.getEndY()))

                    # create (start, end) pairs
                    for i in range(len(points) - 1):
                        start = points[i]
                        end = points[i + 1]
                        net.draw.efig.append((start, end))

            for child in node.getChildren():
                walk_graph_and_extract_data(child, module)

        walk_graph_and_extract_data(graph, module)

    def remove_floating_nodes(self, module, graph):
        """
        Removes nodes from the graph that do not have any incident edges.
        Assumes:
          - graph.getContainedEdges() returns a list of edge dicts with 'sources' and 'targets'
          - graph.getChildren() returns a list of child node dicts with 'id'
        """
        # Step 1: Build a set of node IDs that are connected to any edge
        nodes_with_edges = set()
        for edge in graph.getContainedEdges():
            for node in edge.getSources() + edge.getTargets():
                portname = node.getIdentifier()
                nodes_with_edges.add(node.getIdentifier())
                # what if portname is actually a pin.full_name
                if portname in module.pins:
                    pin = module.pins[portname]
                    nodes_with_edges.add(pin.instance.name)

        # Step 2: mark nodes for no_layout
        all_nodes = list(graph.getChildren())
        for node in all_nodes:
            if node.getIdentifier() not in nodes_with_edges:
                # node.setProperty(CoreOptions.NO_LAYOUT, True)
                node.setParent(None)

    def create_ranks_from_elk_graph(self, module, graph):
        """
        Group nodes by their set of connected nets (net signature).
        Nodes must share the same original nets to be in the same group.
        """
        # Map: net_id -> set of nodes connected to that net
        net_to_nodes = defaultdict(set)

        # Map: node -> set of net_ids it connects to
        node_to_nets = defaultdict(set)

        # Build the mappings
        for edge in graph.getContainedEdges():
            netname = self.id2name(edge.getIdentifier())
            net = module.nets[netname]
            if net.is_buffered_net:
                netname = net.buffer_original_netname
            for node in edge.getSources() + edge.getTargets():
                net_to_nodes[netname].add(node)
                node_to_nets[node].add(netname)

        # Group nodes by their net signature (exact set of nets)
        signature_to_nodes = defaultdict(list)

        nodes = graph.getChildren()  # Get all nodes in the graph
        for node in nodes:
            # Use frozenset as key (immutable, order-independent)
            net_signature = frozenset(node_to_nets[node])
            signature_to_nodes[net_signature].append(node)

        # Convert to list of groups
        node_groups = list(signature_to_nodes.values())

        # Get positions of nodes in each group for PCA
        group_positions = []
        for group in node_groups:
            x_coords = []
            y_coords = []
            for node in group:
                x = node.getX() if hasattr(node, "getX") else 0
                y = node.getY() if hasattr(node, "getY") else 0
                x_coords.append(x)
                y_coords.append(y)

            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            group_positions.append([centroid_x, centroid_y])

        # Use PCA to determine ranks
        if len(group_positions) > 1:
            positions_array = np.array(group_positions)
            pca = PCA(n_components=1)
            pca_coords = pca.fit_transform(positions_array)

            # Sort groups by PCA coordinate
            sorted_indices = np.argsort(pca_coords[:, 0])
            rank_mapping = {idx: rank for rank, idx in enumerate(sorted_indices)}

            for group_idx, group in enumerate(node_groups):
                rank = rank_mapping[group_idx]
                for node in group:
                    node.setProperty(LayeredOptions.LAYERING_LAYER_ID, rank)
                    inst = module.instances[node.getIdentifier()]
                    inst.draw.rank = rank
                    log.info(f"Node {node.getIdentifier()} has rank {rank}")
        else:
            for node in node_groups[0]:
                node.setProperty(LayeredOptions.LAYERING_LAYER_ID, 0)

        # Debug output
        print(f"\nFound {len(node_groups)} groups:")
        for i, (signature, group) in enumerate(signature_to_nodes.items()):
            node_ids = [node.getIdentifier() for node in group]
            print(f"Group {i}: {node_ids}")
            print(f"  Connected nets: {sorted(signature)}")

        return node_groups, signature_to_nodes, net_to_nodes, node_to_nets

    def evaluate_layout(self, module):
        """
        Evaluate the geometric quality of an ELK-generated layout.

        Each net must have `net.draw.efig` as a list of (start, end) tuples: [((x1, y1), (x2, y2)), ...]

        Returns a dictionary with:
            - total_length
            - mean_length
            - num_segments
            - num_crossings
        """
        segments = []
        nets = []
        disconnected_nets = 0

        # Convert each net's drawn segments to shapely LineStrings
        for net_name, net in module.nets.items():
            if not hasattr(net, "draw") or not getattr(net.draw, "fig", None):
                disconnected_nets += 1
                continue

            for start, end in net.draw.efig:
                # Ignore degenerate zero-length segments
                if start == end:
                    continue
                try:
                    line = LineString([start, end])
                    segments.append(line)
                    nets.append(net_name)
                except Exception as e:
                    log.warning(f"Invalid line for net {net_name}: {start}->{end}, {e}")

        if not segments:
            return {
                "total_length": 0.0,
                "mean_length": 0.0,
                "num_segments": 0,
                "num_crossings": 0,
                "disconnected_nets": disconnected_nets,
            }

        # --- Total wire length ---
        total_length = sum(seg.length for seg in segments)
        mean_length = total_length / len(segments)

        # --- Build spatial index for fast crossing detection ---
        tree = STRtree(segments)
        crossings = 0

        for i, seg in enumerate(segments):
            for j in tree.query(seg):
                if j <= i:
                    continue
                other = segments[j]
                if seg.crosses(other):
                    crossings += 1

        # Each crossing counted twice (A→B, B→A)
        num_crossings = crossings // 2

        result = {
            "total_length": total_length,
            "mean_length": mean_length,
            "num_segments": len(segments),
            "num_crossings": num_crossings,
            "disconnected_nets": disconnected_nets,
        }

        log.info(
            f"Layout eval: {result['num_segments']} segments, "
            f"{result['num_crossings']} crossings, "
            f"total length={result['total_length']:.2f}"
        )

        return result

    # File I/O and debugging
    def write_graph_to_file(self, graph, prefix: str) -> None:
        """
        Write ELK graph to files in JSON ELKT DOT formats.
        """
        output_dir = Path("data")
        prefix = f"{prefix}_{graph.getIdentifier()}"

        # --- JSON ---
        json_path = output_dir / "json" / f"{prefix}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        self.elk_utils.write_json_file(graph, json_path)

        elkt_path = output_dir / "elkt" / f"{prefix}.elkt"
        elkt_path.parent.mkdir(parents=True, exist_ok=True)
        self.elk_utils.write_elkt_file(graph, elkt_path)

        dot_path = output_dir / "dot" / f"{prefix}.dot"
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        self.elk_utils.write_dot_file(graph, dot_path)

    def dump_layered_options(self, graph):
        log.info(f"{'Option':50} {'Current':25} {'Default':25} {'Changed':8}")
        log.info("=" * 110)

        for name in dir(LayeredOptions):
            # Skip private names and non-properties
            if name.startswith("_"):
                continue

            try:
                prop = getattr(LayeredOptions, name)
            except Exception:
                continue

            # Check if it's an actual IProperty<?> instance
            if not isinstance(prop, IProperty):
                continue

            if not name.isupper():
                continue

            # Retrieve current and default values
            try:
                current_val = graph.getProperty(prop)
                default_val = prop.getDefault()
                changed = current_val is not None and current_val != default_val
                log.info(
                    f"{name:50} {str(current_val or '(unset)'):25} {str(default_val or '(none)'):25} {'YES' if changed else 'NO':8}"
                )
            except Exception:
                continue

    def dump_elk_config_table(self, obj, filename: str = "data/csv/elk_settings.csv"):
        """
        Dump ELK layout option values (current + default) for a given ELK Options class
        into a CSV file, merging with metadata from elk_options.json.
        """
        # Locate JSON metadata file
        PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
        json_file = PACKAGE_ROOT / "resources" / "elk_options.json"

        if not json_file.exists():
            raise FileNotFoundError(f"Missing {json_file}")

        options = json.loads(json_file.read_text())

        # Build lookup for quick metadata access
        json_lookup = {opt["option"]: opt for opt in options}

        # CSV output path
        csv_path = Path(filename)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "Type",
            "Option",
            "Object",
            "Current",
            "Default",
            "Options",
            "Group",
            "Targets",
            "Description",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Iterate through all uppercase properties of the given ELK Options class
            for option in (LayeredOptions, CoreOptions):
                match = re.search(r"\.(\w+)\'>$", str(option))
                if match:
                    option_name = match.group(1)
                else:
                    option_name = str(option)

                for name in dir(option):
                    if not name.isupper():
                        continue

                    try:
                        prop = getattr(option, name)
                    except Exception:
                        continue

                    if not isinstance(prop, IProperty):
                        continue

                    try:
                        current_val = obj.getProperty(prop).toString()
                        default_val = prop.getDefault().toString()
                    except Exception:
                        current_val = None
                        default_val = None

                    try:
                        val_options = [val.toString() for val in obj.getProperty(prop).values()]
                    except Exception:
                        val_options = None

                    # Metadata enrichment
                    meta = json_lookup.get(name, {})
                    obj_name = obj.toString()
                    group = meta.get("group", "")
                    targets = meta.get("targets", "")
                    desc = meta.get("description", "")

                    # Short form ID (tail of getId().toString())
                    try:
                        opt_id = prop.getId().toString().split(".")[-1]
                    except Exception:
                        opt_id = name

                    # Write CSV row
                    writer.writerow(
                        {
                            "Type": option_name,
                            "Option": opt_id,
                            "Object": obj_name,
                            "Current": str(current_val or ""),
                            "Default": str(default_val or ""),
                            "Options": str(val_options or ""),
                            "Group": group,
                            "Targets": targets,
                            "Description": desc,
                        }
                    )

        log.info(f"Dumped ELK option data for {obj} → {csv_path}")

    # Symbol and port handling
    def update_graph_with_symbols(self, module, graph):
        for node in graph.getChildren():
            node_id = node.getIdentifier()
            ref_module = module.instances[node_id].module
            if ref_module.name in self.symbol_outlines:
                node.eUnset(node.eClass().getEStructuralFeature("x"))
                node.eUnset(node.eClass().getEStructuralFeature("y"))
                self.update_node_with_symbol(node, module, self.symbol_outlines[ref_module.name])
                node.setProperty(CoreOptions.PORT_CONSTRAINTS, PortConstraints.FIXED_POS)

        for edge in graph.getContainedEdges():
            edge.getSections().clear()
            edge.getProperties().clear()

        for node in graph.getChildren():
            for port in node.getPorts():
                continue
                if not port.getProperty(self.keep_port_loc_property):
                    # Unset X and Y
                    port.setProperty(CoreOptions.DIRECTION, None)
                    port.setProperty(CoreOptions.POSITION, None)
                    port.eUnset(port.eClass().getEStructuralFeature("x"))
                    port.eUnset(port.eClass().getEStructuralFeature("y"))

    def update_node_with_symbol(self, node, module, symbol):
        old_pins_loc = {}
        (hw, hh) = (node.getWidth() / 2, node.getHeight() / 2)
        for port in node.getPorts():
            port_id = port.getIdentifier()
            pinname = module.pins[port_id].name
            old_pins_loc[pinname] = Point(port.getX() - hw, port.getY() - hh)

        r0_pins_loc = {}
        for name, port in symbol.ports.items():
            r0_pins_loc[name] = port.fig

        old_pins_loc = self.center_pins(old_pins_loc)

        new_pins_loc, new_orientation = self.get_best_orientation(old_pins_loc, r0_pins_loc)
        (h, w) = self.update_symbol_size_for_orientation(symbol, new_orientation)
        node.setWidth(w)
        node.setHeight(h)
        (hw, hh) = (node.getWidth() / 2, node.getHeight() / 2)
        for port in node.getPorts():
            port_id = port.getIdentifier()
            pinname = module.pins[port_id].name
            loc = new_pins_loc[pinname]
            port.setX(hw + loc.x)
            port.setY(hh + loc.y)
            side = self.get_port_side(hw, hh, loc)
            port.setProperty(CoreOptions.PORT_SIDE, side)
            # port.setProperty(self.keep_port_loc_property, True)

    def get_port_side(self, hw, hh, loc):
        """Determine port side based on position relative to center."""
        # Determine port side
        rel_x = loc.x
        rel_y = loc.y
        if abs(rel_y - hh) < abs(rel_x - hw):
            # More horizontal displacement → EAST/WEST
            side = PortSide.EAST if rel_x > 0 else PortSide.WEST
        else:
            # More vertical displacement → NORTH/SOUTH
            side = PortSide.SOUTH if rel_y > 0 else PortSide.NORTH
        return side

    def center_pins(self, pins_loc: dict) -> dict:
        """
        Recenters pins so their geometric midpoint is at (0, 0).

        Args:
            pins_loc (dict[str, Point]): Original pin locations.

        Returns:
            dict[str, Point]: New pin locations centered around the midpoint.
        """
        # Compute the centroid (midpoint) between all pin coordinates
        xs = [p.x for p in pins_loc.values()]
        ys = [p.y for p in pins_loc.values()]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        # Shift all pins so that (cx, cy) moves to (0, 0)
        centered = {name: Point(p.x - cx, p.y - cy) for name, p in pins_loc.items()}

        return centered

    def update_symbol_size_for_orientation(self, symbol, orientation: str):
        """Return new (width, height) for a given orientation."""
        w, h = symbol.width, symbol.height
        orientation = orientation.upper()

        if orientation in ("R90", "R270", "M90", "M270"):
            return w, h  # swapped
        else:
            return h, w  # unchanged

    def transform_point(self, pt: Point, orientation: str) -> Point:
        """Apply orientation transform to a point (mirror + rotate)."""
        settings = ORIENTATIONS[orientation]
        g = pt
        if settings["mirror_x"]:
            g = scale(g, xfact=-1, yfact=1, origin=(0, 0))
        if settings["mirror_y"]:
            g = scale(g, xfact=1, yfact=-1, origin=(0, 0))
        if settings["rotate"] != 0:
            g = rotate(g, settings["rotate"], origin=(0, 0))
        return g

    def get_best_orientation(self, old_pins_loc: dict, r0_pins_loc: dict):
        """
        Rotate and mirror the reference pin set (r0_pins_loc)
        to find the orientation minimizing total pin displacement.
        """
        best_orientation = None
        best_distance = float("inf")
        best_pins_loc = None

        for orient in ORIENTATIONS.keys():
            # Apply the transform to each r0 pin
            new_pins = {name: self.transform_point(pt, orient) for name, pt in r0_pins_loc.items()}

            # Compute total distance from old pin positions
            total_dist = 0.0
            for name, old_pt in old_pins_loc.items():
                if name in new_pins:
                    total_dist += old_pt.distance(new_pins[name])
                else:
                    # if pin missing, penalize heavily
                    total_dist += 1e6
                log.info(f"{name=}, {old_pt=}, {new_pins[name]=}, {total_dist=}")
            if total_dist < best_distance:
                best_distance = total_dist
                best_orientation = orient
                best_pins_loc = new_pins

        return best_pins_loc, best_orientation

    # Utility functions
    def genid(self, obj, counter: int | None = None):
        h = hash(obj)
        if counter is not None:
            return f"id{h}_{counter}"
        return f"id{h}"

    def id2hash(self, id_str: str) -> int:
        match = re.match(r"id(\d+)(?:_\d+)?$", str(id_str))
        if not match:
            raise ValueError(f"Invalid ID format: {id_str}")
        return match.group(1)

    def id2name(self, id_str: str) -> str:
        match = re.match(r"(\S+)(_idx\d+)$", str(id_str))
        if not match:
            raise ValueError(f"Invalid ID format: {id_str}")
        return match.group(1)

    def get_node_size(self, inst):
        self.size_node_base_w = 6
        self.size_node_base_h = 18
        self.size_factor_per_pin = 2  #  ie size_min_macro + size_factor_per_pin * degree

        # get from a previous run...
        if not inst.module.is_leaf:
            return inst.module.draw.efig

        if inst.is_buffer:
            return 3, 9
        degree = len(inst.pins.keys())
        if degree <= 3:
            return self.size_node_base_w, self.size_node_base_h
        extra = self.size_factor_per_pin * degree
        w = self.size_node_base_w + extra
        h = self.size_node_base_h + extra
        w = round(w, 1)
        h = round(h, 1)
        return w, h
