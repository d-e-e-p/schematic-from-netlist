import csv
import json
import logging as log
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jpype
import jpype.imports
from jpype import JClass, JDouble, JInt
from shapely.affinity import rotate, scale, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree
from tabulate import tabulate

from schematic_from_netlist.interfaces.elk_utils import ElkUtils
from schematic_from_netlist.interfaces.symbol_library import SymbolLibrary
from schematic_from_netlist.utils.config import setup_elk

setup_elk()  # Initialize ELK JVM

# ruff: noqa: E402
# pyright: reportMissingImports=false

from java.lang import Integer
from org.eclipse.elk.alg.layered.options import (
    CrossingMinimizationStrategy,
    FixedAlignment,
    LayeredOptions,
    LayeringStrategy,
    NodePlacementStrategy,
    SelfLoopDistributionStrategy,
    WrappingStrategy,
)

# ELK Core Engine
from org.eclipse.elk.core import RecursiveGraphLayoutEngine

# ELK Option classes for layout configuration
from org.eclipse.elk.core.options import CoreOptions, Direction, EdgeRouting, PortAlignment, PortConstraints, PortSide
from org.eclipse.elk.core.util import BasicProgressMonitor
from org.eclipse.elk.graph import ElkEdge, ElkNode, ElkPort

# ELK IO Utility
from org.eclipse.elk.graph.json import ElkGraphJson
from org.eclipse.elk.graph.properties import IProperty, Property
from org.eclipse.elk.graph.text import ElkGraphStandaloneSetup, ElkGraphTextUtil

# ELK Model classes for programmatic graph construction
from org.eclipse.elk.graph.util import ElkGraphUtil
from org.eclipse.xtext.serializer import ISerializer

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

    def generate_layout_figures(self):
        self.db._build_lookup_tables()
        sorted_modules = sorted(self.db.design.modules.values(), key=lambda m: m.depth, reverse=True)
        for module in sorted_modules:
            if not module.is_leaf:
                self.generate_module_layout(module)

    def generate_module_layout(self, module):
        log.info(f"Generating layout for module {module.name}")

        graph = ElkGraphUtil.createGraph()
        graph.setIdentifier(module.name)

        nodes, ports = self.create_nodes(module, graph)
        self.create_edges(module, graph, ports)

        self.layout_graph(graph, "pass1")
        self.dump_elk_config_table(graph)
        # self.update_graph_with_symbols(module, graph)
        # self.layout_graph(graph, "pass2")
        self.extract_geometry(graph, module)
        self.evaluate_layout(module)

    def get_node_size(self, inst):
        self.size_node_base_w = 6
        self.size_node_base_h = 18
        self.size_factor_per_pin = 2  #  ie size_min_macro + size_factor_per_pin * degree
        if inst.is_buffer:
            return 1, 1
        degree = len(inst.pins.keys())
        if degree <= 3:
            return self.size_node_base_w, self.size_node_base_h
        extra = self.size_factor_per_pin * degree
        w = self.size_node_base_w + extra
        h = self.size_node_base_h + extra
        w = round(w, 1)
        h = round(h, 1)
        return w, h

    def create_nodes(self, module, graph):
        nodes = {}
        ports = {}
        port_size = 0
        for inst in module.instances.values():
            node = ElkGraphUtil.createNode(graph)
            node.setIdentifier(inst.name)
            width, height = self.get_node_size(inst)
            node.setWidth(width)
            node.setHeight(height)
            nodes[inst.name] = node
            for pin in inst.pins.values():
                port = ElkGraphUtil.createPort(node)
                port.setIdentifier(pin.full_name)
                port.setWidth(port_size)
                port.setHeight(port_size)
                ports[pin.full_name] = port
        return nodes, ports

    def create_edges(self, module, graph, ports):
        edges = []
        # TODO: deal with pin directions...
        for net in module.nets.values():
            net_index = 0
            pins = list(net.pins.values())
            if len(pins) > 1:
                # sort by fanout of driver
                sorted_pins = sorted(pins, key=lambda pin: len(pin.instance.pins.values()), reverse=True)
                src = sorted_pins[0]
                src_port = ports[src.full_name]
                for dst in pins[1:]:
                    dst_port = ports[dst.full_name]

                    net_id = f"{net.name}_idx{net_index}"
                    net_index += 1
                    edge = ElkGraphUtil.createEdge(graph)
                    edge.setProperty(CoreOptions.EDGE_THICKNESS, 0.0)
                    edge.setProperty(CoreOptions.DIRECTION, Direction.UNDEFINED)
                    edge.setIdentifier(net_id)
                    edge.getSources().add(src_port)
                    edge.getTargets().add(dst_port)

                    """
                    net_id = f"{net.name}_idx{net_index}"
                    net_index += 1
                    edge = ElkGraphUtil.createEdge(graph)
                    edge.setProperty(CoreOptions.EDGE_THICKNESS, 0.0)
                    edge.setProperty(CoreOptions.DIRECTION, Direction.UNDEFINED)
                    edge.setIdentifier(net_id)
                    edge.getSources().add(dst_port)
                    edge.getTargets().add(src_port)
                    """

    def layout_graph(self, graph, step: str = ""):
        graph.setProperty(CoreOptions.ALGORITHM, "layered")
        graph.setProperty(CoreOptions.ASPECT_RATIO, 2.0)
        graph.setProperty(LayeredOptions.EDGE_ROUTING, EdgeRouting.ORTHOGONAL)
        graph.setProperty(LayeredOptions.PORT_CONSTRAINTS, PortConstraints.FREE)
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

        # Force ELK to ignore old coordinates
        self.write_graph_to_file(graph, f"pre_{step}")

        layout_engine = RecursiveGraphLayoutEngine()
        monitor = BasicProgressMonitor()
        layout_engine.layout(graph, monitor)
        self.write_graph_to_file(graph, f"post_{step}")

    def extract_geometry(self, graph, module):
        def walk_graph_and_extract_data(node, module):
            is_root = node.getParent() is None
            if is_root:
                module.draw.fig = (node.getWidth(), node.getHeight())
            else:
                node_id = node.getIdentifier()
                instance = module.instances[node_id]
                instance.draw.fig = (
                    node.getX(),
                    node.getY(),
                    node.getX() + node.getWidth(),
                    node.getY() + node.getHeight(),
                )
                for port in node.getPorts():
                    port_id = port.getIdentifier()
                    pin = module.pins[port_id]
                    pin.draw.fig = (
                        port.getX(),
                        port.getY(),
                        # port.getX() + port.getWidth(),
                        # port.getY() + port.getHeight(),
                    )

            for edge in node.getContainedEdges():
                # src_pinname = edge.getSources().get(JInt(0)).getIdentifier()
                # dst_pinname = edge.getTargets().get(JInt(0)).getIdentifier()
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
                        net.draw.fig.append((start, end))
                # log.info(f"Net {net.name} routing: {net.draw.fig=} ")

            for child in node.getChildren():
                walk_graph_and_extract_data(child, module)

        walk_graph_and_extract_data(graph, module)

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

    def evaluate_layout(self, module):
        """
        Evaluate the geometric quality of an ELK-generated layout.

        Each net must have `net.draw.fig` as a list of (start, end) tuples: [((x1, y1), (x2, y2)), ...]

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

            for start, end in net.draw.fig:
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
