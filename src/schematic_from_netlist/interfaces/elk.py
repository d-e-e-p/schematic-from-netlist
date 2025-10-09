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
from shapely.geometry import Point, Polygon, box

from schematic_from_netlist.interfaces.elk_utils import ElkUtils
from schematic_from_netlist.interfaces.symbol_library import SymbolLibrary
from schematic_from_netlist.utils.config import setup_elk

setup_elk()

# ruff: noqa: E402
# pyright: reportMissingImports=false
from com.google.gson import JsonObject, JsonParser
from org.eclipse.elk.alg.layered.options import (
    CrossingMinimizationStrategy,
    FixedAlignment,
    LayeredOptions,
    LayeringStrategy,
    NodePlacementStrategy,
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
        self.size_min_macro = 6
        self.size_factor_per_pin = 1  #  ie size_min_macro + size_factor_per_pin * degree

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
        os.makedirs("data/png", exist_ok=True)

        graph = ElkGraphUtil.createGraph()
        graph.setIdentifier(module.name)

        nodes, ports = self.create_nodes(module, graph)
        self.create_edges(module, graph, ports)

        self.layout_graph(graph, "pass1")
        # self.update_graph_with_symbols(module, graph)
        # self.layout_graph(graph, "pass2")
        self.extract_geometry(graph, module)

    def scale_macro_size(self, inst):
        degree = len(inst.pins.keys())
        if degree <= 3:
            return self.size_min_macro
        size_node = self.size_min_macro * self.size_factor_per_pin * degree
        return round(size_node, 1)

    def create_nodes(self, module, graph):
        nodes = {}
        ports = {}
        port_size = 0
        for inst in module.instances.values():
            node = ElkGraphUtil.createNode(graph)
            node.setIdentifier(inst.name)
            size = self.scale_macro_size(inst)
            node.setWidth(6)
            node.setHeight(18)
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
        graph.setProperty(LayeredOptions.EDGE_ROUTING, EdgeRouting.ORTHOGONAL)
        graph.setProperty(CoreOptions.PORT_CONSTRAINTS, PortConstraints.FREE)
        graph.setProperty(LayeredOptions.PORT_CONSTRAINTS, PortConstraints.FREE)
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
        try:
            export_builder = ElkGraphJson.forGraph(graph)

            # Configure options:
            export_builder.prettyPrint(True)  #
            export_builder.omitLayout(False)  # →  removes "layoutOptions": {...}
            export_builder.omitZeroPositions(False)  # → omit nodes with (0,0)
            export_builder.omitZeroDimension(True)  # → omit width=0 or height=0
            export_builder.shortLayoutOptionKeys(True)  # → use short keys
            export_builder.omitUnknownLayoutOptions(True)  # → hide unknown options

            # Finally generate JSON
            json_str = str(export_builder.toJson())
            json_str = re.sub(r'^.*"resolvedAlgorithm".*\n?', "", json_str, flags=re.MULTILINE)

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            log.info(f"Wrote ELK graph JSON to: {json_path}")
        except Exception as e:
            log.exception(f"Failed to write JSON: {e}")

        # --- ELKT ---
        elkt_path = output_dir / "elkt" / f"{prefix}.elkt"
        elkt_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_elkt(graph, elkt_path)

        # --- DOT ---
        injector = GraphvizDotStandaloneSetup().createInjectorAndDoEMFRegistration()
        serializer = injector.getInstance(ISerializer)
        breakpoint()
        dot_str = serializer.serialize(graph)
        dot_path = output_dir / "dot" / f"{prefix}.dot"
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(dot_str)
        log.info(f"Wrote ELK graph DOT to: {dot_path}")

    def write_elkt(self, graph, elkt_path: Path) -> None:
        """Write ELK graph in ELKT format manually."""

        def serialize_node(node, indent=0):
            lines = []
            prefix = "  " * indent
            identifier = node.getIdentifier() or "unnamed"

            lines.append(f"{prefix}node {identifier} {{")

            # Add layout block with size and position on separate lines
            lines.append(f"{prefix}\tlayout [")
            if node.getWidth() > 0 or node.getHeight() > 0:
                lines.append(f"{prefix}\t\tposition: {node.getX()}, {node.getY()}")
                lines.append(f"{prefix}\t\tsize: {node.getWidth()}, {node.getHeight()}")
            lines.append(f"{prefix}\t]")
            lines.append(f'{prefix}\tnodeSize.constraints: "[]"')
            lines.append(f"{prefix}\tcrossingMinimization.positionId: -1")
            lines.append(f"{prefix}\tlayering.layerId: -1")

            # Ports
            for port in node.getPorts():
                port_id = port.getIdentifier() or "unnamed_port"
                lines.append(f"{prefix}\tport {port_id} {{")
                lines.append(f"{prefix}\t\tlayout [")
                lines.append(f"{prefix}\t\t\tposition: {port.getX()}, {port.getY()}")
                lines.append(f"{prefix}\t\t\tsize: {port.getWidth()}, {port.getHeight()}")
                lines.append(f"{prefix}\t\t]")
                port_side = port.getProperty(CoreOptions.PORT_SIDE)
                if port_side:
                    lines.append(f"{prefix}\t\torg.eclipse.elk.^port.side: {port_side}")
                lines.append(f"{prefix}\t}}")

            # Labels
            for label in node.getLabels():
                label_text = label.getText() or ""
                lines.append(f'{prefix}\tlabel "{label_text}"')

            # Child nodes
            for child in node.getChildren():
                lines.extend(serialize_node(child, indent + 1))

            lines.append(f"{prefix}}}")
            return lines

        def serialize_edges(node, indent=0):
            lines = []
            prefix = "  " * indent

            for edge in node.getContainedEdges():
                # Get source
                sources = list(edge.getSources())
                targets = list(edge.getTargets())

                if sources and targets:
                    src = sources[0]
                    tgt = targets[0]

                    # Get the parent node IDs
                    src_node = src.getParent()
                    tgt_node = tgt.getParent()

                    # Build identifiers in the format node_id.port_id
                    src_node_id = src_node.getIdentifier() if hasattr(src_node, "getIdentifier") else str(src_node)
                    src_port_id = src.getIdentifier() if hasattr(src, "getIdentifier") else str(src)
                    tgt_node_id = tgt_node.getIdentifier() if hasattr(tgt_node, "getIdentifier") else str(tgt_node)
                    tgt_port_id = tgt.getIdentifier() if hasattr(tgt, "getIdentifier") else str(tgt)

                    # Build full source and target identifiers
                    src_full_id = f"{src_node_id}.{src_port_id}"
                    tgt_full_id = f"{tgt_node_id}.{tgt_port_id}"

                    edge_id = edge.getIdentifier() or f"{src_node_id}_{src_port_id}_{tgt_node_id}_{tgt_port_id}"

                    # Use the edge identifier format from the expected file
                    lines.append(f"{prefix}edge {edge_id}: {src_full_id} -> {tgt_full_id} {{")

                    # Add edge sections - all in one layout clause
                    lines.append(f"{prefix}\tlayout [")
                    for i, section in enumerate(edge.getSections()):
                        lines.append(f"{prefix}\t\tsection s{i} [")
                        lines.append(f"{prefix}\t\t\tincoming: {src_full_id}")
                        lines.append(f"{prefix}\t\t\toutgoing: {tgt_full_id}")
                        lines.append(f"{prefix}\t\t\tstart: {section.getStartX()}, {section.getStartY()}")
                        lines.append(f"{prefix}\t\t\tend: {section.getEndX()}, {section.getEndY()}")

                        # Add bend points if any
                        bend_points = section.getBendPoints()
                        if bend_points:
                            bends = " | ".join(f"{bp.getX()}, {bp.getY()}" for bp in bend_points)
                            lines.append(f"{prefix}\t\t\tbends: {bends}")

                        lines.append(f"{prefix}\t\t]")
                    lines.append(f"{prefix}\t]")
                    # Add junctionPoints
                    bend_points = []
                    for section in edge.getSections():
                        bend_points.extend(section.getBendPoints())
                    if bend_points:
                        junction_points = "(" + " ; ".join(f"{bp.getX()},{bp.getY()}" for bp in bend_points) + ")"
                        lines.append(f'{prefix}\tjunctionPoints: "{junction_points}"')
                    else:
                        lines.append(f'{prefix}\tjunctionPoints: "()"')
                    lines.append(f"{prefix}}}")

            # Process edges in child nodes
            for child in node.getChildren():
                lines.extend(serialize_edges(child, indent))

            return lines

        # Build ELKT content
        lines = []
        graph_id = graph.getIdentifier() or "root"
        lines.append(f"graph {graph_id}")
        # Add graph-level layout properties
        lines.append("layout [ size: 116, 104 ]")
        lines.append('portLabels.placement: "[OUTSIDE]"')
        lines.append('nodeLabels.placement: "[]"')
        lines.append("algorithm: layered")
        lines.append('nodeSize.constraints: "[]"')
        lines.append("edgeRouting: ORTHOGONAL")
        lines.append('nodeSize.options: "[DEFAULT_MINIMUM_SIZE]"')
        lines.append("hierarchyHandling: SEPARATE_CHILDREN")

        # Add child nodes
        for child in graph.getChildren():
            lines.extend(serialize_node(child, 1))

        # Add edges
        lines.extend(serialize_edges(graph, 1))
        s = "\n".join(lines)
        s = re.sub(r"[/⊕]", "_", s)
        s = re.sub(r"_+", "_", s)

        # Write to file
        with open(elkt_path, "w", encoding="utf-8") as f:
            f.write(s)

        log.info(f"Wrote ELK graph ELKT to: {elkt_path}")

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
