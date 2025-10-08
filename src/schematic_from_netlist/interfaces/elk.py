import json
import logging as log
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jpype
import jpype.imports
from jpype import JClass, JDouble, JInt

from schematic_from_netlist.utils.config import setup_elk

log.basicConfig(level=log.DEBUG)
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
from org.eclipse.elk.core.options import CoreOptions, EdgeRouting, PortAlignment, PortConstraints
from org.eclipse.elk.core.util import BasicProgressMonitor
from org.eclipse.elk.graph import ElkEdge, ElkNode, ElkPort

# ELK IO Utility
from org.eclipse.elk.graph.json import ElkGraphJson
from org.eclipse.elk.graph.text import ElkGraphStandaloneSetup, ElkGraphTextUtil

# ELK Model classes for programmatic graph construction
from org.eclipse.elk.graph.util import ElkGraphUtil
from org.eclipse.xtext.serializer import ISerializer


class ElkInterface:
    def __init__(self, db):
        self.db = db
        self.output_dir = "data/json"
        self.size_min_macro = 30
        self.size_factor_per_pin = 2  #  ie 200-pin macro will be 200x200

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
        # self.update_graph_with_symbols(graph)
        # self.layout_graph(graph, "pass2")
        # self.extract_geometry(graph, module)
        exit()

    def scale_macro_size(self, inst):
        degree = len(inst.pins.keys())
        if degree <= 3:
            return self.size_min_macro
        size_node = self.size_min_macro * self.size_factor_per_pin * degree
        return round(size_node, 1)

    def create_nodes(self, module, graph):
        nodes = {}
        ports = {}
        port_size = 1
        for inst in module.instances.values():
            node = ElkGraphUtil.createNode(graph)
            node.setIdentifier(self.genid(inst))
            size = self.scale_macro_size(inst)
            node.setWidth(size)
            node.setHeight(size)
            nodes[inst.name] = node
            for pin in inst.pins.values():
                port = ElkGraphUtil.createPort(node)
                port.setIdentifier(self.genid(pin))
                port.setWidth(port_size)
                port.setHeight(port_size)
                ports[self.genid(pin)] = port
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
                src_hash = self.genid(src)
                src_port = ports[src_hash]
                for dst in pins[1:]:
                    dst_hash = self.genid(dst)
                    dst_port = ports[dst_hash]
                    net_id = self.genid(net, net_index)
                    net_index += 1
                    edge = ElkGraphUtil.createEdge(graph)
                    edge.setIdentifier(net_id)
                    edge.getSources().add(src_port)
                    edge.getTargets().add(dst_port)

    def layout_graph(self, graph, step: str = ""):
        graph.setProperty(CoreOptions.ALGORITHM, "layered")
        graph.setProperty(LayeredOptions.EDGE_ROUTING, EdgeRouting.ORTHOGONAL)
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
                instance = module.hash2instance[self.id2hash(node_id)]
                instance.draw.fig = (
                    node.getX(),
                    node.getY(),
                    node.getX() + node.getWidth(),
                    node.getY() + node.getHeight(),
                )
                for port in node.getPorts():
                    port_id = port.getIdentifier()
                    pin = module.hash2pin[self.id2hash(port_id)]
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
                if self.id2hash(net_id) not in module.hash2net:
                    breakpoint()
                net = module.hash2net[self.id2hash(net_id)]
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
        Write ELK graph to files in both JSON and ELKT formats.
        """
        output_dir = Path("data")
        prefix = f"{prefix}_{graph.getIdentifier()}"

        # --- JSON ---
        json_path = output_dir / "json" / f"{prefix}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            json_str = ElkGraphJson.forGraph(graph).toJson()
            json_data = json.loads(str(json_str))

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            log.info(f"Wrote ELK graph JSON to: {json_path}")
        except Exception as e:
            log.exception(f"Failed to write JSON: {e}")

        # --- ELKT ---
        elkt_path = output_dir / "elkt" / f"{prefix}.elkt"
        elkt_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_elkt(graph, elkt_path)

    def write_elkt(self, graph, elkt_path: Path) -> None:
        """Write ELK graph in ELKT format manually."""

        def serialize_node(node, indent=0):
            lines = []
            prefix = "  " * indent
            identifier = node.getIdentifier() or "unnamed"

            lines.append(f"{prefix}node {identifier} {{")
            
            # Combine size and position into a single layout clause
            layout_props = []
            if node.getWidth() > 0 or node.getHeight() > 0:
                layout_props.append(f"size: {node.getWidth()}, {node.getHeight()}")
            if node.getX() != 0 or node.getY() != 0:
                layout_props.append(f"position: {node.getX()}, {node.getY()}")
            
            if layout_props:
                lines.append(f"{prefix}  layout [ {', '.join(layout_props)} ]")

            # Ports
            for port in node.getPorts():
                port_id = port.getIdentifier() or "unnamed_port"
                lines.append(f"{prefix}  port {port_id} {{")
                
                # Combine port position and size into a single layout clause
                port_layout_props = []
                port_layout_props.append(f"position: {port.getX()}, {port.getY()}")
                port_layout_props.append(f"size: {port.getWidth()}, {port.getHeight()}")
                lines.append(f"{prefix}    layout [ {', '.join(port_layout_props)} ]")
                
                port_side = port.getProperty(CoreOptions.PORT_SIDE)
                if port_side:
                    lines.append(f"{prefix}    org.eclipse.elk.^port.side: {port_side}")
                lines.append(f"{prefix}  }}")

            # Labels
            for label in node.getLabels():
                label_text = label.getText() or ""
                lines.append(f'{prefix}  label "{label_text}"')

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

                    # Build edge identifier
                    src_id = src.getIdentifier() if hasattr(src, "getIdentifier") else str(src)
                    tgt_id = tgt.getIdentifier() if hasattr(tgt, "getIdentifier") else str(tgt)
                    edge_id = edge.getIdentifier() or f"{src_id}_{tgt_id}"

                    lines.append(f"{prefix}edge {src_id} -> {tgt_id} {{")
                    
                    # Add edge sections - all in one layout clause
                    for i, section in enumerate(edge.getSections()):
                        lines.append(f"{prefix}  layout [")
                        lines.append(f"{prefix}    section s{i} [")
                        lines.append(f"{prefix}      incoming: {src_id}")
                        lines.append(f"{prefix}      outgoing: {tgt_id}")
                        lines.append(f"{prefix}      start: {section.getStartX()}, {section.getStartY()}")
                        lines.append(f"{prefix}      end: {section.getEndX()}, {section.getEndY()}")
                        
                        # Add bend points if any
                        bend_points = section.getBendPoints()
                        if bend_points:
                            bends = " | ".join(f"{bp.getX()}, {bp.getY()}" for bp in bend_points)
                            lines.append(f"{prefix}      bends: {bends}")
                        
                        lines.append(f"{prefix}    ]")
                        lines.append(f"{prefix}  ]")
                    
                    lines.append(f"{prefix}}}")

            # Process edges in child nodes
            for child in node.getChildren():
                lines.extend(serialize_edges(child, indent))

            return lines

        # Build ELKT content
        lines = []
        graph_id = graph.getIdentifier() or "root"
        lines.append(f"graph {graph_id} {{")

        # Add child nodes
        for child in graph.getChildren():
            lines.extend(serialize_node(child, 1))

        # Add edges
        lines.extend(serialize_edges(graph, 1))

        lines.append("}")

        # Write to file
        with open(elkt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log.info(f"Wrote ELK graph ELKT to: {elkt_path}")

    def update_graph_with_symbols(self, graph):
        pass
