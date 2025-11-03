import json
import logging as log
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pygraphviz as pgv


class Graphviz:
    def __init__(self, db):
        self.db = db
        self.output_dir = "data/json"
        self.phase = "initial"
        self.flat = False

    def generate_layout_figures(self, phase: str = "initial", flat: bool = False):
        self.phase = phase
        self.flat = flat
        if flat:
            self.generate_module_layout(self.db.design.flat_module)
        else:
            sorted_modules = sorted(self.db.design.modules.values(), key=lambda m: m.depth, reverse=True)
            for module in sorted_modules:
                if not module.is_leaf:
                    self.generate_module_layout(module)

    def generate_module_layout(self, module):
        log.info(f"Generating layout for module {module.name} phase {self.phase}")
        os.makedirs("data/png", exist_ok=True)

        A = pgv.AGraph(directed=True, strict=False, ratio="auto")
        self.set_attributes(A)
        self.add_nodes(A, module)
        self.add_edges(A, module)
        if self.flat:
            self.add_flat_cluster(A, module)
        else:
            self.add_ranks(A, module)
        A = self.run_graphviz(A, module)
        self.extract_geometry(A, module)

    def add_edges(self, A, module):
        # Add nodes for instances in the cluster, tagging them if they are buffers
        add_stubs = False
        # Add edges for internal nets
        for net in module.nets.values():
            # if net.name.startswith("GND") or net.name.startswith("_3V3") or net.name.startswith("VBUS"):
            #    continue
            if net.is_chained_net:
                weight = 0
            else:
                weight = 1
            pins = list(net.pins.values())
            if len(pins) > 1:
                src = pins[0]
                for dst in pins[1:]:
                    A.add_edge(
                        src.instance.name,
                        dst.instance.name,
                        xlabel=net.name,
                        headlabel=dst.full_name,
                        taillabel=src.full_name,
                        weight=weight,
                    )
            elif len(pins) == 1:
                if add_stubs:
                    pin = pins[0]
                    stub_name = f"stub_{pin.full_name}".replace("/", "_")
                    A.add_node(stub_name, shape="point", width=0.01, height=0.01)
                    A.add_edge(
                        pin.instance.name,
                        stub_name,
                        xlabel=net.name,
                        headlabel=pin.full_name,
                        taillabel=pin.full_name,
                        weight=weight,
                    )

    def add_nodes(self, A, module):
        # --- Node size scaling based on graph degree ---
        size_min_macro = 0.5
        size_buffer = 0.2
        size_factor_per_pin = 0.2  #  ie 200-pin macro will be 20x20

        def scale_macro_size(degree):
            if degree <= 3:
                return size_min_macro
            size_node = size_min_macro * size_factor_per_pin * degree
            return round(size_node, 1)

        for inst in module.instances.values():
            attr = {}
            attr["fixedsize"] = True

            is_leaf = not inst.module or inst.module.is_leaf

            if not is_leaf:
                fig = inst.module.draw.fig
                attr["shape"] = "box"
                attr["width"] = round(fig[0] / 72, 1)
                attr["height"] = round(fig[1] / 72, 1)

            elif inst.is_buffer:
                attr["shape"] = "circle"
                attr["width"] = size_buffer
                attr["height"] = size_buffer
                attr["xlabel"] = inst.name
            else:
                attr["shape"] = "box"
                degree = len(inst.pins)
                size_node = scale_macro_size(degree)
                attr["width"] = size_node
                attr["height"] = size_node

            if self.phase != "initial":
                if fig := inst.draw.fig:
                    # attr["pos"] = f"{fig[0] / 72},{fig[1] / 72}"
                    attr["pos"] = f"{fig[0]},{fig[1]}"
                else:
                    log.error(f"No figure for instance {inst.name}")

            A.add_node(inst.name, **attr)

    def add_ranks(self, A, module):
        # --- ranks based on similarity ---
        # if there are module instances with same ref name and similar name, put them in the same rank

        # --- Step 1: group instances by ref_name ---
        ref_groups = defaultdict(list)
        for name, inst in module.instances.items():
            if not inst.is_buffer:
                ref_groups[inst.module_ref].append(name)

        # --- Step 2: for each ref_name group, find common prefixes ---
        def common_prefix(a, b):
            """Return common prefix string of a and b."""
            prefix = os.path.commonprefix([a, b])
            return prefix

        log.debug(f"ref_groups: {ref_groups=}")
        for ref_name, inst_names in ref_groups.items():
            if len(inst_names) < 2:
                continue  # nothing to group

            # Sort to stabilize grouping
            inst_names.sort()

            # Find all prefixes among instances
            prefix_groups = defaultdict(list)
            for i, name1 in enumerate(inst_names):
                for name2 in inst_names[i + 1 :]:
                    prefix = common_prefix(name1, name2)
                    log.debug(f"ref_name={ref_name}, name1={name1}, name2={name2}, prefix={prefix}")
                    if len(prefix) > 0 and prefix not in ("R", "C", "L", "D"):
                        prefix_groups[prefix].append(name1)
                        prefix_groups[prefix].append(name2)

            log.debug(f"{prefix_groups=}")
            # --- Step 4: create rank groups ---
            for prefix, names in prefix_groups.items():
                unique_names = sorted(set(names))
                if len(unique_names) < 2:
                    continue

                sg_name = f"cluster_{prefix}"

                A.add_subgraph(nbunch=unique_names, rank="same", name=sg_name)
                log.debug(f"rank group: ref_name={ref_name}, prefix={prefix}, size={len(unique_names)}")

    def add_flat_cluster(self, A, module):
        """add clusters for flat designs"""
        hier_groups = defaultdict(list)
        for name, inst in module.instances.items():
            if inst.is_buffer:
                continue
            module_name = inst.hier_module.name
            prefix = inst.hier_prefix
            subgraph_name = f"{prefix}_{module_name}"
            hier_groups[subgraph_name].append(name)

        for hier, names in hier_groups.items():
            unique_names = sorted(set(names))
            if len(unique_names) < 2:
                continue
            sg_name = f"cluster_{hier}"
            A.add_subgraph(nbunch=unique_names, rank="same", name=sg_name)
            log.debug(f"hier group: hier={hier}, size={len(unique_names)}")

    def run_graphviz(self, A, module):
        # Layout and extract size
        os.makedirs("data/dot", exist_ok=True)
        os.makedirs("data/png", exist_ok=True)
        predot = f"data/dot/pre_{self.phase}_{module.name}.dot"
        postdot = f"data/dot/post_{self.phase}_{module.name}.dot"
        postpng = f"data/png/post_{self.phase}_{module.name}.png"
        A.write(predot)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            log.info(f"Running graphviz for module {module.name}")
            if self.phase == "initial":
                A.layout(prog="dot", args="-vy")
                A.write(postdot)
            else:
                # assume placed dot
                os.system(f"neato -y -n2 -Tdot -o {postdot} {predot}")
                A = pgv.AGraph(postdot, strict=False)

            log.info(f"Ran graphviz for module {module.name} with result in {postdot}")
        with warnings.catch_warnings():
            # warnings.filterwarnings("ignore", category=RuntimeWarning)
            # A.draw(f"data/png/post_{module.name}.png", format="png")
            os.system(f"neato -y -n2 -Tpng -o {postpng} {postdot}")
        return A

    def set_attributes(self, A):
        """Sets graph, node, and edge attributes."""
        A.graph_attr.update(
            K="0.2",
            maxiter="50000",
            mclimit="9999",  # allow more time for mincross optimization
            nslimit="9999",  # allow more time for network simplex
            nslimit1="9999",  # same for phase 3 placement
            overlap="vpsc",
            pack="true",
            packmode="graph",
            # sep="+2,2",
            sep="+20,20",
            esep="+2,2",
            nodesep="0.5,0.5",  # minimum space between two adjacent nodes in the same rank,
            epsilon="1e-7",
            rankdir="LR",
            start="rand",  # better escape from local minima
            mode="hier",  # hierarchical bias
            ratio="auto",
            splines="ortho",
        )

        A.node_attr.update(fillcolor="white", margin="0,0", width="0.4", height="0.3", fixedsize="true", label="")

        A.edge_attr.update(len="0.3", weight="1.5")
        A.edge_attr.update(
            {
                "fontsize": "2",
                "arrowhead": "dot",
                "arrowtail": "dot",
                "arrowsize": "0.0",
                "dir": "both",
            }
        )

    def extract_geometry(self, A, module):
        """Extracts geometry from the graph."""

        log.info("Extracting geometry...")
        if "bb" in A.graph_attr.keys():
            bb = A.graph_attr["bb"]
            xmin, ymin, xmax, ymax = map(float, bb.split(","))
            module.draw.fig = (abs(xmax - xmin), abs(ymax - ymin))
        else:
            log.warning(f" No bounding box for module {module.name}")

        for node in A.nodes():
            if node.name.startswith("stub_") or node.name.startswith("cluster_"):
                continue
            pos = node.attr.get("pos")
            width = node.attr.get("width")
            height = node.attr.get("height")
            if pos and width and height:
                x, y = map(float, pos.split(","))
                w = float(width) * 72.0
                h = float(height) * 72.0
                xmin = x - w / 2
                ymin = y - h / 2
                xmax = x + w / 2
                ymax = y + h / 2
                module.instances[node.name].draw.fig = (xmin, ymin, xmax, ymax)

        def parse_edge_pin_positions(edge):
            pos_string = edge.attr.get("pos")
            if not pos_string:
                return None, None, []

            points = pos_string.split()
            tail_coord, head_coord = None, None
            wire_points = []

            # Extract start and end points, and wire points
            for p_str in points:
                if p_str.startswith("s,"):
                    coords = p_str.split(",")[1:]
                    tail_coord = (float(coords[0]), float(coords[1]))
                elif p_str.startswith("e,"):
                    coords = p_str.split(",")[1:]
                    head_coord = (float(coords[0]), float(coords[1]))
                else:
                    coords = p_str.split(",")
                    wire_points.append((float(coords[0]), float(coords[1])))

            segments = []
            if len(wire_points) > 1:
                pt_start = wire_points[0]
                for pt_end in wire_points[1:]:
                    segments.append((pt_start, pt_end))
                    pt_start = pt_end

            return tail_coord, head_coord, segments

        for edge in A.edges():
            tail_coord, head_coord, wire_segments = parse_edge_pin_positions(edge)
            tail_pin = edge.attr.get("taillabel")
            head_pin = edge.attr.get("headlabel")
            net_name = edge.attr.get("xlabel")
            # order is important because for open nets head and tail labels are the same, and we need to overwrite
            if head_pin and not head_coord:
                breakpoint()
            # with tail
            if head_pin and head_coord:
                pin = module.pins.get(head_pin)
                pin.draw.fig = head_coord
            if tail_pin and tail_coord:
                pin = module.pins.get(tail_pin)
                pin.draw.fig = tail_coord
            if net_name:
                module.nets[net_name].draw.fig.extend(wire_segments)

        log.info("Geometry extracted")
