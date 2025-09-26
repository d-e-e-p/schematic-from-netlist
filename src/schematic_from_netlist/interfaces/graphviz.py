import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pygraphviz as pgv


@dataclass
class GeomDB:
    """A database for geometric primitives extracted from the graph layout."""

    ports: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    nets: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    def write_geom_db_report(self, filepath: str = "data/json/read_json.rpt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        out = "PORTS\n"
        for name in sorted(self.ports.keys()):
            port = self.ports[name]
            out += f" {name:20}: {port}\n"

        out += "NETS\n"
        for name in sorted(self.nets.keys()):
            pts = self.nets[name]
            out += f" {name:20}: {pts}\n"

        with open(filepath, "w") as f:
            f.write(out)


class ParseJson:
    def __init__(self, json_file):
        self.json_file = json_file
        with open(self.json_file, "r") as f:
            self.json_data = json.load(f)

    def parse(self) -> GeomDB:
        """Parses the JSON file and populates geomdb."""
        geom_db = GeomDB()

        for edge in self.json_data.get("edges", []):
            try:
                # Extract port rectangle from the head of the edge
                rect_data = edge["_hdraw_"][-1]["rect"]
                text_data = edge["headlabel"]
                x, y, _, _ = rect_data
                geom_db.ports[text_data] = (x, y)

                # Extract port rectangle from the tail of the edge
                rect_data = edge["_tdraw_"][-1]["rect"]
                text_data = edge["taillabel"]
                x, y, _, _ = rect_data
                geom_db.ports[text_data] = (x, y)

                # Extract wire points
                text_data = edge["label"]
                points_data = edge["_draw_"][-1]["points"]
                geom_db.nets[text_data] = points_data

            except (KeyError, IndexError, AttributeError, ValueError, TypeError):
                continue

        print(f"Parsed {len(geom_db.ports)} ports and {len(geom_db.nets)} nets from graph.")
        geom_db.write_geom_db_report()
        return geom_db


class Graphviz:
    def __init__(self, db):
        self.db = db
        self.output_dir = "data/json"

    def get_layout_geom(self):
        geom_db = self.build_graphviz_data()
        # json_parser = ParseJson(json_file)
        # geom_db = json_parser.parse()
        return geom_db

    def build_graphviz_data(self):
        """
        Draws and saves the graph with a hierarchical layout, grouping partitions
        into labeled clusters, and saves it as a JSON file.
        """
        A = pgv.AGraph(directed=True, strict=False, rankdir="TB", ratio="auto")

        info_text = "run"
        A.graph_attr["label"] = info_text
        A.graph_attr["labelloc"] = "t"
        A.graph_attr["fontsize"] = "20"

        # --- Node size scaling ---
        min_macro_size = 1.0
        max_macro_size = 10.0
        buffer_size = 0.2

        # --- Node degree scaling ---
        min_degree = 2
        max_degree = 2
        for _, inst in self.db.top_module.get_all_instances().items():
            degree = len(inst.pins.keys())
            if degree > max_degree:
                max_degree = degree

        def scale_macro_size(degree):
            if max_degree == min_degree:
                return min_macro_size
            size = min_macro_size + (degree - min_degree) * (max_macro_size - min_macro_size) / (max_degree - min_degree)
            # round to one decimal place
            return round(size, 1)

        # Group nodes by partition
        A = pgv.AGraph(directed=True, strict=False, rankdir="TB", ratio="auto")

        info_text = "run"
        A.graph_attr["label"] = info_text
        A.graph_attr["labelloc"] = "t"
        A.graph_attr["fontsize"] = "20"

        # create clusters based on number of groups
        subgraph = {}
        for block_id in self.db.groups:
            subgraph_name = f"cluster_{block_id}"
            A.add_subgraph(name=subgraph_name, label=f"Partition {block_id}", color="lightgrey", style="filled")
            subgraph[block_id] = A.get_subgraph(subgraph_name)

        for _, inst in self.db.top_module.get_all_instances().items():
            block_id = inst.partition
            graph = subgraph[block_id]

            if inst.is_buffer:
                graph.add_node(inst.name, width=buffer_size, height=buffer_size, fixedsize=True, shape="circle")
            else:
                degree = len(inst.pins.keys())
                size = scale_macro_size(degree)
                graph.add_node(inst.name, width=size, height=size, fixedsize=True, shape="box")
                # printf(f"Node {node_name} has degree {degree} and size {size}")

        # Add edges: arrow heads and tail labels swapped see:
        # see https://gitlab.com/graphviz/graphviz/-/issues/144#note_326549080
        for name, net in self.db.nets_by_name.items():
            if 2 <= net.num_conn <= self.db.fanout_threshold:
                allpins = list(net.connections)
                for group in self.db.groups:
                    pins = [pin for pin in allpins if pin.instance.partition == group]
                    if len(pins) < 2:
                        continue
                    src = pins[0]
                    for dst in pins[1:]:
                        A.add_edge(src.instance.name, dst.instance.name, label=name, headlabel=dst.full_name, taillabel=src.full_name)
        print(f"created graph with {len(A.nodes())} nodes, {len(A.edges())} edges")
        # Set graph attributes for better layout

        # Graph-level attributes
        A.graph_attr.update(
            {
                "fontsize": "20",
                "maxiter": "10000",  # Allow more iterations for force-based layouts
                "overlap": "false",  # Avoid node overlaps, TODO: try out vpsc
                "pack": "false",  # Enable packing of disconnected components
                "sep": "+20",  # Extra separation between clusters
                "splines": "ortho",  # Orthogonal edge routing
            }
        )

        # Node-level attributes
        A.node_attr.update(
            {
                "style": "filled",
                "fillcolor": "white",
                "fontsize": "10",
            }
        )

        # Edge-level attributes
        A.edge_attr.update(
            {
                "fontsize": "8",
                "arrowhead": "dot",  # Options: 'dot', 'inv', 'normal', 'none', ...
                "arrowtail": "dot",
                "arrowsize": "0.0",  # Zero-size arrows (basically hidden)
                "dir": "both",  # Draw arrows at both ends
            }
        )

        stage = self.db.stage

        def build_fn(tag: str, data_dir: str = "data") -> str:
            m = re.match(r"^(\w+?)(?:_(\w+))?$", tag)
            if not m:
                raise ValueError(f"Invalid tag format: {tag}")

            prefix, ft = m.groups()

            # If there's no explicit prefix (e.g. "png"), use "stage" as the basename
            if ft is None:
                ft = prefix
                prefix = ""
            else:
                prefix += "_"

            output_dir = os.path.join(data_dir, ft)
            os.makedirs(output_dir, exist_ok=True)
            print(f"{self.db.stage=}")
            return os.path.join(output_dir, f"{prefix}stage{stage}.{ft}")

        # Layout and draw the graph
        A.write(build_fn("pre_dot"))
        A.layout(prog="sfdp", args="")
        A.write(build_fn("placed_dot"))
        A.draw(build_fn("png"), format="png")
        A.draw(build_fn("json"), format="json")
        print(f"Graph saved to {build_fn('png')}")

        def parse_edge_pin_positions(edge):
            """
            Extract the tail and head pin coordinates from an edge's 'pos' attribute.
            """
            pos = edge.attr.get("pos")
            if not pos:
                return None

            points = pos.split()
            tail_coord = None
            head_coord = None

            wire_points = []
            for p in points:
                parts = p.split(",")
                if parts[0] == "s":  # start
                    tail_coord = (float(parts[1]), float(parts[2]))
                elif parts[0] == "e":  # end
                    head_coord = (float(parts[1]), float(parts[2]))
                else:
                    wire_points.append((float(parts[0]), float(parts[1])))

            return tail_coord, head_coord, wire_points

        geom_db = GeomDB()
        for edge in A.edges():
            tail_coord, head_coord, wire_points = parse_edge_pin_positions(edge)
            tail_pin = edge.attr.get("taillabel")
            head_pin = edge.attr.get("headlabel")
            net_name = edge.attr.get("label")

            geom_db.ports[tail_pin] = tail_coord
            geom_db.ports[head_pin] = head_coord
            geom_db.nets[net_name] = wire_points

        return geom_db
