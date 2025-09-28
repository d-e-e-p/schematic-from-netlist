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
    nets: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = field(default_factory=dict)

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
        return geom_db

    def build_graphviz_data(self):
        """
        Performs a two-phase layout: local layout for each cluster, then a global layout.
        """
        # Phase 1: Local layout for each cluster
        local_geom_dbs = self._perform_local_layout()

        # Phase 2: Global layout of clusters
        global_geom_db = self._perform_global_layout()

        # Combine local and global geometries
        final_geom_db = self._combine_geom_dbs(local_geom_dbs, global_geom_db)

        return final_geom_db

    def _perform_local_layout(self):
        """Performs layout for each cluster individually to determine their size."""
        local_geom_dbs = {}
        os.makedirs("data/png", exist_ok=True)

        for cluster_id, cluster in self.db.top_module.clusters.items():
            A = pgv.AGraph(directed=True, strict=False, rankdir="TB", ratio="auto")
            self._set_attributes(A)

            instances_in_cluster = cluster.instances
            inst_names = [inst.name for inst in instances_in_cluster]
            print(f"Performing layout for cluster {cluster_id} with {inst_names=}")
            breakpoint()

            # Add nodes for instances in the cluster, tagging them if they are buffers
            for inst in instances_in_cluster:
                A.add_node(inst.name, _is_buffer=str(inst.is_buffer))

            # Add edges for nets internal to the cluster
            nets_in_cluster = {net for inst in instances_in_cluster for net in inst.get_connected_nets()}
            for net in nets_in_cluster:
                if all(pin.instance.partition == cluster_id for pin in net.pins):
                    pins = list(net.pins)
                    if len(pins) > 1:
                        src = pins[0]
                        for dst in pins[1:]:
                            A.add_edge(
                                src.instance.name,
                                dst.instance.name,
                                label=net.name,
                                headlabel=dst.full_name,
                                taillabel=src.full_name,
                            )
                    elif len(pins) == 1:
                        pin = pins[0]
                        stub_name = f"stub_{pin.full_name}".replace("/", "_")
                        A.add_node(stub_name, shape="point", width=0.01, height=0.01)
                        A.add_edge(
                            pin.instance.name,
                            stub_name,
                            label=net.name,
                            headlabel=pin.full_name,
                            taillabel=pin.full_name,
                        )

            # --- Node size scaling based on graph degree ---
            min_macro_size = 1.0
            max_macro_size = 10.0
            buffer_size = 0.2
            low_fanout_size = 3.0

            min_degree = 2
            max_degree = 2
            # Calculate max_degree from the graph for non-buffer instances
            degrees = [
                A.degree(node) for node in A.nodes() if not node.name.startswith("stub_") and node.attr.get("_is_buffer") == "False"
            ]
            if degrees:
                max_degree = max(degrees) if degrees else 2

            def scale_macro_size(degree):
                if degree <= 3:
                    return low_fanout_size
                if max_degree <= min_degree:
                    return min_macro_size
                # Clamp degree to be at least min_degree for scaling
                degree_to_scale = max(min_degree, degree)
                size = min_macro_size + (degree_to_scale - min_degree) * (max_macro_size - min_macro_size) / (
                    max_degree - min_degree
                )
                return round(size, 1)

            # Update node attributes (shape and size)
            for node in A.nodes():
                if node.name.startswith("stub_"):
                    continue  # Already handled when added

                is_buffer = node.attr.get("_is_buffer") == "True"

                if is_buffer:
                    node.attr["shape"] = "circle"
                    node.attr["width"] = buffer_size
                    node.attr["height"] = buffer_size
                else:
                    node.attr["shape"] = "box"
                    degree = A.degree(node)
                    size = scale_macro_size(degree)
                    node.attr["width"] = size
                    node.attr["height"] = size
                node.attr["fixedsize"] = True

            # Layout and extract size
            os.makedirs("data/dot", exist_ok=True)
            A.write(f"data/dot/local_cluster_{cluster_id}_pre.dot")
            A.layout(prog="fdp", args="-y")
            A.write(f"data/dot/local_cluster_{cluster_id}_post.dot")
            A.draw(f"data/png/local_cluster_{cluster_id}.png", format="png")
            bb = A.graph_attr["bb"]
            if bb:
                xmin, ymin, xmax, ymax = map(float, bb.split(","))
                cluster.size_float = (abs(xmax - xmin), abs(ymax - ymin))

            local_geom_db = self._extract_geometry(A)

            # Add all instance pins to the geom db
            for inst in instances_in_cluster:
                node = A.get_node(inst.name)
                pos = node.attr["pos"]
                if pos:
                    x, y = map(float, pos.split(","))
                    for pin in inst.pins.values():
                        if pin.full_name not in local_geom_db.ports:
                            local_geom_db.ports[pin.full_name] = (x, y)

            local_geom_dbs[cluster_id] = local_geom_db
        return local_geom_dbs

    def _perform_global_layout(self):
        """Performs layout on the clusters as blocks."""
        A = pgv.AGraph(directed=True, strict=False, rankdir="TB", ratio="auto")
        self._set_attributes(A)

        # Add a node for each cluster
        for cluster_id, cluster in self.db.top_module.clusters.items():
            width, height = cluster.size_float if cluster.size_float else (1.0, 1.0)
            A.add_node(
                f"cluster_{cluster_id}",
                label=f"Cluster {cluster_id}",
                width=width / 72.0,
                height=height / 72.0,
                fixedsize=True,
                shape="box",
            )

        # Add edges for inter-cluster nets
        inter_cluster_nets = [net for net in self.db.nets_by_name.values() if len({pin.instance.partition for pin in net.pins}) > 1]
        for net in inter_cluster_nets:
            partitions = list({pin.instance.partition for pin in net.pins})
            for i in range(len(partitions) - 1):
                src_partition = partitions[i]
                dst_partition = partitions[i + 1]
                src_cluster = f"cluster_{src_partition}"
                dst_cluster = f"cluster_{dst_partition}"
                A.add_edge(
                    src_cluster,
                    dst_cluster,
                    label=net.name,
                    headlabel=f"cluster{dst_partition}/{net.name}",
                    taillabel=f"cluster{src_partition}/{net.name}",
                )

        os.makedirs("data/dot", exist_ok=True)
        A.write("data/dot/global_layout_pre.dot")
        A.layout(prog="dot")
        A.write("data/dot/global_layout_post.dot")
        A.draw("data/png/global_layout.png", format="png")

        geom_db = self._extract_geometry(A)

        # Store cluster offsets
        for node in A.nodes():
            cluster_id = int(node.name.split("_")[1])
            cluster = self.db.top_module.clusters.get(cluster_id)
            if cluster:
                pos = node.attr["pos"]
                if pos:
                    x, y = map(float, pos.split(","))
                    width, height = cluster.size_float if cluster.size_float else (0, 0)
                    cluster.offset_float = (x - width / 2, y - height / 2)

        # Add cluster ports to the geometry database from edge label positions
        for edge in A.edges():
            head_label = edge.attr.get("headlabel")
            tail_label = edge.attr.get("taillabel")
            if head_label and edge.attr.get("lp"):
                x, y = map(float, edge.attr.get("lp").split(","))
                geom_db.ports[head_label] = (x, y)
            if tail_label and edge.attr.get("lp"):
                x, y = map(float, edge.attr.get("lp").split(","))
                geom_db.ports[tail_label] = (x, y)

        return geom_db

    def _combine_geom_dbs(self, local_geom_dbs, global_geom_db):
        """Combines local and global geometries into a single GeomDB, appending wire segments."""
        final_geom_db = GeomDB()

        # Process local geometries first
        for cluster_id, local_geom_db in local_geom_dbs.items():
            cluster = self.db.top_module.clusters.get(cluster_id)
            if cluster:
                offset_x, offset_y = cluster.offset_float if cluster.offset_float else (0, 0)

                # Add ports, no overwriting concern here as pin names are unique
                for name, pos in local_geom_db.ports.items():
                    if pos:  # ensure pos is not None
                        final_geom_db.ports[name] = (pos[0] + offset_x, pos[1] + offset_y)

                # Add nets, appending points if net already exists
                for name, segments in local_geom_db.nets.items():
                    offset_segments = [
                        ((p1[0] + offset_x, p1[1] + offset_y), (p2[0] + offset_x, p2[1] + offset_y)) for p1, p2 in segments
                    ]
                    if name in final_geom_db.nets:
                        final_geom_db.nets[name].extend(offset_segments)
                    else:
                        final_geom_db.nets[name] = offset_segments

        # Process global geometries
        for name, pos in global_geom_db.ports.items():
            final_geom_db.ports[name] = pos

        # Add global nets, appending points
        for name, points in global_geom_db.nets.items():
            if name in final_geom_db.nets:
                final_geom_db.nets[name].extend(points)
            else:
                final_geom_db.nets[name] = points

        return final_geom_db

    def _set_attributes(self, A):
        """Sets graph, node, and edge attributes."""
        A.graph_attr.update(
            layout="fdp",
            K="0.2",
            maxiter="10000",
            overlap="false",
            pack="true",
            packmode="graph",
            sep="+2,2",
            esep="+2,2",
            epsilon="0.0001",
            rankdir="tb",
            ratio="auto",
            splines="ortho",
        )

        A.node_attr.update(fillcolor="white", margin="0,0", width="0.4", height="0.3", fixedsize="true")

        A.edge_attr.update(len="0.3", weight="1.5")
        A.edge_attr.update(
            {
                "fontsize": "8",
                "arrowhead": "dot",
                "arrowtail": "dot",
                "arrowsize": "0.0",
                "dir": "both",
            }
        )

    def _extract_geometry(self, A):
        """Extracts geometry from the graph."""
        geom_db = GeomDB()

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
            net_name = edge.attr.get("label")

            if tail_pin and tail_coord:
                geom_db.ports[tail_pin] = tail_coord
            if head_pin and head_coord:
                geom_db.ports[head_pin] = head_coord
            if net_name:
                if net_name in geom_db.nets:
                    geom_db.nets[net_name].extend(wire_segments)
                else:
                    geom_db.nets[net_name] = wire_segments
        return geom_db
