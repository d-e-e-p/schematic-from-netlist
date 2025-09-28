import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

from schematic_from_netlist.interfaces.netlist_database import Pin
from schematic_from_netlist.interfaces.netlist_structures import Instance, PinDirection


class GenSchematicData:
    def __init__(self, db):
        self.db = db
        self.geom_db = self.db.geom_db
        self.sheet_size: Tuple[int, int] = (1000, 1000)
        self.graph_to_sch_scale = 0.24  # from graphviz to LTspice, about 72/0.24 = 300dpi
        self.schematic_grid_size = 16  # needs to be a multiple of 50 mils on kicad import for wires to connect to pin
        # TODO: many make it equal area to the smallest other macro?
        self.min_block_size_in_gridlines = 1
        self.min_block_size_min_connections = 3
        self.block_moat_size_in_gridlines = 0

    def generate_schematic(self):
        self.db.clear_all_shapes()
        self.scale_and_annotate_clusters()
        self.mark_multi_fanout_buffers()
        self.find_block_bounding_boxes()
        self.find_net_shapes()
        self.patch_pins_of_buffers()
        self.db.remove_multi_fanout_buffers()
        # self.center_schematic()
        self.calc_sheet_size()
        # self.validate_integer_geometry()
        # self.compare_cluster_bboxes()
        # self.flip_y_axis()
        return self

    # first some geom helpers
    def scale_rect(self, rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """Scale a rectangle (x1,y1,x2,y2) from grid units to real units."""
        s = self.graph_to_sch_scale
        return (int(round(rect[0] * s)), int(round(rect[1] * s)), int(round(rect[2] * s)), int(round(rect[3] * s)))

    def scale_points(self, points: list[tuple[float, float]]) -> list[tuple[int, int]]:
        """Scale a list of points [(x,y), ...]."""
        s = self.graph_to_sch_scale
        return [(int(round(x * s)), int(round(y * s))) for (x, y) in points]

    def scale_point(self, point: tuple[float, float]) -> tuple[int, int]:
        """Scale a single points (x,y)"""
        s = self.graph_to_sch_scale
        x, y = point
        return (int(round(x * s)), int(round(y * s)))

    def center_of_points(self, points):
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        num_points = len(points)
        center_x = round(sum_x / num_points)
        center_y = round(sum_y / num_points)
        point = (center_x, center_y)
        return point

    def find_block_bounding_boxes(self):
        """record inst shape as bounding box covering all pin geom ..."""
        instname2shapes = defaultdict(list)
        # pass 1: record pin shapes
        for _, inst in self.db.top_module.get_all_instances().items():
            for _, pin in inst.pins.items():
                if pin.full_name in self.geom_db.ports:
                    pin.shape = self.scale_point(self.geom_db.ports[pin.full_name])
                    instname2shapes[inst.name].append(pin.shape)
                else:
                    print(f"missing geom data for {pin.full_name=}")

        # pass 2: found bonding box
        for instname, pt_list in instname2shapes.items():
            if not pt_list:
                print(f"Warning: Inst {instname} has no geom data")
                continue

            # TODO: need to grow nicely -- if only 1 port
            # if len(pt_list) == 1:

            x_min = min(p[0] for p in pt_list)
            y_min = min(p[1] for p in pt_list)
            x_max = max(p[0] for p in pt_list)
            y_max = max(p[1] for p in pt_list)

            # too small -- need to grow
            size = self.min_block_size_in_gridlines
            if x_max - x_min < size:
                x_min -= size // 2
                x_max += size // 2
            if y_max - y_min < size:
                y_min -= size // 2
                y_max += size // 2

            # block_moat_size a bit
            size = self.block_moat_size_in_gridlines

            if len(pt_list) > self.min_block_size_min_connections:
                x_min -= size
                x_max += size
                y_min -= size
                y_max += size

            rect = (x_min, y_min, x_max, y_max)

            inst = self.db.inst_by_name.get(instname)
            inst.shape = rect

    def scale_and_uniq_points(self, points):
        """
        Remove adjacent identical points from a list.
        """
        if not points:
            return []

        scaled_points = self.scale_points(points)

        unique_points = [scaled_points[0]]  # Always keep the first point
        for pt in scaled_points[1:]:
            # Only add if different from the previous point
            if pt != unique_points[-1]:
                unique_points.append(pt)

        return unique_points

    def find_net_shapes(self):
        for name, net in self.db.top_module.get_all_nets().items():
            if name in self.geom_db.nets:
                segments = self.geom_db.nets[name]
                for seg_start, seg_end in segments:
                    pt_start = self.scale_point(seg_start)
                    pt_end = self.scale_point(seg_end)
                    if pt_start != pt_end:
                        net.shape.append((pt_start, pt_end))

    def route_to_center(self, pt_center, pt_port):
        pass

    def mark_multi_fanout_buffers(self):
        """locate inst/net by pattern"""

        pattern_net = re.compile(rf"^(?P<original>.+?){re.escape(self.db.inserted_net_suffix)}")

        def extract_original_from_net(name: str) -> str | None:
            """Return the original net name if it matches the split pattern, else None."""
            if m := pattern_net.match(name):
                return m.group("original")
            return None

        for name, net in self.db.top_module.get_all_nets().items():
            if original_net_name := extract_original_from_net(name):
                net.is_buffered_net = True
                net.buffer_original_netname = original_net_name

        pattern_cell = re.compile(rf"^{re.escape(self.db.inserted_buf_prefix)}\d+_(?P<original>.+)$")

        def extract_original_from_cell(name: str) -> str | None:
            if m := pattern_cell.match(name):
                return m.group("original")
            return None

        for name, inst in self.db.top_module.get_all_instances().items():
            # if prefix is self.inserted_buf_prefix
            if original := extract_original_from_cell(name):
                inst.is_buffer = True
                inst.buffer_original_netname = original

    def patch_pins_of_buffers(self):
        """
        shorts pins of buffers instances
        """

        for _, inst in self.db.top_module.get_all_instances().items():
            if inst.is_buffer:
                points = []
                for _, pin in inst.pins.items():
                    if pin.shape:
                        points.append(pin.shape)
                if not points:
                    print(f"perhaps problem? {inst.name} with {inst.pins.keys()} has no shapes")
                    continue
                pt_center = self.center_of_points(points)

                if inst.buffer_original_netname in self.db.nets_by_name:
                    net = self.db.nets_by_name[inst.buffer_original_netname]
                    for pt in points:
                        segment = (pt_center, pt)
                        net.buffer_patch_points.append(segment)
                    print(f"Patched buffer {inst.name} with net {inst.buffer_original_netname=} {net.buffer_patch_points=}")
                else:
                    print(f"Warning: Net {inst.buffer_original_netname} not found")
                self.db.top_module.remove_instance(inst)
        # ok now we're ready to remove logical buffers and restore logical connection

    def calc_sheet_size(self):
        """too concise perhaps?"""
        coords = []

        for _, net in self.db.top_module.get_all_nets().items():
            for p1, p2 in net.shape:
                coords.extend([p1, p2])

            if net.buffer_patch_points:
                for p1, p2 in net.buffer_patch_points:
                    coords.extend([p1, p2])

        padding = 100  # units is in grid at this point
        if coords:
            xs = [x for x, y in coords]
            ys = [y for x, y in coords]
            # if min(xs) < padding or min(ys) < padding:
            #    self.shift_all_geom_by(padding - min(xs), padding - min(ys))

            xsize = round(max(xs) - min(xs) + 2 * padding)
            ysize = round(max(ys) - min(ys) + 2 * padding)

            self.sheet_size = (xsize, ysize)
        else:
            self.sheet_size = (padding, padding)

    def compare_cluster_bboxes(self):
        """Compares the bounding box of instances within a cluster to the cluster's own shape."""
        print("\n--- Cluster Bounding Box Comparison ---")
        for cluster_id, cluster in self.db.top_module.clusters.items():
            if not cluster.instances:
                continue

            min_x, min_y = 1_000_000, 1_000_000
            max_x, max_y = -1_000_000, -1_000_000

            for inst in cluster.instances:
                if inst.shape:
                    x1, y1, x2, y2 = inst.shape
                    min_x = min(min_x, x1)
                    min_y = min(min_y, y1)
                    max_x = max(max_x, x2)
                    max_y = max(max_y, y2)

            if max_x == -1_000_000:  # No instance shapes found in cluster
                inst_bbox_str = "No instance shapes found"
            else:
                inst_bbox_str = f"Instances BBox: [({min_x}, {min_y}), ({max_x}, {max_y})]"

            cluster_shape_str = f"Cluster Shape: {cluster.shape}"

            print(f"Cluster {cluster_id}:")
            print(f"  {inst_bbox_str}")
            print(f"  {cluster_shape_str}")
        print("-------------------------------------\n")

    def scale_and_annotate_clusters(self):
        clusters = self.db.top_module.clusters

        # Scale sizes and offsets
        for cluster in clusters.values():
            cluster.size = self.scale_point(cluster.size_float)
            cluster.offset = self.scale_point(cluster.offset_float)

        # Create dummy instances for clusters to own the pins
        cluster_instances = {}
        for cluster_id in clusters.keys():
            inst = Instance(name=f"cluster_{cluster_id}", module_ref="CLUSTER", parent_module=self.db.top_module)
            cluster_instances[cluster_id] = inst

        # Add cluster pins from geom_db
        for port_name, pos in self.geom_db.ports.items():
            m = re.match(r"cluster(\d+)/(.+)", port_name)
            if m:
                cluster_id = int(m.group(1))
                net_name = m.group(2)
                pin_name = net_name

                cluster = clusters.get(cluster_id)
                cluster_inst = cluster_instances.get(cluster_id)

                if cluster and cluster_inst:
                    net = self.db.find_net(net_name)
                    pin = Pin(name=pin_name, direction=PinDirection.INOUT, instance=cluster_inst, net=net)
                    pin.shape = self.scale_point(pos)
                    cluster.add_pin(pin)

        # Set cluster shapes
        for cluster in clusters.values():
            if cluster.offset and cluster.size:
                offset_x, offset_y = cluster.offset
                size_x, size_y = cluster.size
                cluster.shape = (offset_x, offset_y, offset_x + size_x, offset_y + size_y)

    def center_schematic(self):
        """Center the entire schematic around (0,0)."""
        min_x, min_y = 1_000_000, 1_000_000
        max_x, max_y = -1_000_000, -1_000_000

        # Find bounding box of all clusters
        for cluster in self.db.top_module.clusters.values():
            if cluster.shape:
                x1, y1, x2, y2 = cluster.shape
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)

        if max_x == -1_000_000:  # No shapes found
            return

        # Calculate center and shift amount
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        shift_x = -center_x
        shift_y = -center_y

        print(
            f"Centering schematic: BBox before=[({min_x}, {min_y}), ({max_x}, {max_y})], Center=({center_x}, {center_y}), Shift=({shift_x}, {shift_y})"
        )

        # Shift all geometric data
        self.shift_all_geom_by(shift_x, shift_y)

        # For verification, calculate new bounding box
        new_min_x, new_min_y = 1_000_000, 1_000_000
        new_max_x, new_max_y = -1_000_000, -1_000_000
        for cluster in self.db.top_module.clusters.values():
            if cluster.shape:
                x1, y1, x2, y2 = cluster.shape
                new_min_x = min(new_min_x, x1)
                new_min_y = min(new_min_y, y1)
                new_max_x = max(new_max_x, x2)
                new_max_y = max(new_max_y, y2)

        print(f"Centering schematic: BBox after=[({new_min_x}, {new_min_y}), ({new_max_x}, {new_max_y})]")

    def shift_all_geom_by(self, dx, dy):
        """Shift all geometric data by a given offset."""
        # Shift instance shapes
        for inst in self.db.top_module.get_all_instances().values():
            if inst.shape:
                x1, y1, x2, y2 = inst.shape
                inst.shape = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            for pin in inst.pins.values():
                if pin.shape:
                    pin.shape = (pin.shape[0] + dx, pin.shape[1] + dy)

        # Shift net shapes
        for net in self.db.top_module.get_all_nets().values():
            net.shape = [((p1[0] + dx, p1[1] + dy), (p2[0] + dx, p2[1] + dy)) for p1, p2 in net.shape]
            net.buffer_patch_points = [((p1[0] + dx, p1[1] + dy), (p2[0] + dx, p2[1] + dy)) for p1, p2 in net.buffer_patch_points]

        # Shift cluster shapes, pins, and offsets
        for cluster in self.db.top_module.clusters.values():
            if cluster.shape:
                x1, y1, x2, y2 = cluster.shape
                cluster.shape = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            if cluster.offset:
                cluster.offset = (cluster.offset[0] + dx, cluster.offset[1] + dy)
            for pin in cluster.pins.values():
                if pin.shape:
                    pin.shape = (pin.shape[0] + dx, pin.shape[1] + dy)

    def validate_integer_geometry(self):
        """Validate that all geometry data consists of integers."""
        for inst in self.db.top_module.get_all_instances().values():
            if inst.shape:
                assert all(isinstance(v, int) for v in inst.shape), (
                    f"Instance {inst.name} shape has non-integer values: {inst.shape}"
                )
            for pin in inst.pins.values():
                if pin.shape:
                    assert all(isinstance(v, int) for v in pin.shape), (
                        f"Pin {pin.full_name} shape has non-integer values: {pin.shape}"
                    )
        for net in self.db.top_module.get_all_nets().values():
            for p1, p2 in net.shape:
                assert all(isinstance(v, int) for v in p1), f"Net {net.name} shape has non-integer values: {p1}"
                assert all(isinstance(v, int) for v in p2), f"Net {net.name} shape has non-integer values: {p2}"
        for cluster in self.db.top_module.clusters.values():
            if cluster.shape:
                assert all(isinstance(v, int) for v in cluster.shape), (
                    f"Cluster {cluster.id} shape has non-integer values: {cluster.shape}"
                )
            for pin in cluster.pins.values():
                if pin.shape:
                    assert all(isinstance(v, int) for v in pin.shape), (
                        f"Cluster pin {pin.full_name} shape has non-integer values: {pin.shape}"
                    )
        print("Geometry validation passed: All coordinates are integers.")
