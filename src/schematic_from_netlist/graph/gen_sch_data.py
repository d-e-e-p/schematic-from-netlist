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
        self.min_block_size_in_gridlines = 10
        self.block_moat_size_in_gridlines = 1

    # first some geom helpers
    def scale_rect(self, rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """Scale a rectangle (x1,y1,x2,y2) from grid units to real units."""
        s = self.graph_to_sch_scale
        return (round(rect[0] * s), round(rect[1] * s), round(rect[2] * s), round(rect[3] * s))

    def scale_points(self, points: list[tuple[float, float]]) -> list[tuple[int, int]]:
        """Scale a list of points [(x,y), ...]."""
        s = self.graph_to_sch_scale
        return [(round(x * s), round(y * s)) for (x, y) in points]

    def scale_point(self, point: tuple[float, float]) -> tuple[int, int]:
        """Scale a single points (x,y)"""
        s = self.graph_to_sch_scale
        x, y = point
        return (round(x * s), round(y * s))

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
                pt_center = self.center_of_points(points)

                if inst.buffer_original_netname in self.db.nets_by_name:
                    net = self.db.nets_by_name[inst.buffer_original_netname]
                    for pt in points:
                        segment = (pt_center, pt)
                        net.buffer_patch_points.append(segment)
                    print(f"Patched buffer {inst.name} with net {inst.buffer_original_netname=} {net.buffer_patch_points=}")
                else:
                    print(f"Warning: Net {inst.buffer_original_netname} not found")
        # ok now we're ready to remove logical buffers and restore logical connection

    def shift_all_geom_by(self, dx, dy):
        """
        for inst_shape in self.inst_shapes:
            x1, y1, x2, y2 = inst_shape.rect
            inst_shape.rect = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            for port_shape in inst_shape.port_shapes:
                port_shape.point = (port_shape.point[0] + dx, port_shape.point[1] + dy)

        for net_shape in self.net_shapes:
            net_shape.points = [(x + dx, y + dy) for x, y in net_shape.points]
        """

    def calc_sheet_size(self):
        """too concise perhaps?"""
        coords = []

        for _, net in self.db.top_module.get_all_nets().items():
            for x, y in net.shape:
                coords.extend([x, y])

            if net.buffer_patch_points:
                for x, y in net.buffer_patch_points:
                    coords.extend([x, y])

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

    def flip_y_axis(self):
        """
        In schematic drawing programs historically the origin is upper-left.
        This flips all shapes vertically relative to the sheet height.
        """
        _, sheet_height = self.sheet_size

        # Flip instance rectangles
        for shape in self.inst_shapes:
            x1, y1, x2, y2 = shape.rect
            # Reflect across horizontal axis at sheet_height
            y1_flipped = sheet_height - y1
            y2_flipped = sheet_height - y2
            # Normalize (y1 should be min, y2 max)
            y1n, y2n = min(y1_flipped, y2_flipped), max(y1_flipped, y2_flipped)
            shape.rect = (x1, y1n, x2, y2n)

        for shape in self.inst_shapes:
            for shape in shape.port_shapes:
                x, y = shape.point
                # Reflect across horizontal axis at sheet_height
                y_flipped = sheet_height - y
                shape.point = (x, y_flipped)

        # Flip net points
        for shape in self.net_shapes:
            flipped_points = []
            for x, y in shape.points:
                flipped_points.append((x, sheet_height - y))
            shape.points = flipped_points

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

    def generate_schematic(self):
        self.db.clear_all_shapes()
        self.scale_and_annotate_clusters()
        self.mark_multi_fanout_buffers()
        self.find_block_bounding_boxes()
        self.find_net_shapes()
        self.patch_pins_of_buffers()
        self.db.remove_multi_fanout_buffers()
        self.calc_sheet_size()
        # self.flip_y_axis()
        return self
