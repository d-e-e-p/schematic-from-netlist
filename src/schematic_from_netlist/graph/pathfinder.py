import enum
import math
import os

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx

from schematic_from_netlist.graph.geom_utils import Geom
from schematic_from_netlist.graph.router import Router


class Side(enum.Enum):
    NONE = 0
    TOP = 1
    BOTTOM = 2
    LEFT = 3
    RIGHT = 4


class Pathfinder:
    def __init__(self, db, schematic_db):
        self.db = db
        self.schematic_db = schematic_db
        self.G = nx.grid_2d_graph(0, 0)
        self.obstacles = set()

    def modify_line_blockages(self, create_blockage_questionmark, pt_start, pt_end):
        # Scale to grid coordinates
        x0, y0 = (pt_start[0], pt_start[1])
        x1, y1 = (pt_end[0], pt_end[1])

        def bresenham(x0, y0, x1, y1):
            """Yield integer grid points along a line from (x0, y0) to (x1, y1)."""
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                yield x0, y0
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        # Get obstacle points along the line and remove them from the graph
        blockages = list(bresenham(x0, y0, x1, y1))
        if create_blockage_questionmark:
            self.obstacles.update(blockages)
        else:
            self.obstacles.difference_update(blockages)

    def modify_wire_blockages(self, create_blockage_questionmark, wire_shape):
        pt_start = wire_shape.points[0]
        for pt in wire_shape.points[1:]:
            pt_end = pt
            self.modify_line_blockages(create_blockage_questionmark, pt_start, pt_end)
            pt_start = pt_end

    def get_pin_side(self, rect, pt):
        """
        Determines which side of the macro a pin is on and the distance to it.

        Returns:
            tuple: (Side, float)
        """
        x, y = pt
        x1, y1, x2, y2 = rect

        # Calculate distances to each boundary
        dist_left = abs(x - x1)
        dist_right = abs(x - x2)
        dist_top = abs(y - y1)
        dist_bottom = abs(y - y2)

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_dist == dist_left:
            return (Side.LEFT, dist_left)
        elif min_dist == dist_right:
            return (Side.RIGHT, dist_right)
        elif min_dist == dist_top:
            return (Side.TOP, dist_top)
        elif min_dist == dist_bottom:
            return (Side.BOTTOM, dist_bottom)

    def clear_space_for_pin_access(self, rect, pt):
        """
        Generates a list of coordinates representing a path escaping a macro.
        """
        extra_clearance = 3
        x_start, y_start = pt
        # Unpack the new return value from get_pin_side
        escape_direction, distance_to_edge = self.get_pin_side(rect, pt)

        path_coords = []
        current_x, current_y = x_start, y_start

        for _ in range(distance_to_edge + extra_clearance):
            path_coords.append((current_x, current_y))
            path_coords.append((current_x + 1, current_y))
            path_coords.append((current_x, current_y + 1))
            path_coords.append((current_x - 1, current_y))
            path_coords.append((current_x, current_y - 1))

            if escape_direction == Side.LEFT:
                current_x -= 1
            elif escape_direction == Side.RIGHT:
                current_x += 1
            elif escape_direction == Side.TOP:
                current_y -= 1
            elif escape_direction == Side.BOTTOM:
                current_y += 1

        return path_coords

    def create_inst_blockages(self, inst_shape):
        """don't route over blocks"""
        halo = 2
        ll_x, ll_y, ur_x, ur_y = inst_shape.rect
        blockages = [(x, y) for x in range(ll_x - halo, ur_x + 1 + halo) for y in range(ll_y - halo, ur_y + 1 + halo)]
        self.obstacles.update(blockages)

    def find_target_and_clearence_of_pin(self, net):
        endpoints = []
        clearences = []
        # assume has shape!
        for pin in net.connections:
            print(f"looking for pin {pin.full_name} of {pin.instance.name} : {self.schematic_db.portshape_by_name.keys()}")
            if pin.full_name not in self.schematic_db.portshape_by_name:
                breakpoint()
            port_shape = self.schematic_db.portshape_by_name[pin.full_name]
            inst_shape = self.schematic_db.instshape_by_name[pin.instance.name]
            endpoints.append(port_shape.point)
            clearences.extend(self.clear_space_for_pin_access(inst_shape.rect, port_shape.point))
            # if net.name == "c_fanout_buffer_3":
            #    breakpoint()
        return endpoints, clearences

    def convert_paths_to_shapes(self, net, paths):
        stop = net.name == "c_fanout_buffer_3"
        all_segments = Geom.extract_segments_from_all_paths(paths, stop)
        # throw away turn info for now
        all_segs = [seg for segs, _ in all_segments for seg in segs]
        print(f"{net.name} has {len(all_segs)} segments: {all_segs=}")
        shape = self.schematic_db.netshape_by_name[net.name]
        shape.segments = all_segs

    def reroute_net(self, net, points_to_connect):
        # ---  Compute Pairwise Shortest Paths using igraph ---
        default_costs = {"pref_dir_cost": 1, "wrong_way_cost": 2, "via_cost": 5}
        width, height = self.schematic_db.sheet_size
        router = Router(width, height, default_costs)
        paths = router.route(points_to_connect, self.obstacles)
        router.visualize_routing(net.name, paths, points_to_connect, self.obstacles)
        self.convert_paths_to_shapes(net, paths)

    def cleanup_routes(self):
        width, height = self.schematic_db.sheet_size
        print(f"Grid {width=} X {height=} ")

        for inst_shape in self.schematic_db.inst_shapes:
            self.create_inst_blockages(inst_shape)

        # Process wires
        for wire_shape in self.schematic_db.net_shapes:
            self.modify_wire_blockages(create_blockage_questionmark=True, wire_shape=wire_shape)

        sorted_nets = sorted(self.db.nets_by_name.values(), key=lambda net: net.num_conn, reverse=True)
        for net in sorted_nets:
            if net.num_conn > 1:
                if net.name in self.schematic_db.netshape_by_name:
                    wire_shape = self.schematic_db.netshape_by_name[net.name]
                    self.modify_wire_blockages(create_blockage_questionmark=False, wire_shape=wire_shape)
                    endpoints, clearences = self.find_target_and_clearence_of_pin(net)
                    print(f"Rerouting net {net.name} with endpoints {endpoints} and clearences {clearences}")
                    self.obstacles.difference_update(clearences)
                    self.reroute_net(net, endpoints)
