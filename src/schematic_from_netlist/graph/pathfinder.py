import enum
import math
import os

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx

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
        for pin in net.pins:
            for inst_shape in self.schematic_db.inst_shapes:
                for port_shape in inst_shape.port_shapes:
                    if pin.name == port_shape.pin.name:
                        endpoints.append(port_shape.point)
                        clearences.extend(self.clear_space_for_pin_access(inst_shape.rect, port_shape.point))
        return endpoints, clearences

    def merge_collinear_segments(self, segments):
        """
        Merge consecutive collinear segments into maximal straight lines.

        Args:
            segments: list of ((x1,y1),(x2,y2))

        Returns:
            list of merged segments
        """

        if not segments:
            return []

        merged = [list(segments[0])]  # start with first segment endpoints

        for (x1, y1), (x2, y2) in segments[1:]:
            x0, y0 = merged[-1][-1]  # last point of current merged
            x_prev, y_prev = merged[-1][-2]

            # direction of current merged
            v1 = (x0 - x_prev, y0 - y_prev)
            # direction of candidate segment
            v2 = (x2 - x1, y2 - y1)

            # check collinearity (cross product = 0) and same orientation
            if v1[0] * v2[1] - v1[1] * v2[0] == 0:
                # extend last segment to new endpoint
                merged[-1][-1] = (x2, y2)
            else:
                # start a new merged segment
                merged.append([(x1, y1), (x2, y2)])

        # convert back to tuple pairs
        return [tuple(pair) for pair in merged]

    def extract_turn_points(self, all_paths):
        """
        Collapse all paths, handle gaps (non-adjacent points), and
        return line segments with turns.

        Args:
            all_paths (list of list of tuples)):
                Each path is a list of (x,y,layer).

        Returns:
            list of list of segments:
                Each segment is ((x1,y1),(x2,y2)) continuous piece.
        """

        def is_turn(p1, p2, p3):
            """Check if p2 is a turning point (not collinear)."""
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            return v1[0] * v2[1] - v1[1] * v2[0] != 0

        def is_adjacent(p1, p2):
            """Check if p1 and p2 are grid-adjacent (Manhattan distance=1)."""
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

        all_segments = []

        for path in all_paths:
            collapsed = [(x, y) for (x, y, _) in path]

            segments = []
            turns = []
            current_seg = [collapsed[0]]

            # print("combined with turns")

            for i in range(1, len(collapsed)):
                p_prev = collapsed[i - 1]
                p_curr = collapsed[i]

                # Check adjacency
                if not is_adjacent(p_prev, p_curr):
                    # break segment
                    if len(current_seg) > 1:
                        segments.append((current_seg[0], current_seg[-1]))
                    # print(f" --- GAP between {p_prev} and {p_curr} ---")
                    current_seg = [p_curr]
                    continue

                # Add to current segment
                current_seg.append(p_curr)

                # Check turn (need one behind and one ahead)
                if i < len(collapsed) - 1:
                    p_next = collapsed[i + 1]
                    if is_turn(p_prev, p_curr, p_next):
                        turns.append(p_curr)

            # Close final segment
            if len(current_seg) > 1:
                segments.append((current_seg[0], current_seg[-1]))

            all_segments.append((segments, turns))

        col_segments = self.merge_collinear_segments(segments)
        return col_segments

    def convert_paths_to_shapes(self, net, paths):
        segments = self.extract_turn_points(paths)
        shape = self.schematic_db.netshape_by_name[net.name]
        shape.segments = segments

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
            if net.num_conn > 4:
                if net.name in self.schematic_db.netshape_by_name:
                    wire_shape = self.schematic_db.netshape_by_name[net.name]
                    self.modify_wire_blockages(create_blockage_questionmark=False, wire_shape=wire_shape)
                    endpoints, clearences = self.find_target_and_clearence_of_pin(net)
                    print(f"Rerouting net {net.name} with endpoints {endpoints} and clearences {clearences}")
                    self.obstacles.difference_update(clearences)
                    self.reroute_net(net, endpoints)
