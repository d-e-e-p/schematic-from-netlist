# src/schematic_from_netlist/router/astar_router.py

from __future__ import annotations

import heapq
import logging as log
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import linemerge, snap, unary_union

from schematic_from_netlist.astar_router.ar_cost_estimator import CostEstimator
from schematic_from_netlist.astar_router.ar_debug import plot_occupancy_summary, plot_routing_debug_image, plot_routing_summary
from schematic_from_netlist.astar_router.ar_occupancy import OccupancyMap
from schematic_from_netlist.database.netlist_structures import Module, Net, Pin

# A node in the A* search grid
# (f_score, g_score, (x, y), parent_node)
SearchNode = Tuple[float, float, Tuple[int, int], Optional[Tuple[int, int]]]

# root_logger = log.getLogger()
# root_logger.setLevel(log.DEBUG)


class AstarRouter:
    def __init__(self, db):
        self.db = db
        self.occupancy_map: Optional[OccupancyMap] = None
        self.cost_estimator: Optional[Cost] = None
        self.grid_size = 1
        self.halo_size = 2
        self.halo_geometries: List = []
        self.cost_estimator = CostEstimator(self.occupancy_map)

    def adjust_pin_locations(self, module: Module):
        """
        Adjusts pin locations for non-fixed pins on macros to improve routability.
        This is a complex optimization problem. A simple heuristic could be to
        move the pin towards the centroid of the other pins in its net.
        """
        log.info("Adjusting pin locations...")
        for net in module.nets.values():
            # Calculate centroid of all pins in the net
            # ...
            for pin in net.pins:
                if hasattr(pin, "draw") and hasattr(pin.draw, "geom") and not pin.draw.geom.fixed:
                    # This pin can be moved.
                    # The pin belongs to an instance, we need to find which one.
                    # The pin's movement is constrained by the macro's boundary.
                    # Placeholder for pin adjustment logic.
                    pass

    def route_net(self, module: Module, net: Net):
        """
        Routes a single net. For multi-pin nets, it builds a minimum spanning tree
        like structure by connecting each pin to the nearest point on the existing route tree.
        """
        pins = list(net.pins.values())
        if len(pins) < 2:
            return

        # Start with the first pin as the initial routed tree
        routed_pins = {pins[0]}
        unrouted_pins = set(pins[1:])
        route_tree = []

        while unrouted_pins:
            start_pin, end_pin = self._find_next_connection(routed_pins, unrouted_pins, route_tree)

            if start_pin is None or end_pin is None:
                log.warning(f"Could not find next connection for net {net.name}")
                break

            path = self.astar_search(start_pin, end_pin)

            if path:
                log.debug(f"Adding path to route tree: {path}")
                route_tree.append(path)
                # Debug: print occupancy before update
                log.debug(f"Updating occupancy for path: {path}")
                self.occupancy_map.update_occupancy(path)
                # Debug: print some occupancy values to verify
                # Sample a few points along the path to check their occupancy
                log.info(f"testing path for occupancy: {path}")
                for i in range(5):
                    sample_point = path.interpolate(i / 4, normalized=True)
                    ix, iy = self.occupancy_map._world_to_grid(sample_point.x, sample_point.y)
                    occupancy = self.occupancy_map.grid[ix, iy]
                    log.info(f"Occupancy at grid ({ix}, {iy}) after update: {occupancy}")
            else:
                log.warning(
                    f"A* search failed for net {net.name} between {start_pin.full_name} at {start_pin.draw.geom} and {end_pin.full_name} at {end_pin.draw.geom}."
                )

            routed_pins.add(end_pin)
            unrouted_pins.remove(end_pin)

        if route_tree:
            net.draw.geom = self.merge_routes(route_tree)
            # self.occupancy_map.update_occupancy(net.draw.geom) # This was moved inside the loop
            output_path = f"data/images/route/{module.name}_{net.name}.png"
            plot_routing_debug_image(module, net, route_tree, self.occupancy_map, self.cost_estimator, output_path)

    def _find_next_connection(self, routed_pins, unrouted_pins, route_tree):
        # Find the cheapest connection from an unrouted pin to the routed tree.
        # This is a simplification. A proper approach would use something like Prim's algorithm.
        best_start = None
        best_end = None
        min_dist = float("inf")

        for end_pin in unrouted_pins:
            if not route_tree:  # First connection
                # Find closest routed pin (there's only one)
                start_pin = next(iter(routed_pins))
                dist = start_pin.draw.geom.distance(end_pin.draw.geom)
                if dist < min_dist:
                    min_dist = dist
                    best_start = start_pin
                    best_end = end_pin
            else:
                # Find closest point on the existing route tree
                # This is computationally expensive. A simplification is to check against all routed pins.
                for start_pin in routed_pins:
                    dist = start_pin.draw.geom.distance(end_pin.draw.geom)
                    if dist < min_dist:
                        min_dist = dist
                        best_start = start_pin
                        best_end = end_pin
        return best_start, best_end

    def _initialize_search(self, start_pin: Pin, end_pin: Pin) -> tuple:
        """Initialize search parameters and clear occupancy at start/end nodes."""
        start_point = start_pin.draw.geom
        end_point = end_pin.draw.geom

        log.debug(f"A* search from {start_point} to {end_point}")

        start_node = self.occupancy_map._world_to_grid(start_point.x, start_point.y)
        end_node = self.occupancy_map._world_to_grid(end_point.x, end_point.y)

        # Temporarily clear occupancy at start and end nodes
        start_node_occupancy = self.occupancy_map.grid[start_node]
        end_node_occupancy = self.occupancy_map.grid[end_node]
        self.occupancy_map.grid[start_node] = 0
        self.occupancy_map.grid[end_node] = 0

        log.info(f"Start node: {start_node}, End node: {end_node}")
        log.info(f"Start node occupancy: {self.occupancy_map.grid[start_node]}")
        log.info(f"End node occupancy: {self.occupancy_map.grid[end_node]}")

        return start_point, end_point, start_node, end_node, start_node_occupancy, end_node_occupancy

    def _handle_trivial_case(self, start_point, end_point, start_node, end_node) -> Optional[LineString]:
        """Handle trivial cases where start and end nodes are the same."""
        if start_node == end_node:
            px, py = start_point.x, start_point.y
            ex, ey = end_point.x, end_point.y
            return LineString([(px, py), (px, ey), (ex, ey)])
        return None

    def _process_neighbor(self, current, neighbor, parent, macro_center, start_node, end_node, open_set, came_from, g_score):
        """Process a neighbor node during A* search with detailed diagnostics."""
        # Get occupancy and cost details
        move_cost = self.cost_estimator.get_neighbor_move_cost(current, neighbor, parent, macro_center, start_node)
        tentative_g_score = g_score[current] + move_cost

        # Detailed debug logging
        log.debug(
            f"Processing neighbor {neighbor}:"
            f"  Current node: {current}"
            f"  Parent node: {parent}"
            f"  Move cost: {move_cost}"
            f"  Tentative g_score: {tentative_g_score}"
        )

        # Check if we should update this neighbor's path
        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
            h = self._heuristic(neighbor, end_node)
            f = tentative_g_score + h

            # Log heuristic and total cost
            log.debug(f"  Heuristic (h): {h}")
            log.debug(f"  Total cost (f): {f}")

            # Update scores and add to open set
            g_score[neighbor] = tentative_g_score
            heapq.heappush(open_set, (f, tentative_g_score, neighbor, current))
            came_from[neighbor] = current

            log.debug(f"  Updated path through {neighbor} with f={f}")

            # Log comparison with previous best if it exists
            if neighbor in g_score:
                log.debug(f"  Previous g_score: {g_score[neighbor]}")
                log.debug(f"  Improvement: {g_score[neighbor] - tentative_g_score}")
        else:
            log.debug(f"  Keeping existing path (current g_score: {g_score[neighbor]})")

    def _create_final_path(self, path, start_point, end_point) -> LineString:
        """Create the final path with proper bends at start and end points."""
        path_points = list(path.coords)
        if len(path_points) < 2:
            # Handle short paths with an L-bend
            px, py = start_point.x, start_point.y
            ex, ey = end_point.x, end_point.y
            return LineString([(px, py), (ex, py), (ex, ey)])

        p1 = path_points[1]
        pn_2 = path_points[-2]

        # Determine bend points
        bend_start = (p1[0], start_point.y) if path_points[0][0] == p1[0] else (start_point.x, p1[1])
        bend_end = (pn_2[0], end_point.y) if path_points[-1][0] == pn_2[0] else (end_point.x, pn_2[1])

        # Construct and clean up path
        new_path_points = [start_point, bend_start] + path_points[1:-1] + [bend_end, end_point]
        final_path = [new_path_points[0]]
        for point in new_path_points[1:]:
            if point != final_path[-1]:
                final_path.append(point)

        return LineString(final_path)

    def astar_search(self, start_pin: Pin, end_pin: Pin) -> Optional[LineString]:
        """
        A* search on a grid to find a path between two pins with detailed diagnostics.
        """
        # Initialize search with detailed logging
        log.info(f"\nStarting A* search from {start_pin.full_name} to {end_pin.full_name}")
        start_point, end_point, start_node, end_node, start_occ, end_occ = self._initialize_search(start_pin, end_pin)

        # Check for trivial case
        if trivial_path := self._handle_trivial_case(start_point, end_point, start_node, end_node):
            log.debug("Trivial path found (start and end nodes are the same)")
            return trivial_path

        # Initialize search structures
        open_set = []
        start_h = self._heuristic(start_node, end_node)
        heapq.heappush(open_set, (start_h, 0.0, start_node, None))
        came_from = {}
        g_score = {start_node: 0}
        path = None
        iteration = 0

        # Main search loop with iteration tracking
        while open_set:
            iteration += 1
            f, g, current, _ = heapq.heappop(open_set)
            parent = came_from.get(current)

            log.debug(
                f"Iteration {iteration}:"
                f"  Processing node {current}"
                f"  Current f_score: {f}"
                f"  Current g_score: {g}"
                f"  Parent node: {parent}"
            )

            if not parent:
                macro_center = self.get_center_of_pin_macro(start_point, start_pin, end_pin)
                log.info(f"  Macro center: {macro_center}")

            if current == end_node:
                path = self._reconstruct_path(came_from, current)
                log.info(f"Path found after {iteration} iterations: {path}")
                break

            # Process all neighbors with detailed diagnostics
            neighbors = list(self._get_neighbors(current))
            log.debug(f"  Processing {len(neighbors)} neighbors")
            for neighbor in neighbors:
                self._process_neighbor(
                    current, neighbor, parent, macro_center, start_node, end_node, open_set, came_from, g_score
                )

            # Log open set status
            log.debug(f"  Open set size: {len(open_set)}")
            if open_set:
                next_f, next_g, next_node, _ = open_set[0]
                log.debug(f"  Next node to process: {next_node} with f={next_f}")

        if path is None:
            log.warning(f"  No path found from {start_node} to {end_node}")
            return None

        return self._create_final_path(path, start_point, end_point)

    def _get_neighbors(self, pos):
        x, y = pos
        # 4-directional movement for orthogonal routing
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.occupancy_map.nx and 0 <= ny < self.occupancy_map.ny:
                yield (nx, ny)

    def _heuristic(self, p1, p2):
        # Manhattan distance on the grid for orthogonal routing
        return (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) * self.grid_size

    def _reconstruct_path(self, came_from, end_node) -> LineString:
        # Reconstruct the path from end to start
        path_points = []
        current = end_node

        # Add the end point first (we'll reverse later)
        while current in came_from:
            x, y = current
            # Convert grid coordinates to world coordinates
            world_x = self.occupancy_map.minx + x * self.grid_size
            world_y = self.occupancy_map.miny + y * self.grid_size
            path_points.append((world_x, world_y))
            current = came_from[current]

        # Add the start node
        x, y = current
        world_x = self.occupancy_map.minx + (x) * self.grid_size
        world_y = self.occupancy_map.miny + (y) * self.grid_size
        path_points.append((world_x, world_y))

        path_points.reverse()

        # Ensure the path starts and ends at the actual pin locations
        # Get the start and end pin locations from the search context
        # We need to modify this to use the actual pin points
        # For now, we'll use the first and last points
        if path_points:
            # Replace first point with actual start pin location
            # Replace last point with actual end pin location
            # This requires access to the pin locations, which we don't have here
            # We'll need to pass them in or find another way
            pass

        return LineString(path_points)

    def get_center_of_pin_macro(self, start_point, start_pin, end_pin):
        if start_point == start_pin.draw.geom:
            pw = start_pin.instance.draw.geom.centroid
            parent = self.occupancy_map._world_to_grid(pw.x, pw.y)
        elif start_point == end_pin.draw.geom:
            pw = end_pin.instance.draw.geom.centroid
            parent = self.occupancy_map._world_to_grid(pw.x, pw.y)
        return parent

    def reroute(self):
        log.info("Starting A* router...")
        for module in self.db.design.modules.values():
            if module.is_leaf:
                continue

            log.info(f"Routing module: {module.name}")

            # Determine module bounds to include all pins and module geometry
            all_points = []
            if module.draw.geom is not None:
                minx, miny, maxx, maxy = module.draw.geom.bounds
                all_points.extend([(minx, miny), (maxx, maxy)])

            # Add all pin locations to the bounds
            for net in module.nets.values():
                for pin in net.pins.values():
                    if pin.draw.geom:
                        all_points.append((pin.draw.geom.x, pin.draw.geom.y))

            if not all_points:
                log.warning(f"Module {module.name} has no geometry or pins, cannot determine bounds. Skipping.")
                continue

            # Calculate bounds from all points
            xs, ys = zip(*all_points)
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)

            log.info(f"Calculated bounds including pins: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")
            # Expand bounds by 2x to ensure all points are within
            width = maxx - minx
            height = maxy - miny
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2

            # Expand the bounds to be twice as large
            new_width = width * 2
            new_height = height * 2
            minx = int(center_x - new_width / 2)
            maxx = int(center_x + new_width / 2)
            miny = int(center_y - new_height / 2)
            maxy = int(center_y + new_height / 2)

            # Add additional padding to be safe
            padding = self.grid_size * 20
            minx -= padding
            miny -= padding
            maxx += padding
            maxy += padding

            self.occupancy_map = OccupancyMap((minx, miny, maxx, maxy), self.grid_size)
            self.cost_estimator = CostEstimator(self.occupancy_map)
            log.info(f"Occupancy map grid size: {self.occupancy_map.nx} x {self.occupancy_map.ny}")
            log.info(f"Occupancy map bounds: ({minx}, {miny}) to ({maxx}, {maxy})")

            # Add blockages for all component instances
            for inst in module.instances.values():
                if inst.draw.geom:
                    self.occupancy_map.add_blockage(inst.draw.geom)
                    halo = inst.draw.geom.buffer(self.halo_size)
                    self.occupancy_map.add_halo(halo)

            # Don't add blockages for pins, as we want to be able to route to them
            # Instead, we'll handle pin access during the A* search by temporarily clearing their occupancy

            self.adjust_pin_locations(module)

            sorted_nets = sorted(
                [n for n in module.nets.values() if 2 <= n.num_conn < self.db.fanout_threshold],
                key=lambda net: net.num_conn,
            )

            os.makedirs("data/images/route", exist_ok=True)
            for i, net in enumerate(sorted_nets):
                log.info(f"Routing net {i + 1}/{len(sorted_nets)}: {net.name}")
                self.route_net(module, net)

            # Generate summary plots after routing all nets in the module
            summary_path = f"data/images/route/{module.name}_summary.png"
            occupancy_summary_path = f"data/images/route/{module.name}_occupancy.png"

            plot_routing_summary(module, sorted_nets, self.occupancy_map, summary_path)
            plot_occupancy_summary(module, self.occupancy_map, occupancy_summary_path)
        log.info("A* router finished.")

    def merge_routes(self, route_tree, tol=1):
        """
        Merge a list of route segments into a single continuous polyline where possible.
        Snaps near-touching endpoints before merging to avoid fragmented paths.
        """

        if not route_tree:
            return None

        #  Snap everything together
        snapped = [self._round_coords_to_grid(seg) for seg in route_tree]

        #  Union after snapping
        union_geom = unary_union(snapped)

        #  Linemerge on normalized input
        try:
            merged = linemerge(union_geom)
        except ValueError:
            merged = union_geom  # fallback

        #  Normalize result → MultiLineString
        if isinstance(merged, LineString):
            merged = self._remove_collinear_points(merged)
            # log.debug(f"L {route_tree=} -> {snapped=} {merged=}")
            return MultiLineString([merged])

        if isinstance(merged, MultiLineString):
            merged = [self._remove_collinear_points(ls) for ls in merged.geoms]
            # log.debug(f"M {route_tree=} -> {snapped=} {merged=}")
            return merged

        #  Final fallback: pick only the line parts
        lines = [g for g in merged.geoms if isinstance(g, (LineString, MultiLineString))]
        # log.debug(f"G {route_tree=} -> {snapped=} {merged=}")
        return MultiLineString(lines)

    def _remove_collinear_points(self, line: LineString) -> LineString:
        coords = list(line.coords)
        cleaned = [coords[0]]

        for i in range(1, len(coords) - 1):
            x1, y1 = cleaned[-1]
            x2, y2 = coords[i]
            x3, y3 = coords[i + 1]

            # Check if middle point is collinear with neighbors
            if (x1 == x2 == x3) or (y1 == y2 == y3):
                continue

            cleaned.append((x2, y2))

        cleaned.append(coords[-1])
        return LineString(cleaned)

    def _round_coords_to_grid(self, geom):
        """Return a new geometry with all coordinates rounded to nearest int."""
        if isinstance(geom, LineString):
            return LineString([(int(round(x)), int(round(y))) for x, y in geom.coords])

        if isinstance(geom, MultiLineString):
            return MultiLineString([LineString([(int(round(x)), int(round(y))) for x, y in line.coords]) for line in geom.geoms])

        if isinstance(geom, GeometryCollection):
            return GeometryCollection([self._round_coords_to_grid(g) for g in geom.geoms])

        # Passthrough anything unexpected
        return geom
