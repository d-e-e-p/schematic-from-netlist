import logging as log
import sys
import time
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from models import CostBuckets, CostEstimator, RoutingContext
from router import SimultaneousRouter
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree
from test_cases import create_hard_test_case
from visualization import plot_result

log.basicConfig(level=log.INFO)


class AstarRouter:
    def __init__(self, db):
        self.db = db

    def route_nets(self, nets, obstacles):
        """
        Route multiple nets in order of span
        """
        # Sort nets by bbox diagonal (larger first)
        nets.sort(key=lambda net: self.get_bbox_diagonal(net), reverse=True)

        routed_nets = []
        for i, net in enumerate(nets):
            log.info(f"{i} : Routing net {net.name} with {len(net.pins)} pins")

            # Temporarily remove this net's existing paths for routing
            old_routed_paths = net.routed_paths
            # flatten list

            other_paths = [n.routed_paths for j, n in enumerate(nets) if j != i]
            old_total_cost, old_step_costs = self.calculate_existing_cost(net, old_routed_paths, obstacles, other_paths)

            # Calculate bounds from existing route
            router_bounds = self.get_preexisting_route_bounds(old_routed_paths)

            # Create router with bounds
            router = SimultaneousRouter(grid_spacing=1.0, obstacle_geoms=obstacles, halo_size=4, bounds=router_bounds)
            new_routed_paths, new_total_cost = router.route_net(net, other_paths)

            # Calculate existing cost with other nets' paths
            new_total_cost, new_step_costs = self.calculate_existing_cost(net, new_routed_paths, obstacles, other_paths)

            if not new_routed_paths and not old_routed_paths:
                log.warning(f"Net {net.name} failed to route")
                return

            # Only update if new route is better
            log.info(f"Net {net.name} old paths {old_routed_paths} new paths {new_routed_paths}")
            if new_total_cost < old_total_cost:
                print(f"  New route cost {new_total_cost} < orig {old_total_cost} : keeping new route")
                net.routed_paths = new_routed_paths
                net.total_cost = new_total_cost
                net.step_costs = new_step_costs
            else:
                print(f"  New route cost {new_total_cost} > orig {old_total_cost} : reverting to old")
                net.routed_paths = old_routed_paths
                net.total_cost = old_total_cost
                net.step_costs = old_step_costs

            plot_result(net, obstacles, nets)

        # After all nets are routed, check for overlaps between nets
        self.check_for_overlaps(nets)

        """
        # Post-processing: Ensure all nets are connected
        for net in nets:
            if net.routed_paths:
                # Check if the net is connected
                merged = linemerge(net.routed_paths)
                if isinstance(merged, MultiLineString):
                    # The net is disconnected, try to connect the pieces
                    log.warning(f"Net {net.name} is disconnected?")

            # Clean up zero-length segments
            cleaned_paths = []
            for path in net.routed_paths:
                if isinstance(path, LineString):
                    # Check if it's a zero-length segment
                    if path.length > 1e-6:  # Use a small epsilon to account for floating point errors
                        cleaned_paths.append(path)
                elif isinstance(path, MultiLineString):
                    # For MultiLineString, check each segment
                    valid_segments = []
                    for segment in path.geoms:
                        if segment.length > 1e-6:
                            valid_segments.append(segment)
                    if valid_segments:
                        cleaned_paths.append(MultiLineString(valid_segments))

            # Replace with cleaned paths
            net.routed_paths = cleaned_paths
        """

        return nets

    def iter_lines(self, geom):
        """Yield LineStrings from a geometry or nested geometry structure."""

        # Single LineString
        if isinstance(geom, LineString):
            yield geom
            return

        # MultiLineString or GeometryCollection
        if isinstance(geom, (MultiLineString, GeometryCollection)):
            for g in geom.geoms:
                yield from self.iter_lines(g)
            return

        # List or tuple of geometries
        if isinstance(geom, Iterable) and not isinstance(geom, (str, bytes)):
            for g in geom:
                yield from self.iter_lines(g)
            return

        raise TypeError(f"Unsupported type in iter_lines: {type(geom)} → {geom!r}")

    def calculate_existing_cost(self, net, path, obstacles, existing_paths):
        """Calculate the cost of an existing route using the same cost estimation as routing"""
        # Create temporary router to get halo regions and context
        if not path:
            return np.inf, MultiLineString()

        pins = net.pins
        router = SimultaneousRouter(grid_spacing=1.0, obstacle_geoms=obstacles, halo_size=4)
        cost_estimator = CostEstimator(grid_spacing=1.0)
        snapped_pins = [router._snap_to_grid(p) for p in pins]
        terminal_set = set(snapped_pins)

        log.info(f"existing path net {net.name}  {existing_paths=}")
        other_paths = unary_union(existing_paths)
        other_paths_index = STRtree(list(other_paths.geoms)) if other_paths else None

        # Create routing context WITH existing nets
        context = RoutingContext(
            obstacles_index=router.obstacles_index,
            obstacles=obstacles,
            existing_paths_index=other_paths_index,
            existing_paths=other_paths,
            halo_geoms=router.halo_geoms,
            halo_index=router.halo_index,
            terminal_set=terminal_set,
        )

        total_cost = 0.0
        parent_node = None

        # Extract and interpolate all coordinates from all path segments
        log.info(f"existing path net {net.name}  {path=}")
        all_coords = []
        for line in self.iter_lines(path):
            coords = list(line.coords)

            # Interpolate between points to get all grid steps
            interpolated = []
            for i in range(len(coords) - 1):
                start = router._snap_to_grid(coords[i])
                end = router._snap_to_grid(coords[i + 1])

                # Generate intermediate grid points
                if start[0] == end[0]:  # Vertical move
                    step = 1 if end[1] > start[1] else -1
                    interpolated += [(start[0], y) for y in range(int(start[1]), int(end[1]) + step, step)]
                elif start[1] == end[1]:  # Horizontal move
                    step = 1 if end[0] > start[0] else -1
                    interpolated += [(x, start[1]) for x in range(int(start[0]), int(end[0]) + step, step)]
                else:  # Diagonal (shouldn't happen in our router)
                    interpolated.append(start)
                    interpolated.append(end)

            all_coords.extend(interpolated)

        log.info(f"Extract and interpolate all coordinates net {net.name}  {all_coords=}")
        # Remove duplicate consecutive points
        cleaned_coords = []
        prev = None
        for coord in all_coords:
            snapped = router._snap_to_grid(coord)
            if snapped != prev:
                cleaned_coords.append(snapped)
                prev = snapped

        log.info(f"Remove duplicate consecutive points net {net.name}  {cleaned_coords=}")
        # Reset step costs before calculation
        step_costs = {}

        # Process every consecutive pair of cleaned coordinates
        for i in range(len(cleaned_coords) - 1):
            current_node = cleaned_coords[i]
            neighbor_node = cleaned_coords[i + 1]

            # Skip zero-length moves
            if current_node == neighbor_node:
                continue

            # Skip cost calculation if moving into a terminal
            if neighbor_node in terminal_set:
                continue

            # Calculate move cost using the same logic as during routing
            move_cost = cost_estimator.get_move_cost(current_node, neighbor_node, parent_node, context)
            log.info(f"{net.name}: Cost from {current_node} to {neighbor_node}: {move_cost}")

            # Store cost at segment midpoint
            step_costs[neighbor_node] = move_cost

            total_cost += move_cost

            # Update parent for bend detection
            parent_node = current_node

        return total_cost, step_costs

    def get_preexisting_route_bounds(self, path):
        router_bounds = None
        if path:
            all_coords = []
            if isinstance(path, LineString):
                all_coords.extend(list(path.coords))
            elif isinstance(path, MultiLineString):
                for line in path.geoms:
                    all_coords.extend(list(line.coords))
            if all_coords:
                xs = [c[0] for c in all_coords]
                ys = [c[1] for c in all_coords]
                router_bounds = (min(xs), min(ys), max(xs), max(ys))
        return router_bounds

    def get_bbox_diagonal(self, net):
        """Get the diagonal length of the bounding box to estimate net length"""
        if not net.pins:
            return 0
        xs = [p[0] for p in net.pins]
        ys = [p[1] for p in net.pins]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5

    def intersection_overlap_length(self, intersection):
        """Return total length of geometry overlap from a Shapely intersection."""

        if intersection is None:
            return 0

        # Case 1: A numpy array of geometries (common with STRtree queries)
        if isinstance(intersection, np.ndarray):
            return sum(round(g.length) for g in intersection if isinstance(g, LineString) and not g.is_empty)

        # Case 2: A single geometry
        if hasattr(intersection, "is_empty") and not intersection.is_empty:
            if isinstance(intersection, LineString):
                return round(intersection.length)

        return 0

        # After all nets are routed, check for overlaps between nets

    def check_for_overlaps(self, nets):
        print("\nChecking for overlaps between nets:")
        total_overlap = 0
        for i, net1 in enumerate(nets):
            for j, net2 in enumerate(nets):
                if i < j:  # Only check each pair once
                    # Check for overlaps between net1 and net2
                    path1 = net1.routed_paths
                    path2 = net2.routed_paths
                    intersection = path1.intersection(path2)
                    overlap_length = self.intersection_overlap_length(intersection)
                    if overlap_length > 0:
                        total_overlap += overlap_length
                        log.warning(f"WARNING: Overlap between {net1.name} and {net2.name}, length: {overlap_length}")
                        log.warning(f"  Path1: {path1}")
                        log.warning(f"  Path2: {path2}")
                        log.warning(f"  Intersection: {intersection}")

        if total_overlap > 0:
            log.warning(f"\nTotal overlap length: {total_overlap}")


if __name__ == "__main__":
    # Define nets
    # nets, obstacles = create_hard_test_case("macro_grid")
    log.basicConfig(level=log.INFO)
    nets, obstacles = create_hard_test_case("precision")

    # Route all nets
    db = []
    ar = AstarRouter(db)
    routed_nets = ar.route_nets(nets, obstacles)

    # Print results
    for net in routed_nets:
        if net.routed_paths:
            # Convert to MultiLineString to ensure consistent output format
            # Create a MultiLineString from all paths
            multi_line = MultiLineString(net.routed_paths)
            print(f"Net {net.name} geometry: {multi_line.wkt}")
        else:
            print(f"Net {net.name} failed to route")
