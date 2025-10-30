import logging as log
import sys
import time
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree

from schematic_from_netlist.database.netlist_structures import Module, Net
from schematic_from_netlist.sastar_router.models import CostBuckets, CostEstimator, PNet, RoutingContext
from schematic_from_netlist.sastar_router.sim_router import SimultaneousRouter
from schematic_from_netlist.sastar_router.test_cases import create_hard_test_case
from schematic_from_netlist.sastar_router.visualization import plot_result

log.basicConfig(level=log.INFO)


class AstarRouter:
    def __init__(self, db):
        self.db = db

    def route_design(self):
        log.info("Routing design using SAstar Router")
        for module in self.db.design.modules.values():
            obstacles = self.get_macro_geometries(module)
            routed_nets = self.route_nets(module.nets, obstacles)

    def route_nets(self, nets, obstacles):
        """
        Route multiple nets in order of span
        """
        # Sort nets by bbox diagonal (larger first)
        sorted_nets = sorted(nets.values(), key=lambda net: self.get_orthogonal_span(net), reverse=True)

        routed_nets = []
        for i, net in enumerate(sorted_nets):
            log.info(f"{i} : Routing net {net.name} with {len(net.pins)} pins")

            # Temporarily remove this net's existing paths for routing
            old_routed_paths = net.draw.geom
            # flatten list

            other_paths = [n.draw.geom for j, n in enumerate(sorted_nets) if j != i]
            old_total_cost, old_step_costs = self.calculate_existing_cost(net, old_routed_paths, obstacles, other_paths)

            # Calculate bounds from existing route
            router_halo = self.get_route_halo(old_routed_paths, net.pins, padding=1)
            # router_halo = self.get_preexisting_route_bounds(old_routed_paths)

            # Create router with bounds
            router = SimultaneousRouter(grid_spacing=1.0, obstacle_geoms=obstacles, halo_size=4, bounds=router_halo)
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
                net.draw.geom = new_routed_paths
                net.total_cost = new_total_cost
                net.step_costs = new_step_costs
            else:
                print(f"  New route cost {new_total_cost} > orig {old_total_cost} : reverting to old")
                net.draw.geom = old_routed_paths
                net.total_cost = old_total_cost
                net.step_costs = old_step_costs

            plot_result(net, obstacles, nets)

        # After all nets are routed, check for overlaps between nets
        self.check_for_overlaps(nets)

        """
        # Post-processing: Ensure all nets are connected
        for net in nets:
            if net.draw.geom:
                # Check if the net is connected
                merged = linemerge(net.draw.geom)
                if isinstance(merged, MultiLineString):
                    # The net is disconnected, try to connect the pieces
                    log.warning(f"Net {net.name} is disconnected?")

            # Clean up zero-length segments
            cleaned_paths = []
            for path in net.draw.geom:
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
            net.draw.geom = cleaned_paths
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

        router = SimultaneousRouter(grid_spacing=1.0, obstacle_geoms=obstacles, halo_size=4)
        cost_estimator = CostEstimator(grid_spacing=1.0)
        terminal_set = set([p.draw.geom for p in net.pins.values()])

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
        log.info(f"calculate_existing_cost existing path net {net.name}  {path=}")
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

            # Store cost at the neighbor node
            step_costs[neighbor_node] = move_cost

            total_cost += move_cost

            # Update parent for bend detection
            parent_node = current_node

        return total_cost, step_costs

    def get_preexisting_route_bounds(self, path):
        router_bounds = None
        padding = 5
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
                router_bounds = box(min(xs), min(ys), max(xs), max(ys))
        if not router_bounds:
            return None
        halo = router_bounds.buffer(padding, cap_style=2, join_style=2)
        return halo

    def get_route_halo(self, path, pins, padding=5):
        """
        Create a route-following halo (buffer) around an existing path geometry.
        """
        if not path:
            return None

        # Normalize input to a MultiLineString
        if isinstance(path, LineString):
            geom = path
        elif isinstance(path, MultiLineString):
            geom = unary_union(path)
        elif isinstance(path, (list, tuple)):
            geom = unary_union([p for p in path if isinstance(p, LineString)])
        else:
            raise TypeError(f"Unsupported geometry type: {type(path)}")

        # The route-following “band” (Shapely buffer)
        halo = geom.buffer(padding, cap_style=2, join_style=2)
        padding = 1  # need to pad the pad to prevent terminals from being omitted
        halo = halo.buffer(padding, cap_style=2, join_style=2)
        return halo

    def get_orthogonal_span(self, net):
        """Get the ortho length of the bounding box to estimate net length"""
        if not net.pins:
            return 0
        xs = [p.draw.geom.x for p in net.pins.values()]
        ys = [p.draw.geom.y for p in net.pins.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (max_x - min_x) + (max_y - min_y)

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
        for i, net1 in enumerate(nets.values()):
            for j, net2 in enumerate(nets.values()):
                if i < j:  # Only check each pair once
                    # Check for overlaps between net1 and net2
                    path1 = net1.draw.geom
                    path2 = net2.draw.geom
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

    def get_macro_geometries(self, module: Module):
        """Get all macro geometries in a module."""
        geoms = []
        for i in module.get_all_instances().values():
            if hasattr(i.draw, "geom") and i.draw.geom:
                if isinstance(i.draw.geom, Polygon):
                    geoms.append(i.draw.geom)
                elif isinstance(i.draw.geom, MultiPolygon):
                    geoms.extend(list(i.draw.geom.geoms))
        # return unary_union(geoms) if geoms else Polygon()
        return geoms if geoms else Polygon()

    def get_halo_geometries(self, macros, buffer_dist: int = 10) -> Polygon:
        """Get halo geometries around macros."""
        if macros.is_empty:
            return Polygon()
        return macros.buffer(buffer_dist)


if __name__ == "__main__":
    # Define nets
    log.basicConfig(level=log.INFO)
    # nets, obstacles = create_hard_test_case("macro_grid")
    nets, obstacles = create_hard_test_case("precision")

    # Route all nets
    db = []
    ar = AstarRouter(db)
    routed_nets = ar.route_nets(nets, obstacles)

    # Print results
    for net in routed_nets:
        if net.draw.geom:
            # Convert to MultiLineString to ensure consistent output format
            # Create a MultiLineString from all paths
            multi_line = MultiLineString(net.draw.geom)
            print(f"Net {net.name} geometry: {multi_line.wkt}")
        else:
            print(f"Net {net.name} failed to route")
