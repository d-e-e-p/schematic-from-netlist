# Pattern Route-based global router

from __future__ import annotations

import itertools
import logging as log
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rtree import index
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, nearest_points, snap, unary_union
from shapely.strtree import STRtree

from schematic_from_netlist.database.netlist_structures import Module, Net, Pin
from schematic_from_netlist.global_router.gr_candidate_paths import (
    generate_candidate_paths,
    get_halo_geometries,
    get_l_path_corner,
    get_macro_geometries,
)
from schematic_from_netlist.global_router.gr_cost_calculator import CostCalculator
from schematic_from_netlist.global_router.gr_debug import RouterDebugger
from schematic_from_netlist.global_router.gr_structures import Junction, RoutingContext, Topology

# Pattern Route parameters
ROUTE_WEIGHTS = {"wirelength": 1.0, "congestion": 2.0, "halo": 5.0, "crossing": 5.0, "macro": 1000.0, "track": 20.0}
JUNCTION_SPACING_PENALTY = 2000.0  # Cost for placing junctions too close to each other
MAX_PATH_COST = 100.0  # Mean path cost multiplier for pruning
GRID_SPACING = 1.0  # Track spacing for snapping
MAX_FANOUT = 15  # Maximum junction fanout


def test_for_int_list_of_points(location_list, tol=1e-9):
    """Return True if all Points in list have integer coordinates (within tolerance)."""
    for pt in location_list:
        if not isinstance(pt, Point):
            return False  # not a Point
        x_int = abs(pt.x - round(pt.x)) < tol
        y_int = abs(pt.y - round(pt.y)) < tol
        if not (x_int and y_int):
            return False
    return True


class GlobalRouter:
    def __init__(self, db):
        self.db = db
        self.junctions: Dict[Module, List[Topology]] = defaultdict(list)
        self._debugger = RouterDebugger()
        self._cost_calculator = CostCalculator(self._debugger)

    def create_routing_context(
        self,
        topo: Topology,
    ):
        """Create and attach a RoutingContext to the topology."""
        # For pins inside macros, route to the macro center
        pin_macros: Dict[Pin, Polygon] = {}
        pins = [p for p in topo.net.pins.values() if hasattr(p.draw, "geom") and p.draw.geom is not None]
        for p in pins:
            if macro := p.instance.draw.geom:
                pin_macros[p] = macro
                p.draw.geom = Point(round(macro.centroid.x), round(macro.centroid.y))
            else:
                log.warning(f"Pin {p.name} instance {p.instance.name} has no geom .")

        topo.context = RoutingContext()
        module = topo.net.module
        topo.context.macros = get_macro_geometries(module)
        topo.context.halos = get_halo_geometries(topo.context.macros)
        topo.context.module = module
        topo.context.pin_macros = pin_macros
        self.update_routing_context(topo)

    def update_routing_context(
        self,
        topo: Topology,
    ):
        """Create and attach a RoutingContext to the topology."""
        net = topo.net
        congestion_idx, other_nets_geoms = self._build_congestion_index(net.module, net)

        h_tracks, v_tracks = self._build_track_occupancy(other_nets_geoms)
        h_tracks, v_tracks = self.remove_macros_from_tracks(h_tracks, v_tracks, topo.context.pin_macros)

        topo.context.net = net
        topo.context.congestion_idx = congestion_idx
        topo.context.other_nets_geoms = other_nets_geoms
        topo.context.h_tracks = h_tracks
        topo.context.v_tracks = v_tracks

    def insert_routing_junctions(self):
        # Process groups
        for module in self.db.design.modules.values():
            if module.is_leaf:
                continue

            log.info(f"Processing module {module.name} with {len(module.nets)} nets in stages.")

            # --- Pre-computation for the whole module ---
            macros = get_macro_geometries(module)
            halos = get_halo_geometries(macros)
            sorted_nets = sorted(
                [n for n in module.nets.values() if 2 < n.num_conn < self.db.fanout_threshold],
                key=lambda net: net.num_conn,
            )

            # --- STAGE 1
            log.info(f"--- Stage 1: Initial Topology Generation for {len(sorted_nets)} nets ---")
            for net in sorted_nets:
                net.draw.geom = None  # Clear existing geometry
                topo = Topology(net=net)
                self.create_routing_context(topo)
                net.draw.topo = topo
                self.create_initial_junctions(topo)
                self._rebuild_net_geometry(topo)  # Update geometry for next net
            self._debugger.plot_junction_summary(module, stage="1", title="Initial Topology")

            # --- STAGE 2
            log.info(f"--- Stage 2: Pruning Junctions ---")
            for net in sorted_nets:
                if hasattr(net.draw, "topo") and net.draw.topo:
                    topo = net.draw.topo
                    self._prune_redundant_junctions(topo)
                    self.update_routing_context(topo)
                    self._rebuild_net_geometry(topo)

            self._debugger.plot_junction_summary(module, stage="2", title="Pruning Junctions")
            # --- STAGE 3
            log.info(f"--- Stage 3: Optimizing Junction Locations ---")
            for net in sorted_nets:
                if hasattr(net.draw, "topo") and net.draw.topo:
                    topo = net.draw.topo
                    self._optimize_junction_locations(topo)
                    self.update_routing_context(topo)
                    self._rebuild_net_geometry(topo)

            self._debugger.plot_junction_summary(module, stage="3", title="Opt junc loc")
            # --- STAGE 4
            log.info("--- Stage 4: Global Search by jumping junctions over macros ---")
            num_iterations = 3
            for _ in range(num_iterations):
                for net in sorted_nets:
                    if hasattr(net.draw, "topo") and net.draw.topo:
                        topo = net.draw.topo
                        self.update_routing_context(topo)
                        self._jump_junctions_over_macros(
                            topo,
                            topo.context,
                        )
                        self._rebuild_net_geometry(topo)

            self._debugger.plot_junction_summary(module, stage="4", title="jumping")
            # --- STAGE 5
            log.info(f"--- Stage 5: Local search: sliding junctions around a bit ---")
            for net in sorted_nets:
                if hasattr(net.draw, "topo") and net.draw.topo:
                    topo = net.draw.topo
                    self.update_routing_context(topo)
                    self._slide_junctions(
                        topo,
                    )
                    self._rebuild_net_geometry(topo)

            self._debugger.plot_junction_summary(module, stage="5", title="sliding")
            # --- STAGE 6
            log.info(f"--- Stage 6: Finalizing Routes and Pin Locations ---")
            for net in sorted_nets:
                if hasattr(net.draw, "topo") and net.draw.topo:
                    topo = net.draw.topo
                    self._finalize_routes_and_pin_locations(topo)
                    self.junctions[module].append(topo)

        log.info(f"Total junctions created: {sum(len(v) for v in self.junctions.values())}")

        # Log detailed junction summary
        self._debugger.log_junction_summary(self.junctions)
        for module in self.junctions:
            self._debugger.plot_junction_summary(module, stage="6", title="final")

        for module in self.db.design.modules.values():
            self._diagnose_crossings(module)

        return self.junctions

    def _build_congestion_index(self, module: Module, current_net: Net) -> Tuple[index.Index, List[LineString]]:
        """Build an R-tree index for congestion analysis."""
        congestion_idx = index.Index()
        other_nets_geoms = []
        other_nets = [n for n in module.nets.values() if n is not current_net and hasattr(n.draw, "geom") and n.draw.geom]

        i = 0
        for net in other_nets:
            geom = net.draw.geom
            if isinstance(geom, LineString):
                congestion_idx.insert(i, geom.bounds)
                other_nets_geoms.append(geom)
                i += 1
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    congestion_idx.insert(i, line.bounds)
                    other_nets_geoms.append(line)
                    i += 1
        return congestion_idx, other_nets_geoms

    def _build_track_occupancy(
        self, geoms: List[LineString]
    ) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
        """
        Builds track occupancy maps, rounding all coordinates to the nearest integer.
        """
        h_tracks = defaultdict(list)
        v_tracks = defaultdict(list)
        tolerance = 1e-9  # Define tolerance for comparisons

        for line in geoms:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = Point(coords[i])
                p2 = Point(coords[i + 1])

                # *** ADDED CHECK ***
                # Skip zero-length segments caused by repeated coordinates
                if p1.distance(p2) < tolerance:
                    continue
                # *** END OF ADDED CHECK ***

                # Use a tolerance for floating point comparisons
                if abs(p1.y - p2.y) < tolerance:  # Horizontal
                    # Round all values to the nearest integer
                    key_y = int(round(p1.y))
                    val_x1 = int(round(p1.x))
                    val_x2 = int(round(p2.x))
                    h_tracks[key_y].append(tuple(sorted((val_x1, val_x2))))

                elif abs(p1.x - p2.x) < tolerance:  # Vertical
                    # Round all values to the nearest integer
                    key_x = int(round(p1.x))
                    val_y1 = int(round(p1.y))
                    val_y2 = int(round(p2.y))
                    v_tracks[key_x].append(tuple(sorted((val_y1, val_y2))))

        # Helper to merge intervals (expect ints)
        def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            if not intervals:
                return []
            intervals.sort()
            merged = [intervals[0]]
            for current in intervals[1:]:
                last = merged[-1]
                if current[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged.append(current)
            return merged

        # Merge intervals for each track
        for y in h_tracks:
            h_tracks[y] = merge_intervals(h_tracks[y])
        for x in v_tracks:
            v_tracks[x] = merge_intervals(v_tracks[x])

        return h_tracks, v_tracks

    def create_initial_junctions(
        self,
        topo: Topology,
    ):
        """Creates the initial MST-based topology for a single net."""

        log.debug(f"Creating initial topology for net {topo.net.name} ")

        # Initialize MST algorithm (Prim's)
        pins = [p for p in topo.net.pins.values() if hasattr(p.draw, "geom") and p.draw.geom is not None]
        unconnected_pins = set(pins)
        start_pin = unconnected_pins.pop()
        tree_connection_points: Set[Pin | Junction] = {start_pin}

        # Greedy merge loop
        while unconnected_pins:
            min_cost = float("inf")
            best_path = None
            best_new_pin = None
            best_connection_point = None

            for new_pin in unconnected_pins:
                for conn_point in tree_connection_points:
                    p1 = conn_point.draw.geom if isinstance(conn_point, Pin) else conn_point.location
                    p2 = new_pin.draw.geom

                    candidate_paths = generate_candidate_paths(p1, p2, topo.context)
                    for path_geom in candidate_paths:
                        p1_macro = topo.context.pin_macros.get(conn_point) if isinstance(conn_point, Pin) else None
                        p2_macro = topo.context.pin_macros.get(new_pin)
                        metrics, _ = self._cost_calculator.calculate_path_cost(
                            path_geom,
                            topo.context,
                            p1_macro,
                            p2_macro,
                        )
                        if metrics.total_cost < min_cost:
                            min_cost = metrics.total_cost
                            best_path = path_geom
                            best_new_pin = new_pin
                            best_connection_point = conn_point

            if best_path is None:
                log.warning(f"Could not find a path for net {topo.net.name}, skipping initial topology.")
                return

            # Add new pin and junction to the tree
            if best_new_pin:
                unconnected_pins.remove(best_new_pin)
            corner = get_l_path_corner(best_path)
            junction_name = f"J_{topo.net.name}_{len(topo.junctions)}"
            # Ensure we only add valid Pin or Junction objects to children
            children = set()
            if best_new_pin is not None:
                children.add(best_new_pin)
            if best_connection_point is not None:
                children.add(best_connection_point)
            junction = Junction(name=junction_name, location=corner, children=children)
            topo.junctions.append(junction)
            tree_connection_points.add(junction)

            if isinstance(best_connection_point, Pin):
                tree_connection_points.remove(best_connection_point)

    def _rebuild_net_geometry(self, topology: Topology):
        """Rebuilds the net's geometry based on its current topology of junctions and pins."""
        context = topology.context

        def rounded_point(pt):
            return Point(round(pt[0]), round(pt[1]))

        new_geoms = []
        for junction in topology.junctions:
            start_point = junction.location

            cached_paths = {}
            if junction.geom:
                for line in junction.geom.geoms:
                    p1 = rounded_point(line.coords[0])
                    p2 = rounded_point(line.coords[-1])
                    # Using almost_equals for robust float comparisons
                    if p1.distance(start_point) < 1e-6:
                        cached_paths[p2] = line
                    elif p2.distance(start_point) < 1e-6:
                        cached_paths[p1] = line

            junction_paths = []
            for child in junction.children:
                if isinstance(child, Pin):
                    end_point = child.draw.geom
                    if end_point is None:
                        continue
                    if not isinstance(end_point, Point):
                        end_point = end_point.centroid
                else:  # Junction
                    end_point = child.location

                best_path = None
                # Check for a cached path
                cached_end_point_found = None
                for cached_end_point in cached_paths:
                    if cached_end_point.distance(end_point) < 1e-6:
                        cached_end_point_found = cached_end_point
                        break

                if cached_end_point_found:
                    best_path = cached_paths[cached_end_point_found]

                if best_path is None:
                    # If not cached, compute, select best, and cache
                    candidate_paths = generate_candidate_paths(start_point, end_point, context)

                    min_cost = float("inf")
                    p1_macro_to_ignore = None  # Junctions are not in macros
                    p2_macro_to_ignore = context.pin_macros.get(child) if isinstance(child, Pin) else None

                    if not candidate_paths:
                        if start_point.distance(end_point) > 1e-9:
                            best_path = LineString([start_point, end_point])
                    else:
                        for path in candidate_paths:
                            metrics, _ = self._cost_calculator.calculate_path_cost(
                                path,
                                context,
                                p1_macro_to_ignore,
                                p2_macro_to_ignore,
                            )
                            if metrics.total_cost < min_cost:
                                min_cost = metrics.total_cost
                                best_path = path

                if best_path:
                    new_geoms.append(best_path)
                    junction_paths.append(best_path)

            if junction_paths:
                junction.geom = MultiLineString(junction_paths)
            else:
                junction.geom = None

        if new_geoms:
            union_geom = unary_union(new_geoms)
            merged_geom = None
            if isinstance(union_geom, LineString):
                merged_geom = linemerge([union_geom])
            elif isinstance(union_geom, MultiLineString):
                merged_geom = linemerge(union_geom)
            if isinstance(merged_geom, LineString):
                topology.net.draw.geom = MultiLineString([merged_geom])
            elif isinstance(merged_geom, MultiLineString):
                topology.net.draw.geom = merged_geom
            else:
                topology.net.draw.geom = MultiLineString([g for g in new_geoms if isinstance(g, LineString)])
        else:
            topology.net.draw.geom = None

    def _prune_redundant_junctions(self, topology: Topology):
        """Remove junctions with degree <= 2, as they are redundant."""
        while True:
            adj = defaultdict(set)
            for j in topology.junctions:
                for child in j.children:
                    adj[j].add(child)
                    adj[child].add(j)

            junction_to_prune = None
            for j in topology.junctions:
                if len(adj[j]) <= 2:
                    junction_to_prune = j
                    break

            if not junction_to_prune:
                break  # No more junctions to prune

            neighbors = list(adj[junction_to_prune])
            if len(neighbors) == 2:
                n1, n2 = neighbors[0], neighbors[1]
                # Remove old edges from neighbors' children sets
                if isinstance(n1, Junction) and junction_to_prune in n1.children:
                    n1.children.remove(junction_to_prune)
                if isinstance(n2, Junction) and junction_to_prune in n2.children:
                    n2.children.remove(junction_to_prune)

                # Add new edge between neighbors
                if isinstance(n1, Junction):
                    n1.children.add(n2)
                if isinstance(n2, Junction):
                    n2.children.add(n1)

            elif len(neighbors) == 1:
                n1 = neighbors[0]
                if isinstance(n1, Junction) and junction_to_prune in n1.children:
                    n1.children.remove(junction_to_prune)

            topology.junctions.remove(junction_to_prune)

    def _optimize_junction_locations(
        self,
        topology: Topology,
        passes: int = 3,
    ):
        """Iteratively improve junction locations based on a weighted median of their connections."""
        # Build parent map
        context = topology.context
        parent_map: Dict[Junction, Junction] = {}
        for p in topology.junctions:
            for c in p.children:
                if isinstance(c, Junction):
                    parent_map[c] = p

        for _ in range(passes):
            # Iterate from leaves to root (approximately, based on construction order)
            for j in topology.junctions:
                coords_x = []
                coords_y = []

                # Children connections
                for child in j.children:
                    if isinstance(child, Pin):
                        p = child.draw.geom
                        if p is None:
                            continue
                        if not isinstance(p, Point):
                            p = p.centroid
                        # Higher weight for pins
                        coords_x.extend([p.x] * 2)
                        coords_y.extend([p.y] * 2)
                    elif isinstance(child, Junction):
                        p = child.location
                        coords_x.append(p.x)
                        coords_y.append(p.y)

                # Parent connection
                if j in parent_map:
                    parent = parent_map[j]
                    p = parent.location
                    coords_x.append(p.x)
                    coords_y.append(p.y)

                if coords_x:
                    # Update location to the median of connection points
                    new_loc = Point(int(round(np.median(coords_x))), int(round((np.median(coords_y)))))

                    # If the ideal location is restricted, find the nearest valid point on the boundary
                    if new_loc.within(context.macros) or new_loc.within(context.halos):
                        restricted_area = unary_union([context.macros, context.halos])
                        if restricted_area.boundary and not restricted_area.boundary.is_empty:
                            new_loc = nearest_points(new_loc, restricted_area.boundary)[1]

                    j.location = new_loc
                    j.geom = None
                    if j in parent_map and parent_map[j]:
                        parent_map[j].geom = None

    def _slide_junctions(
        self,
        topology: Topology,
    ):
        """
        Post-processing step to slide each junction in all directions to see if cost is reduced.
        This is run iteratively:
        1. In each iteration, find the best move for every junction based on the current topology.
        2. After checking all junctions, apply all the beneficial moves simultaneously.
        3. Re-plot the routing and repeat until no more cost reduction is possible.
        """
        if not topology.net:
            return
        search_step_size = 4
        log.debug(f"Starting junction sliding for net {topology.net.name}")

        context = topology.context
        macros = context.macros
        halos = context.halos
        module = context.module

        # Build an STRtree of all junctions from OTHER nets for quick spatial queries
        other_junction_points = []
        for net in module.nets.values():
            if net is not topology.net and hasattr(net.draw, "topo") and net.draw.topo:
                for j in net.draw.topo.junctions:
                    other_junction_points.append(j.location)
        junction_tree = STRtree(other_junction_points) if other_junction_points else None

        parent_map = self._build_junction_parent_map(topology)
        max_iterations = 10  # To prevent infinite loops
        all_tried_locations: Dict[Junction, Dict[Tuple[int, int], float]] = defaultdict(dict)
        for i in range(max_iterations):
            proposed_moves = {}
            cost_reduced_this_iter = False

            for j_idx, junction in enumerate(topology.junctions):
                initial_location = junction.location
                context.net = topology.net
                min_cost = self._cost_calculator.calculate_total_cost(
                    topology,
                    context,
                )
                best_location = initial_location
                tried_locations_costs = {(int(round(initial_location.x)), int(round(initial_location.y))): min_cost}

                # --- Calculate dynamic search step size ---
                connected_points = []
                adj = defaultdict(set)
                for j in topology.junctions:
                    for child in j.children:
                        adj[j].add(child)
                        adj[child].add(j)
                for neighbor in adj[junction]:
                    geom = neighbor.draw.geom if isinstance(neighbor, Pin) else neighbor.location
                    if isinstance(geom, Point):
                        connected_points.append(geom)
                    elif geom:
                        connected_points.append(geom.centroid)

                if connected_points:
                    x_coords = [p.x for p in connected_points]
                    y_coords = [p.y for p in connected_points]
                    span_x = max(x_coords) - min(x_coords) if x_coords else 0
                    span_y = max(y_coords) - min(y_coords) if y_coords else 0
                    span = max(span_x, span_y)
                    search_step_size = max(search_step_size, int(span / 10))  # larger step size for larger spans
                # ---

                # Explore surrounding spots in a wider range
                search_range = 2 * search_step_size
                for dx in range(-search_range, search_range + 1, search_step_size):
                    for dy in range(-search_range, search_range + 1, search_step_size):
                        if dx == 0 and dy == 0:
                            continue
                        new_location = Point(initial_location.x + dx, initial_location.y + dy)
                        if topology.net.name == "n_sck":
                            log.info(f" testing new location {new_location} for {junction.name}")

                        if new_location.within(macros) or new_location.within(halos):
                            tried_locations_costs[(int(round(new_location.x)), int(round(new_location.y)))] = float("inf")
                            continue

                        junction.location = new_location
                        current_cost = self._cost_calculator.calculate_total_cost(
                            topology,
                            context,
                        )
                        if topology.net.name == "n6":  # pyright: ignore
                            current_cost = self._cost_calculator.calculate_total_cost(
                                topology, context, True, f"debug_{topology.net.name}_{j_idx}_iter_{i}_{int(current_cost)}_"
                            )

                        # Add junction spacing penalty
                        if junction_tree:
                            nearby_junctions = junction_tree.query(new_location.buffer(4))
                            current_cost += JUNCTION_SPACING_PENALTY * len(nearby_junctions)

                        tried_locations_costs[(int(round(new_location.x)), int(round(new_location.y)))] = current_cost

                        if current_cost < min_cost:
                            min_cost = current_cost
                            best_location = new_location

                # Restore original location for next junction's calculation
                junction.location = initial_location

                # Accumulate tried locations and costs
                for loc, cost in tried_locations_costs.items():
                    all_tried_locations[junction][loc] = cost

                if best_location != initial_location:
                    proposed_moves[junction] = best_location
                    cost_reduced_this_iter = True

            if not cost_reduced_this_iter:
                log.debug(f"Junction sliding for net {topology.net.name} converged after {i} iterations.")
                break

            # Apply all proposed moves for this iteration
            for junction, new_location in proposed_moves.items():
                junction.location = new_location
                junction.geom = None
                if junction in parent_map and parent_map[junction]:
                    parent_map[junction].geom = None
                log.debug(f"  Moved junction {junction.name} to {new_location}")

            # Generate intermediate cost calculation plot
            log.info(f"calculating cost and dumping to prefix slide_iter_{i}_ for module:{module.name} net:{topology.net.name}")
            self._cost_calculator.calculate_total_cost(
                topology,
                context,
                debug_plot=True,
                plot_filename_prefix=f"slide_iter_{i}_",
            )

        else:
            log.warning(f"Junction sliding for net {topology.net.name} did not converge after {max_iterations} iterations.")

        # After all iterations, generate a summary heatmap for each junction
        for junction, tried_locations in all_tried_locations.items():
            if len(tried_locations) > 1:
                best_loc_tuple = min(tried_locations.keys(), key=lambda k: tried_locations[k])
                best_location = Point(best_loc_tuple)
                min_cost = tried_locations[best_loc_tuple]
                self._debugger._plot_junction_move_heatmap(
                    module,
                    topology,
                    junction,
                    tried_locations,
                    best_location,
                    min_cost,
                    macros,
                    halos,
                    filename_prefix="slide_summary_",
                )

    def _get_dominant_pin_direction(self, j_loc: Point, pin_locations: List[Point]) -> Optional[str]:
        """Calculates the dominant direction of pins relative to a junction."""
        if not pin_locations:
            return None

        vectors = np.array([(p.x - j_loc.x, p.y - j_loc.y) for p in pin_locations])
        tolerance = 1e-6
        avg_vector = np.mean(vectors, axis=0)

        if abs(avg_vector[0]) > abs(avg_vector[1]):  # Horizontal
            if avg_vector[0] < 0 and np.all(vectors[:, 0] < tolerance):
                return "W"
            elif avg_vector[0] > 0 and np.all(vectors[:, 0] > -tolerance):
                return "E"
        else:  # Vertical
            if avg_vector[1] > 0 and np.all(vectors[:, 1] > -tolerance):
                return "N"
            elif avg_vector[1] < 0 and np.all(vectors[:, 1] < tolerance):
                return "S"
        return None

    def _jump_junctions_over_macros(
        self,
        topology: Topology,
        context: RoutingContext,
    ):
        """
        Tries to find better junction locations by jumping over macros.
        This is for cases where a junction is stuck on one side of a macro due to a halo,
        while all its connected pins are on the same side.
        """
        macros = context.macros
        halos = context.halos
        module = context.module
        log.info(f"Starting junction jumping for module:{module.name} net:{topology.net.name}")

        parent_map = self._build_junction_parent_map(topology)
        memo_downstream_pins = {}
        all_tried_locations: Dict[Junction, Dict[Tuple[int, int], float]] = defaultdict(dict)

        max_iterations = 10
        for i in range(max_iterations):
            # 1. Pre-compute candidate locations for all junctions
            junction_candidate_locations: Dict[Junction, List[Point]] = {}
            for junction in topology.junctions:
                initial_location = junction.location
                context.net = topology.net
                downstream_pins = self._get_downstream_pins(junction, parent_map, memo_downstream_pins)
                if not downstream_pins:
                    candidate_locations = [initial_location]
                else:
                    candidate_locations = self._find_jump_candidate_locations(initial_location, downstream_pins, macros, halos)
                # Also include the initial location as a candidate
                is_initial_loc_present = any(loc.distance(initial_location) < 1e-9 for loc in candidate_locations)
                if not is_initial_loc_present:
                    candidate_locations.append(initial_location)
                junction_candidate_locations[junction] = candidate_locations

            # 2. Test out the topo cost for all combinations of positions
            junctions_to_optimize = topology.junctions
            num_junctions = len(junctions_to_optimize)
            location_lists = [junction_candidate_locations[j] for j in junctions_to_optimize]
            original_locations = {j: j.location for j in junctions_to_optimize}

            min_cost = self._cost_calculator.calculate_total_cost(topology, context)
            best_combination = original_locations.copy()
            cost_reduced_this_iter = False
            proposed_moves = {}

            if num_junctions <= 4:
                num_combinations = np.prod([len(l) for l in location_lists]) if location_lists else 0
                if num_combinations > 100_000:
                    log.warning(
                        f"For net {topology.net.name}, skipping combinatorial optimization in jump phase due to too many combinations: {num_combinations}"
                    )
                else:
                    min_cost_for_loc: Dict[Tuple[Junction, Tuple[int, int]], float] = defaultdict(lambda: float("inf"))

                    for location_combination in itertools.product(*location_lists):
                        current_combination = {j: loc for j, loc in zip(junctions_to_optimize, location_combination)}
                        # Apply new locations
                        for junction, new_loc in current_combination.items():
                            junction.location = new_loc

                        current_cost = self._cost_calculator.calculate_total_cost(topology, context)

                        if current_cost < min_cost:
                            min_cost = current_cost
                            best_combination = current_combination

                        # For heatmap data
                        for junction, new_loc in current_combination.items():
                            loc_tuple = (int(round(new_loc.x)), int(round(new_loc.y)))
                            if current_cost < min_cost_for_loc[(junction, loc_tuple)]:
                                min_cost_for_loc[(junction, loc_tuple)] = current_cost

                    # After iterating through combinations, populate all_tried_locations
                    for (junction, loc_tuple), cost in min_cost_for_loc.items():
                        all_tried_locations[junction][loc_tuple] = cost
            else:  # num_junctions > 4, optimize one at a time
                min_cost_for_loc: Dict[Tuple[Junction, Tuple[int, int]], float] = defaultdict(lambda: float("inf"))
                new_best_combination = {}
                for j_to_opt in junctions_to_optimize:
                    # Restore all other junctions to original locations for a clean test
                    for j, loc in original_locations.items():
                        if j != j_to_opt:
                            j.location = loc

                    best_loc_for_j = original_locations[j_to_opt]
                    # Set all junctions to original locations to get baseline cost for this sub-problem
                    for j, loc in original_locations.items():
                        j.location = loc
                    min_cost_for_j = self._cost_calculator.calculate_total_cost(topology, context)

                    for new_loc in junction_candidate_locations[j_to_opt]:
                        j_to_opt.location = new_loc
                        current_cost = self._cost_calculator.calculate_total_cost(topology, context)

                        loc_tuple = (int(round(new_loc.x)), int(round(new_loc.y)))
                        if current_cost < min_cost_for_loc[(j_to_opt, loc_tuple)]:
                            min_cost_for_loc[(j_to_opt, loc_tuple)] = current_cost

                        if current_cost < min_cost_for_j:
                            min_cost_for_j = current_cost
                            best_loc_for_j = new_loc

                    new_best_combination[j_to_opt] = best_loc_for_j

                # Check if this new combination is better than original
                for j, loc in new_best_combination.items():
                    j.location = loc

                final_cost = self._cost_calculator.calculate_total_cost(topology, context)
                if final_cost < min_cost:
                    min_cost = final_cost
                    best_combination = new_best_combination

                # After iterating, populate all_tried_locations
                for (junction, loc_tuple), cost in min_cost_for_loc.items():
                    all_tried_locations[junction][loc_tuple] = cost

            # Restore original locations before determining moves
            for junction, loc in original_locations.items():
                junction.location = loc

            # Determine proposed moves from the best combination
            for junction, best_loc in best_combination.items():
                if best_loc.distance(original_locations[junction]) > 1e-9:
                    proposed_moves[junction] = best_loc
                    cost_reduced_this_iter = True

            self._log_jump_iteration_results(i, topology, context, module, cost_reduced_this_iter, proposed_moves, parent_map)

            if not cost_reduced_this_iter:
                break
        else:
            log.warning(f"Junction jumping for net {topology.net.name} did not converge after {max_iterations} iterations.")

        # After all iterations, generate a summary heatmap for each junction
        for junction, tried_locations in all_tried_locations.items():
            if tried_locations:
                best_loc_tuple = min(tried_locations.keys(), key=lambda k: tried_locations[k])
                best_location = Point(best_loc_tuple)
                min_cost = tried_locations[best_loc_tuple]
                self._debugger._plot_junction_move_heatmap(
                    topology.context.module,
                    topology,
                    junction,
                    tried_locations,
                    best_location,
                    min_cost,
                    topology.context.macros,
                    topology.context.halos,
                    filename_prefix="jump_summary_",
                )

    def _build_junction_parent_map(self, topology: Topology) -> Dict[Junction, Optional[Junction]]:
        """Build a mapping of junctions to their parent junctions."""
        parent_map: Dict[Junction, Optional[Junction]] = {}
        if topology.junctions:
            root = topology.junctions[0]
            q = [root]
            visited = {root}
            parent_map[root] = None
            while q:
                p = q.pop(0)
                for c_obj in p.children:
                    if isinstance(c_obj, Junction) and c_obj not in visited:
                        visited.add(c_obj)
                        parent_map[c_obj] = p
                        q.append(c_obj)
        return parent_map

    def _get_downstream_pins(
        self, junction: Junction, parent_map: Dict[Junction, Optional[Junction]], memo_downstream_pins: Dict[Junction, List[Pin]]
    ) -> List[Pin]:
        """Get all pins downstream of a junction."""
        if junction in memo_downstream_pins:
            return memo_downstream_pins[junction]

        pins = []
        q = list(junction.children)
        visited = {junction, parent_map.get(junction)} if parent_map.get(junction) else {junction}

        while q:
            node = q.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if isinstance(node, Pin):
                pins.append(node)
            elif isinstance(node, Junction):
                q.extend(list(node.children))

        memo_downstream_pins[junction] = pins
        return pins

    def _find_jump_candidate_locations(
        self,
        junction_loc: Point,
        downstream_pins: List[Pin],
        macros: BaseGeometry,
        halos: BaseGeometry,
    ) -> List[Point]:
        """
        Find candidate locations by starting at the centroid of downstream pins
        and sliding away from the original junction location to find valid spots.
        """

        def rounded_point(x, y):
            return Point(round(x), round(y))

        def rounded_Point(pt):
            return Point(round(pt.x), round(pt.y))

        pin_locations = [p.draw.geom for p in downstream_pins if p.draw and p.draw.geom]
        pin_locations = [p.centroid if not isinstance(p, Point) else p for p in pin_locations]
        if not test_for_int_list_of_points(pin_locations):
            log.warning(f"not int list of points: {pin_locations=}")
        if not pin_locations:
            return []

        # 1. Calculate centroid of downstream pins
        centroid_x = np.mean([p.x for p in pin_locations])
        centroid_y = np.mean([p.y for p in pin_locations])
        centroid = rounded_point(centroid_x, centroid_y)

        restricted_area = unary_union([macros, halos])

        # 2. Define a ray from the centroid going in the opposite direction of the junction.
        if centroid.distance(junction_loc) < 1e-6:
            return [centroid] if not centroid.within(restricted_area) else []

        direction_vector = np.array([centroid.x - junction_loc.x, centroid.y - junction_loc.y])
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return [centroid] if not centroid.within(restricted_area) else []
        direction_vector /= norm

        # A large number to ensure the ray extends beyond all geometry.
        ray_length = 10000.0
        ray_end = rounded_point(centroid.x + direction_vector[0] * ray_length, centroid.y + direction_vector[1] * ray_length)
        ray = LineString([centroid, ray_end])

        # 3. Find the parts of the ray outside the restricted area.
        valid_line_parts = ray.difference(restricted_area)

        if valid_line_parts.is_empty:
            return []  # The ray never escapes the restricted area.

        # 4. Find the closest point on the valid line parts to the centroid.
        # This will be the first point on the ray outside the restricted area.
        _, first_valid_pos = nearest_points(centroid, valid_line_parts)

        if first_valid_pos is None:
            return []

        # 5. From the first valid position, generate a few more candidates along the ray.
        candidate_locations = [rounded_Point(first_valid_pos)]
        num_candidates = 4
        step_size = GRID_SPACING * 2  # Use grid spacing for a sensible step
        for i in range(1, num_candidates + 1):
            candidate = rounded_point(
                first_valid_pos.x + direction_vector[0] * step_size * i,
                first_valid_pos.y + direction_vector[1] * step_size * i,
            )
            if not candidate.within(restricted_area):
                candidate_locations.append(candidate)

        if not test_for_int_list_of_points(candidate_locations):
            log.warning(f"not int list of points: {candidate_locations=}")
        return candidate_locations

    def _log_jump_iteration_results(
        self,
        iteration: int,
        topology: Topology,
        context: RoutingContext,
        module: Module,
        cost_reduced: bool,
        proposed_moves: Dict[Junction, Point],
        parent_map: Dict[Junction, Optional[Junction]],
    ):
        """Log and plot results of a jump iteration."""
        log.info(f"calculating cost and dumping to prefix jump_iter_{iteration}_ for module:{module.name} net:{topology.net.name}")
        self._cost_calculator.calculate_total_cost(
            topology,
            context,
            debug_plot=True,
            plot_filename_prefix=f"jump_iter_{iteration}_",
        )

        if cost_reduced:
            for junction, new_location in proposed_moves.items():
                junction.location = new_location
                junction.geom = None
                if parent_map.get(junction):
                    parent_map.get(junction).geom = None
                log.info(f"  Jumped junction {junction.name} to {new_location}")

    def _finalize_routes_and_pin_locations(
        self,
        topology: Topology,
    ):
        """
        Finalize routes by updating pin locations for pins inside macros and then
        clipping wire geometry to exclude regions inside macros.
        """
        context = topology.context
        # --- 1. Update pin locations ---
        for junction in topology.junctions:
            start_point = junction.location
            path = junction.geom
            path = self.round_lines(path)
            for child in junction.children:
                if isinstance(child, Pin) and child in context.pin_macros:
                    macro_poly = child.instance.draw.geom
                    intersection = path.intersection(macro_poly)
                    new_pin_loc = None
                    if not intersection.is_empty:
                        if isinstance(intersection, Point):
                            new_pin_loc = intersection
                        else:
                            # Pick closest boundary point
                            new_pin_loc = nearest_points(start_point, intersection)[1]
                    if new_pin_loc:
                        child.draw.geom = new_pin_loc
                    else:
                        log.warning(
                            f"Could not find intersection for pin {child.full_name} on macro boundary. Using original pin location."
                        )

                    # Subtract macro interiors
                    clipped = path.difference(macro_poly)
                    if clipped.is_empty:
                        log.warning(f"All wire geometry for net {topology.net.name} lies inside macros — removed.")
                        continue
                    if type(clipped) == LineString:
                        path = linemerge([clipped])
                    else:
                        path = linemerge(clipped)
            if isinstance(path, LineString):
                junction.geom = MultiLineString([path])
            else:
                junction.geom = path

        # --- 2. Update net geometry ---
        new_geoms = [j.geom for j in topology.junctions]
        if new_geoms:
            union_geom = unary_union(new_geoms)
            merged_geom = None
            if isinstance(union_geom, LineString):
                merged_geom = linemerge([union_geom])
            elif isinstance(union_geom, MultiLineString):
                merged_geom = linemerge(union_geom)
            if isinstance(merged_geom, LineString):
                topology.net.draw.geom = MultiLineString([merged_geom])
            elif isinstance(merged_geom, MultiLineString):
                topology.net.draw.geom = merged_geom
            else:
                topology.net.draw.geom = MultiLineString([g for g in new_geoms if isinstance(g, LineString)])
        else:
            topology.net.draw.geom = None

    def _diagnose_crossings(self, module: Module):
        """Checks all routed nets in a module for crossings or overlaps and logs them."""
        log.info(f"--- Crossing Diagnosis for Module: {module.name} ---")
        nets = [n for n in module.nets.values() if hasattr(n.draw, "geom") and n.draw.geom and not n.draw.geom.is_empty]

        net_lines = {}
        for net in nets:
            geom = net.draw.geom
            if isinstance(geom, LineString):
                net_lines[net.name] = [geom]
            elif isinstance(geom, MultiLineString):
                net_lines[net.name] = list(geom.geoms)

        violations = 0
        net_names = list(net_lines.keys())
        for i in range(len(net_names)):
            for j in range(i + 1, len(net_names)):
                net1_name = net_names[i]
                net2_name = net_names[j]
                for line1 in net_lines[net1_name]:
                    for line2 in net_lines[net2_name]:
                        if line1.intersects(line2):
                            # Ignore intersections at the exact endpoints, as they are likely intended connections
                            if line1.touches(line2):
                                intersection = line1.intersection(line2)
                                if isinstance(intersection, Point):
                                    is_endpoint = False
                                    for p in [
                                        Point(line1.coords[0]),
                                        Point(line1.coords[-1]),
                                        Point(line2.coords[0]),
                                        Point(line2.coords[-1]),
                                    ]:
                                        if p.equals(intersection):
                                            is_endpoint = True
                                            break
                                    if is_endpoint:
                                        continue

                            intersection = line1.intersection(line2)
                            log.warning(
                                f"Violation between {net1_name} and {net2_name}: "
                                f"type = {type(intersection).__name__}, WKT = {intersection.wkt}"
                            )
                            violations += 1

        if violations == 0:
            log.info("No crossing/overlap violations found.")
        else:
            log.warning(f"Found {violations} crossing/overlap violations.")
        log.info(f"--- End of Diagnosis ---")

    @staticmethod
    def _subtract_intervals(
        source_intervals: List[Tuple[int, int]],
        subtraction_intervals: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Subtracts a list of intervals from another list of intervals.

        For example:
        source: [(0, 100)]
        subtraction: [(20, 30), (50, 60)]
        returns: [(0, 20), (30, 50), (60, 100)]
        """
        if not subtraction_intervals:
            return source_intervals

        remaining_parts = source_intervals
        for sub_start, sub_end in subtraction_intervals:
            next_remaining_parts = []
            for part_start, part_end in remaining_parts:
                # Calculate the actual overlap region
                overlap_start = max(part_start, sub_start)
                overlap_end = min(part_end, sub_end)

                # If there's a valid overlap to subtract
                if overlap_start < overlap_end:
                    # Keep the part before the overlap
                    if part_start < overlap_start:
                        next_remaining_parts.append((part_start, overlap_start))
                    # Keep the part after the overlap
                    if overlap_end < part_end:
                        next_remaining_parts.append((overlap_end, part_end))
                else:
                    # No overlap, keep the original part
                    next_remaining_parts.append((part_start, part_end))

            remaining_parts = next_remaining_parts

        return remaining_parts

    def remove_macros_from_tracks(
        self,
        h_tracks: Dict[int, List[Tuple[int, int]]],
        v_tracks: Dict[int, List[Tuple[int, int]]],
        pin_macros: Dict[Pin, Polygon],  # Key can be any object, value is Polygon
    ) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
        """
        Removes areas covered by macros from track occupancy maps.

        Args:
            h_tracks: Dictionary of horizontal tracks {y: [(x1, x2), ...]}.
            v_tracks: Dictionary of vertical tracks {x: [(y1, y2), ...]}.
            pin_macros: A dictionary of Shapely Polygons representing macro areas.

        Returns:
            A tuple containing the modified (h_tracks, v_tracks).
        """
        new_h_tracks = defaultdict(list)
        new_v_tracks = defaultdict(list)
        macros = list(pin_macros.values())

        if not macros:
            return h_tracks, v_tracks

        # Find the global bounding box of all macros to create "infinite" lines
        # for intersection tests.
        all_bounds = [m.bounds for m in macros]
        min_x = min(b[0] for b in all_bounds) - 1
        min_y = min(b[1] for b in all_bounds) - 1
        max_x = max(b[2] for b in all_bounds) + 1
        max_y = max(b[3] for b in all_bounds) + 1

        # Process Horizontal Tracks
        for y, segments in h_tracks.items():
            current_segments = segments
            for macro in macros:
                # Quick check to see if the track can possibly intersect the macro
                if not (macro.bounds[1] <= y <= macro.bounds[3]):
                    continue

                # Create a long horizontal line to find the macro's slice at this y
                intersection_line = LineString([(min_x, y), (max_x, y)])
                macro_slice = macro.intersection(intersection_line)

                subtraction_intervals = []
                if not macro_slice.is_empty:
                    # Handle both single and multiple intersection segments
                    if isinstance(macro_slice, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
                        geoms = list(macro_slice.geoms)
                    else:
                        geoms = [macro_slice]
                    for geom in geoms:
                        coords = geom.coords
                        subtraction_intervals.append(tuple(sorted((int(round(coords[0][0])), int(round(coords[1][0]))))))

                current_segments = self._subtract_intervals(current_segments, subtraction_intervals)

            if current_segments:
                new_h_tracks[y] = current_segments

        # Process Vertical Tracks
        for x, segments in v_tracks.items():
            current_segments = segments
            for macro in macros:
                # Quick check to see if the track can possibly intersect the macro
                if not (macro.bounds[0] <= x <= macro.bounds[2]):
                    continue

                # Create a long vertical line to find the macro's slice at this x
                intersection_line = LineString([(x, min_y), (x, max_y)])
                macro_slice = macro.intersection(intersection_line)

                subtraction_intervals = []
                if not macro_slice.is_empty:
                    if isinstance(macro_slice, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
                        geoms = list(macro_slice.geoms)
                    else:
                        geoms = [macro_slice]
                    for geom in geoms:
                        coords = geom.coords
                        subtraction_intervals.append(tuple(sorted((int(round(coords[0][1])), int(round(coords[1][1]))))))

                current_segments = self._subtract_intervals(current_segments, subtraction_intervals)

            if current_segments:
                new_v_tracks[x] = current_segments

        log.debug(f"before {h_tracks=} {v_tracks=} ")
        log.debug(f"after {new_h_tracks=}  {new_v_tracks=} ")
        log.debug(f"macros = {macros}")
        return new_h_tracks, new_v_tracks

    def round_lines(self, geom):
        """Round coordinates of a LineString or MultiLineString to integer grid."""
        if geom.is_empty:
            return geom

        if geom.geom_type == "LineString":
            return LineString([(int(round(x)), int(round(y))) for x, y in geom.coords])

        elif geom.geom_type == "MultiLineString":
            rounded_lines = [LineString([(int(round(x)), int(round(y))) for x, y in line.coords]) for line in geom.geoms]
            return MultiLineString(rounded_lines)

        else:
            raise TypeError(f"Unsupported geometry type: {geom.geom_type}")
