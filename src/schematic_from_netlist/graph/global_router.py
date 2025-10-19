# Pattern Route-based global router

from __future__ import annotations

import logging as log
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rtree import index
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, nearest_points, snap, unary_union
from shapely.strtree import STRtree

from schematic_from_netlist.database.netlist_structures import Module, Net, Pin
from schematic_from_netlist.graph.cost_calculator import CostCalculator
from schematic_from_netlist.graph.router_debug import RouterDebugger
from schematic_from_netlist.graph.routing_helpers import (
    generate_l_paths,
    get_halo_geometries,
    get_l_path_corner,
    get_macro_geometries,
)
from schematic_from_netlist.graph.routing_utils import Junction, RoutingContext, Topology

# Pattern Route parameters
ROUTE_WEIGHTS = {"wirelength": 1.0, "congestion": 2.0, "halo": 5.0, "crossing": 5.0, "macro": 1000.0, "track": 20.0}
JUNCTION_SPACING_PENALTY = 2000.0  # Cost for placing junctions too close to each other
MAX_PATH_COST = 100.0  # Mean path cost multiplier for pruning
GRID_SPACING = 1.0  # Track spacing for snapping
MAX_FANOUT = 15  # Maximum junction fanout


class GlobalRouter:
    def __init__(self, db):
        self.db = db
        self.junctions: Dict[Module, List[Topology]] = defaultdict(list)
        self._debugger = RouterDebugger()
        self._cost_calculator = CostCalculator(self._debugger)

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

            # --- Data structures to hold intermediate results ---
            net_topologies: Dict[Net, Topology] = {}
            net_pin_macros: Dict[Net, Dict[Pin, Polygon]] = {}

            # --- STAGE 1: Initial Topology Generation ---
            log.info(f"--- Stage 1: Initial Topology Generation for {len(sorted_nets)} nets ---")
            for net in sorted_nets:
                net.draw.geom = None  # Clear existing geometry
                congestion_idx, other_nets_geoms = self._build_congestion_index(module, net)
                h_tracks, v_tracks = self._build_track_occupancy(other_nets_geoms)
                topo, pin_macros = self._create_initial_topology(
                    net, macros, halos, congestion_idx, other_nets_geoms, h_tracks, v_tracks
                )
                if topo:
                    net_topologies[net] = topo
                    net_pin_macros[net] = pin_macros or {}
                    self._rebuild_net_geometry(topo)  # Update geometry for next net

            # --- STAGE 2: Pruning Junctions ---
            log.info(f"--- Stage 2: Pruning Junctions ---")
            for topo in net_topologies.values():
                self._prune_redundant_junctions(topo)
                self._rebuild_net_geometry(topo)

            # --- STAGE 3: Optimizing Junction Locations ---
            log.info(f"--- Stage 3: Optimizing Junction Locations ---")
            for net, topo in net_topologies.items():
                self._optimize_junction_locations(topo, macros, halos, net_pin_macros[net])
                self._rebuild_net_geometry(topo)

            # --- STAGE 4: Sliding Junctions (Congestion-Aware) ---
            log.info(f"--- Stage 4: Sliding Junctions (Congestion-Aware) ---")
            for net, topo in net_topologies.items():
                congestion_idx, other_nets_geoms = self._build_congestion_index(module, net)
                h_tracks, v_tracks = self._build_track_occupancy(other_nets_geoms)
                self._slide_junctions(
                    topo,
                    macros,
                    halos,
                    congestion_idx,
                    other_nets_geoms,
                    module,
                    net_pin_macros[net],
                    net_topologies,
                    h_tracks,
                    v_tracks,
                )
                self._rebuild_net_geometry(topo)

            # --- STAGE 4.5: Jumping Junctions over Macros ---
            log.info("--- Stage 4.5: Jumping Junctions over Macros ---")
            for net, topo in net_topologies.items():
                congestion_idx, other_nets_geoms = self._build_congestion_index(module, net)
                h_tracks, v_tracks = self._build_track_occupancy(other_nets_geoms)
                self._jump_junctions_over_macros(
                    topo,
                    macros,
                    halos,
                    congestion_idx,
                    other_nets_geoms,
                    module,
                    net_pin_macros[net],
                    net_topologies,
                    h_tracks,
                    v_tracks,
                )
                self._rebuild_net_geometry(topo)

            # --- STAGE 5: Finalize Routes ---
            log.info(f"--- Stage 5: Finalizing Routes and Pin Locations ---")
            for net, topo in net_topologies.items():
                self._finalize_routes_and_pin_locations(topo, module, net_pin_macros[net])
                self.junctions[module].append(topo)

        log.info(f"Total junctions created: {sum(len(v) for v in self.junctions.values())}")

        # Log detailed junction summary
        self._debugger.log_junction_summary(self.junctions)
        self._debugger.plot_junction_summary(self.junctions)

        for module in self.db.design.modules.values():
            if module.is_leaf:
                continue
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
    ) -> Tuple[Dict[float, List[Tuple[float, float]]], Dict[float, List[Tuple[float, float]]]]:
        h_tracks = defaultdict(list)
        v_tracks = defaultdict(list)

        for line in geoms:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = Point(coords[i])
                p2 = Point(coords[i + 1])

                # Use a tolerance for floating point comparisons
                if abs(p1.y - p2.y) < 1e-9:  # Horizontal
                    h_tracks[p1.y].append(tuple(sorted((p1.x, p2.x))))
                elif abs(p1.x - p2.x) < 1e-9:  # Vertical
                    v_tracks[p1.x].append(tuple(sorted((p1.y, p2.y))))

        # Helper to merge intervals
        def merge_intervals(intervals):
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

        for y in h_tracks:
            h_tracks[y] = merge_intervals(h_tracks[y])
        for x in v_tracks:
            v_tracks[x] = merge_intervals(v_tracks[x])

        return h_tracks, v_tracks

    def _create_initial_topology(
        self,
        net: Net,
        macros: Polygon | BaseGeometry,
        halos: Polygon | BaseGeometry,
        congestion_idx,
        other_nets_geoms,
        h_tracks,
        v_tracks,
    ) -> Tuple[Optional[Topology], Optional[Dict[Pin, Polygon]]]:
        """Creates the initial MST-based topology for a single net."""
        pins = [p for p in net.pins.values() if hasattr(p.draw, "geom") and p.draw.geom is not None]
        if len(pins) < 3:
            return None, None

        log.debug(f"Creating initial topology for net {net.name} with {len(pins)} pins.")

        # For pins inside macros, route to the macro center
        pin_macros: Dict[Pin, Polygon] = {}
        for p in pins:
            if macro := p.instance.draw.geom:
                pin_macros[p] = macro
                p.draw.geom = macro.centroid
            else:
                log.warning(f"Pin {p.name} instance {p.instance.name} has no geom .")

        # Initialize MST algorithm (Prim's)
        unconnected_pins = set(pins)
        start_pin = unconnected_pins.pop()
        tree_connection_points: Set[Pin | Junction] = {start_pin}
        topology = Topology(net=net)

        context = RoutingContext(
            macros=macros,
            halos=halos,
            congestion_idx=congestion_idx,
            other_nets_geoms=other_nets_geoms,
            h_tracks=h_tracks,
            v_tracks=v_tracks,
            pin_macros=pin_macros,
        )

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

                    candidate_paths = self._generate_candidate_paths(p1, p2, halos)
                    for path_geom in candidate_paths:
                        p1_macro = pin_macros.get(conn_point) if isinstance(conn_point, Pin) else None
                        p2_macro = pin_macros.get(new_pin)
                        metrics, _ = self._cost_calculator.calculate_path_cost(
                            path_geom,
                            context,
                            p1_macro,
                            p2_macro,
                        )
                        if metrics.total_cost < min_cost:
                            min_cost = metrics.total_cost
                            best_path = path_geom
                            best_new_pin = new_pin
                            best_connection_point = conn_point

            if best_path is None:
                log.warning(f"Could not find a path for net {net.name}, skipping initial topology.")
                return None, None

            # Add new pin and junction to the tree
            if best_new_pin:
                unconnected_pins.remove(best_new_pin)
            corner = get_l_path_corner(best_path)
            junction_name = f"J_{net.name}_{len(topology.junctions)}"
            # Ensure we only add valid Pin or Junction objects to children
            children = set()
            if best_new_pin is not None:
                children.add(best_new_pin)
            if best_connection_point is not None:
                children.add(best_connection_point)
            junction = Junction(name=junction_name, location=corner, children=children)
            topology.junctions.append(junction)
            tree_connection_points.add(junction)

            if isinstance(best_connection_point, Pin):
                tree_connection_points.remove(best_connection_point)

        return topology, pin_macros

    def _generate_candidate_paths(self, p1: Point | None, p2: Point | None, halos: Polygon | BaseGeometry) -> List[LineString]:
        """Generate candidate L-shaped paths, including escape routes from halos."""
        paths = []
        if not p1 or not p2:
            return paths

        # 1. Basic L-paths
        paths.extend(generate_l_paths(p1, p2))

        # 2. Escape routes
        p1_in_halo = p1.within(halos)
        p2_in_halo = p2.within(halos)

        escape_points = {}
        if p1_in_halo:
            if not halos.boundary.is_empty:
                escape_points[1] = nearest_points(p1, halos.boundary)[1]

        if p2_in_halo:
            if not halos.boundary.is_empty:
                escape_points[2] = nearest_points(p2, halos.boundary)[1]

        e1 = escape_points.get(1)
        e2 = escape_points.get(2)

        if e1:
            paths.extend(generate_l_paths(e1, p2))
        if e2:
            paths.extend(generate_l_paths(p1, e2))
        if e1 and e2:
            paths.extend(generate_l_paths(e1, e2))

        # Return unique paths
        unique_paths = []
        seen = set()
        for p in paths:
            if p.wkt not in seen:
                unique_paths.append(p)
                seen.add(p.wkt)
        return unique_paths

    def _rebuild_net_geometry(self, topology: Topology):
        """Rebuilds the net's geometry based on its current topology of junctions and pins."""
        new_geoms = []
        for junction in topology.junctions:
            start_point = junction.location
            for child in junction.children:
                if isinstance(child, Pin):
                    end_point = child.draw.geom
                    if end_point is None:
                        continue
                    if not isinstance(end_point, Point):
                        end_point = end_point.centroid
                else:  # Junction
                    end_point = child.location

                # Generate orthogonal path to the final pin location (or other junction)
                l_paths = generate_l_paths(start_point, end_point)
                if l_paths:
                    new_geoms.append(l_paths[0])
                elif start_point.distance(end_point) > 1e-9:
                    new_geoms.append(LineString([start_point, end_point]))

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
        macros: Polygon | BaseGeometry,
        halos: Polygon | BaseGeometry,
        pin_macros: Dict[Pin, Polygon],
        passes: int = 3,
    ):
        """Iteratively improve junction locations based on a weighted median of their connections."""
        # Build parent map
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
                    new_loc = Point(float(np.median(coords_x)), float(np.median(coords_y)))

                    # If the ideal location is restricted, find the nearest valid point on the boundary
                    if new_loc.within(macros) or new_loc.within(halos):
                        restricted_area = unary_union([macros, halos])
                        if not restricted_area.boundary.is_empty:
                            new_loc = nearest_points(new_loc, restricted_area.boundary)[1]

                    j.location = new_loc

    def _slide_junctions(
        self,
        topology: Topology,
        macros: Polygon | BaseGeometry,
        halos: Polygon | BaseGeometry,
        congestion_idx,
        other_nets_geoms,
        module: Module,
        pin_macros: Dict[Pin, Polygon],
        all_topologies: Dict[Net, Topology],
        h_tracks,
        v_tracks,
    ):
        """
        Post-processing step to slide each junction in all directions to see if cost is reduced.
        This is run iteratively:
        1. In each iteration, find the best move for every junction based on the current topology.
        2. After checking all junctions, apply all the beneficial moves simultaneously.
        3. Re-plot the routing and repeat until no more cost reduction is possible.
        """
        search_step_size = 4
        log.debug(f"Starting junction sliding for net {topology.net.name}")

        # Build an STRtree of all junctions from OTHER nets for quick spatial queries
        other_junction_points = []
        for net, topo in all_topologies.items():
            if net is not topology.net:
                for j in topo.junctions:
                    other_junction_points.append(j.location)
        junction_tree = STRtree(other_junction_points) if other_junction_points else None

        context = RoutingContext(
            macros=macros,
            halos=halos,
            congestion_idx=congestion_idx,
            other_nets_geoms=other_nets_geoms,
            h_tracks=h_tracks,
            v_tracks=v_tracks,
            pin_macros=pin_macros,
            module=module,
        )

        max_iterations = 10  # To prevent infinite loops
        all_tried_locations: Dict[Junction, Dict[Tuple[float, float], float]] = defaultdict(dict)
        for i in range(max_iterations):
            proposed_moves = {}
            cost_reduced_this_iter = False

            for j_idx, junction in enumerate(topology.junctions):
                initial_location = junction.location
                min_cost = self._cost_calculator.calculate_total_cost(
                    topology,
                    context,
                )
                best_location = initial_location
                tried_locations_costs = {(initial_location.x, initial_location.y): min_cost}

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
                        if new_location.within(macros) or new_location.within(halos):
                            tried_locations_costs[(new_location.x, new_location.y)] = float("inf")
                            continue

                        junction.location = new_location
                        current_cost = self._cost_calculator.calculate_total_cost(
                            topology,
                            context,
                        )

                        # Add junction spacing penalty
                        if junction_tree:
                            nearby_junctions = junction_tree.query(new_location.buffer(4))
                            current_cost += JUNCTION_SPACING_PENALTY * len(nearby_junctions)

                        tried_locations_costs[(new_location.x, new_location.y)] = current_cost

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
                best_loc_tuple = min(tried_locations, key=tried_locations.get)
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
        macros: Polygon | BaseGeometry,
        halos: Polygon | BaseGeometry,
        congestion_idx,
        other_nets_geoms,
        module: Module,
        pin_macros: Dict[Pin, Polygon],
        all_topologies: Dict[Net, Topology],
        h_tracks,
        v_tracks,
    ):
        """
        Tries to find better junction locations by jumping over macros.
        This is for cases where a junction is stuck on one side of a macro due to a halo,
        while all its connected pins are on the same side.
        """
        log.info(f"Starting junction jumping for module:{module.name} net:{topology.net.name}")

        macro_polygons = [inst.draw.geom for inst in module.instances.values()]
        if not macro_polygons:
            return
        macro_tree = STRtree(macro_polygons)

        context = RoutingContext(
            macros=macros,
            halos=halos,
            congestion_idx=congestion_idx,
            other_nets_geoms=other_nets_geoms,
            h_tracks=h_tracks,
            v_tracks=v_tracks,
            pin_macros=pin_macros,
            module=module,
        )

        parent_map = self._build_junction_parent_map(topology)
        memo_downstream_pins = {}
        all_tried_locations: Dict[Junction, Dict[Tuple[float, float], float]] = defaultdict(dict)

        max_iterations = 10
        for i in range(max_iterations):
            proposed_moves = {}
            cost_reduced_this_iter = False

            for junction in topology.junctions:
                initial_location = junction.location
                min_cost = self._cost_calculator.calculate_total_cost(topology, context)
                best_location = initial_location

                downstream_pins = self._get_downstream_pins(junction, parent_map, memo_downstream_pins)
                if not downstream_pins:
                    continue

                candidate_locations = self._find_jump_candidate_locations(junction.location, downstream_pins, macros, halos)

                tried_locations_costs = {}
                for new_loc in candidate_locations:
                    if new_loc.within(macros) or new_loc.within(halos):
                        tried_locations_costs[(new_loc.x, new_loc.y)] = float("inf")
                        continue
                    junction.location = new_loc
                    current_cost = self._cost_calculator.calculate_total_cost(topology, context)
                    tried_locations_costs[(new_loc.x, new_loc.y)] = current_cost
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_location = new_loc

                # Also record the initial location
                junction.location = initial_location
                initial_cost = self._cost_calculator.calculate_total_cost(topology, context)
                tried_locations_costs[(initial_location.x, initial_location.y)] = initial_cost

                # Accumulate tried locations and costs
                for loc, cost in tried_locations_costs.items():
                    all_tried_locations[junction][loc] = cost

                if best_location != initial_location:
                    proposed_moves[junction] = best_location
                    cost_reduced_this_iter = True

            self._log_jump_iteration_results(i, topology, context, module, cost_reduced_this_iter, proposed_moves)

            if not cost_reduced_this_iter:
                break

        else:
            log.warning(f"Junction jumping for net {topology.net.name} did not converge after {max_iterations} iterations.")

        # After all iterations, generate a summary heatmap for each junction
        for junction, tried_locations in all_tried_locations.items():
            if tried_locations:
                best_loc_tuple = min(tried_locations, key=tried_locations.get)
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
        and sliding towards the original junction location to find valid spots.
        """
        pin_locations = [p.draw.geom for p in downstream_pins if hasattr(p.draw, "geom") and p.draw.geom]
        pin_locations = [p.centroid if not isinstance(p, Point) else p for p in pin_locations]
        if not pin_locations:
            return []

        # 1. Calculate centroid of downstream pins
        centroid_x = np.mean([p.x for p in pin_locations])
        centroid_y = np.mean([p.y for p in pin_locations])
        current_pos = Point(centroid_x, centroid_y)

        restricted_area = unary_union([macros, halos])
        search_line = LineString([current_pos, junction_loc])

        if search_line.length < 1e-6:
            return [current_pos] if not current_pos.within(restricted_area) else []

        # 2. Slide along the line from centroid to junction until we are out of restricted areas
        num_steps = 20
        first_valid_pos = None
        for i in range(num_steps + 1):
            fraction = i / num_steps
            current_pos = search_line.interpolate(fraction, normalized=True)
            if not current_pos.within(restricted_area):
                first_valid_pos = current_pos
                break
        breakpoint()

        if first_valid_pos is None:
            return []  # No valid position found along the line

        # 3. From the first valid position, generate a few more candidates towards the junction
        candidate_locations = [first_valid_pos]
        remaining_line = LineString([first_valid_pos, junction_loc])
        if remaining_line.length > 1.0:
            num_candidates = 4
            for i in range(1, num_candidates + 1):
                fraction = i / num_candidates
                candidate = remaining_line.interpolate(fraction, normalized=True)
                if not candidate.within(restricted_area):
                    candidate_locations.append(candidate)

        return candidate_locations

    def _log_jump_iteration_results(
        self,
        iteration: int,
        topology: Topology,
        context: RoutingContext,
        module: Module,
        cost_reduced: bool,
        proposed_moves: Dict[Junction, Point],
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
                log.info(f"  Jumped junction {junction.name} to {new_location}")

    def _finalize_routes_and_pin_locations(
        self,
        topology: Topology,
        module: Module,
        pin_macros: Dict[Pin, Polygon],
    ):
        """
        Finalize routes by updating pin locations for those inside macros and then rebuilding the net's geometry
        with orthogonal paths.
        """
        # Re-fetch everything needed for cost calculation
        macros = get_macro_geometries(module)
        halos = get_halo_geometries(macros)
        congestion_idx, other_nets_geoms = self._build_congestion_index(module, topology.net)
        h_tracks, v_tracks = self._build_track_occupancy(other_nets_geoms)

        context = RoutingContext(
            macros=macros,
            halos=halos,
            congestion_idx=congestion_idx,
            other_nets_geoms=other_nets_geoms,
            h_tracks=h_tracks,
            v_tracks=v_tracks,
            pin_macros=pin_macros,
            module=module,
        )

        # 1. First pass: Update pin locations for pins inside macros
        for junction in topology.junctions:
            start_point = junction.location
            for child in junction.children:
                if isinstance(child, Pin) and child in pin_macros:
                    end_point = child.draw.geom
                    if end_point is None:
                        continue
                    macro = pin_macros[child]

                    # Determine the best L-path to the macro center
                    l_paths = generate_l_paths(start_point, end_point)
                    if not l_paths:
                        continue

                    metrics1, _ = self._cost_calculator.calculate_path_cost(
                        l_paths[0],
                        context,
                        p2_macro_to_ignore=macro,
                    )
                    metrics2, _ = self._cost_calculator.calculate_path_cost(
                        l_paths[1],
                        context,
                        p2_macro_to_ignore=macro,
                    )
                    best_path_to_center = l_paths[0] if metrics1.total_cost <= metrics2.total_cost else l_paths[1]

                    # Find intersection with boundary and update pin location
                    intersection = best_path_to_center.intersection(macro.boundary)
                    new_pin_loc = None
                    if not intersection.is_empty:
                        if isinstance(intersection, Point):
                            new_pin_loc = intersection
                        else:  # MultiPoint, LineString, etc.
                            new_pin_loc = nearest_points(start_point, intersection)[1]

                    if new_pin_loc:
                        child.draw.geom = new_pin_loc
                    else:
                        log.warning(
                            f"Could not find intersection for pin {child.full_name} on macro boundary. "
                            f"Junction at {start_point}, macro centroid at {end_point}. Using original pin location."
                        )

        # 2. Second pass: Rebuild the entire net geometry with final pin locations
        new_geoms = []
        for junction in topology.junctions:
            start_point = junction.location
            for child in junction.children:
                if isinstance(child, Pin):
                    end_point = child.draw.geom
                    if end_point is None:
                        continue
                    if not isinstance(end_point, Point):
                        end_point = end_point.centroid
                else:  # Junction
                    end_point = child.location

                # Generate orthogonal path to the final pin location (or other junction)
                # We don't need to cost this, just pick one deterministically for the final geometry
                l_paths = generate_l_paths(start_point, end_point)
                if l_paths:
                    new_geoms.append(l_paths[0])
                elif start_point.distance(end_point) > 1e-9:
                    new_geoms.append(LineString([start_point, end_point]))

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
