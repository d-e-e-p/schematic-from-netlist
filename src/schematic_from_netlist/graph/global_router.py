# Pattern Route-based global router

from __future__ import annotations

import logging as log
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import shapely
from rtree import index
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import linemerge, snap, unary_union
from shapely.strtree import STRtree

from schematic_from_netlist.database.netlist_structures import Module, Net, Pin
from schematic_from_netlist.graph.geom_utils import Geom

# Pattern Route parameters
ROUTE_WEIGHTS = {
    'wirelength': 1.0,
    'congestion': 2.0,
    'halo': 3.0,
    'crossing': 1.5
}
MAX_PATH_COST = 10.0  # Mean path cost multiplier for pruning
GRID_SPACING = 1.0  # Track spacing for snapping
MAX_FANOUT = 15  # Maximum junction fanout


@dataclass
class Topology:
    net: Net
    junctions: List[Junction] = field(default_factory=list)
    metrics = {}


@dataclass
class Junction:
    name: str
    location: Tuple[int, int]
    children: Set[Junction | Pin] = field(default_factory=set)

    def __hash__(self):
        return hash((self.name, self.location))


class GlobalRouter:
    def __init__(self, db):
        self.db = db
        self.junctions: Dict[Module, List[Topology]] = defaultdict(list)

    def _log_junction_summary(self, junctions):
        """Log detailed summary of inserted junctions."""
        from tabulate import tabulate
        
        summary = []
        for module, topos in junctions.items():
            for topo in topos:
                # Count unique children
                unique_children = set()
                for junction in topo.junctions:
                    unique_children.update(child for child in junction.children 
                                        if isinstance(child, Pin))  # Only count pin connections
                
                row = [
                    module.name,
                    topo.net.name,
                    topo.net.num_conn,
                    len(topo.junctions),
                    len(unique_children)
                ]
                summary.append(row)
                
                # Log detailed junction info
                for junction in topo.junctions:
                    log.info(f"Inserting {junction.name=} at {junction.location}")
                    for child in junction.children:
                        if isinstance(child, Junction):
                            log.info(f"  Connected to junction {child.name}")
                        elif isinstance(child, Pin):
                            log.info(f"  Connected to pin {child.name}")
                        else:
                            log.info(f"  Connected to unknown child type {type(child)}")
        
        # Log summary table
        headers = ["Module", "Net", "Connections", "Junctions", "Children"]
        log.info("Junction Insertion Summary:\n" + tabulate(summary, headers=headers, tablefmt="pipe"))

    def insert_routing_junctions(self):
        # Process groups
        for module in self.db.design.modules.values():
            sorted_nets = sorted(module.nets.values(), key=lambda net: net.num_conn)
            log.info(f"Processing module {module.name} with {len(sorted_nets)} nets")
            
            for net in sorted_nets:
                log.info(f"Processing net {net.name} with {net.num_conn} connections")
                
                if 2 < net.num_conn < self.db.fanout_threshold:
                    log.info(f"Net {net.name} is within fanout threshold ({self.db.fanout_threshold})")
                    topo = self.process_net(module, net)
                    if topo:
                        log.info(f"Created topology for net {net.name} with {len(topo.junctions)} junctions")
                        self.junctions[module].append(topo)
                    else:
                        log.warning(f"No topology created for net {net.name}")
                else:
                    log.info(f"Skipping net {net.name} - fanout {net.num_conn} outside threshold")
                    
        log.info(f"Total junctions created: {sum(len(v) for v in self.junctions.values())}")
        
        # Log detailed junction summary
        self._log_junction_summary(self.junctions)
        
        return self.junctions

    def process_net(self, module: Module, net: Net) -> Optional[Topology]:
        """Process a single net using Pattern Route-based routing.

        Args:
            module: Parent module containing the net
            net: Net to process

        Returns:
            Topology object with junctions and routes, or None if skipped
        """
        # Stage 0 - Preprocessing
        net.draw.geom = None  # Clear old geometry
        pins = [p.draw.geom for p in net.pins.values()]
        log.info(f"Processing net {net.name} with {len(pins)} pins")
        
        if len(pins) < 2:
            log.warning(f"Skipping net {net.name} - only {len(pins)} pins")
            return None

        # Get macro and halo trees for obstacle avoidance
        all_macros = [i.draw.geom for i in module.get_all_instances().values()]
        log.info(f"Found {len(all_macros)} macros in module {module.name}")
        
        macro_tree = STRtree(all_macros)
        halo_tree = STRtree([m.buffer(2) for m in all_macros])  # 2-track halo
        log.info(f"Created STRtrees with {len(macro_tree.geometries)} macros and {len(halo_tree.geometries)} halos")

        # Initialize cost map
        cost_map = self._create_cost_map(module, macro_tree, halo_tree)
        log.info(f"Created cost map with shape {cost_map.shape}")

        # Stage 1 - Pattern Route Generation
        routes = self._generate_pattern_routes(pins, macro_tree, cost_map)
        log.info(f"Generated {len(routes)} candidate routes")
        
        if not routes:
            log.warning("No valid routes found")
            return None

        # Stage 2 - Junction Tree Formation
        topology = self._form_junction_tree(net, pins, routes)

        # Stage 3 - Geometry Assembly
        if topology:
            self._assemble_geometry(net, topology, cost_map)
            return topology
        return None

    def _sample_junction_candidates(self, pins, bbox, macro_tree, halo_tree) -> List[Point]:
        """Generate valid junction candidates using Hanan grid and kd-tree sampling."""
        # Get all unique x/y coordinates from pins
        xs = sorted({p.x for p in pins})
        ys = sorted({p.y for p in pins})

        log.info(f"Pin coordinates - xs: {xs}, ys: {ys}")

        # Generate Hanan grid points
        candidates = [Point(x, y) for x in xs for y in ys]
        log.info(f"Initial Hanan grid candidates: {len(candidates)}")

        # Always expand search area to ensure we have options
        min_x, min_y, max_x, max_y = bbox
        expansion = 5
        log.info(f"Expanding bbox {bbox} by {expansion}")

        # Add grid points within expanded bounds
        for x in range(int(min_x - expansion), int(max_x + expansion) + 1):
            for y in range(int(min_y - expansion), int(max_y + expansion) + 1):
                pt = Point(x, y)
                if pt not in candidates:
                    candidates.append(pt)

        log.info(f"Total candidates after expansion: {len(candidates)}")

        # Filter invalid points
        valid_candidates = []
        macro_geoms = macro_tree.geometries
        halo_geoms = halo_tree.geometries

        for pt in candidates:
            # Check macro intersection
            if any(pt.intersects(m) for m in macro_geoms):
                continue

            # Check halo with some tolerance
            min_halo_dist = min(h.distance(pt) for h in halo_geoms)
            if min_halo_dist >= -0.1:  # Small negative tolerance for floating point
                valid_candidates.append(pt)

        log.info(f"Valid candidates after filtering: {len(valid_candidates)}")
        if not valid_candidates:
            log.warning(f"No valid candidates found! Macro count: {len(macro_geoms)}, Halo count: {len(halo_geoms)}")
            log.warning(f"First few pins: {[(p.x, p.y) for p in pins[:3]]}")
            log.warning(f"First macro bounds: {macro_geoms[0].bounds if macro_geoms else 'None'}")

            # Fallback - just use pin locations if no other candidates
            return [Point(p.x, p.y) for p in pins]

        return valid_candidates

    def _generate_pattern_routes(self, pins, macro_tree, cost_map) -> List[Tuple]:
        """Generate L/Z-shaped pattern routes between pin pairs."""
        routes = []
        
        # Generate all unique pin pairs
        for i, src in enumerate(pins):
            for j, dst in enumerate(pins):
                if i >= j:
                    continue
                    
                log.debug(f"Processing pin pair: {src} -> {dst}")
                
                # Generate L-shaped paths
                l_paths = [
                    self._create_l_path(src, dst, 'horizontal'),
                    self._create_l_path(src, dst, 'vertical')
                ]
                
                # Generate Z-shaped path if needed
                z_path = self._create_z_path(src, dst)
                if z_path:
                    l_paths.append(z_path)
                    
                # Evaluate and select best path
                best_path = None
                min_cost = float('inf')
                valid_paths = 0
                
                for path in l_paths:
                    if path is None:
                        log.debug("Skipping invalid path")
                        continue
                        
                    # Check for macro intersections
                    intersecting_macros = macro_tree.query(path)
                    if len(intersecting_macros) > 0:
                        log.debug(f"Path intersects {len(intersecting_macros)} macros: {path}")
                        # Try to route around macros
                        detour_path = self._create_detour_path(src, dst, intersecting_macros, macro_tree)
                        if detour_path:
                            path = detour_path
                            log.debug(f"Using detour path: {detour_path}")
                        else:
                            continue
                            
                    # Calculate path cost
                    cost = self._calculate_path_cost(path, cost_map)
                    log.debug(f"Path cost: {cost} for {path}")
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_path = path
                        valid_paths += 1
                            
                if best_path:
                    if min_cost < MAX_PATH_COST * self._mean_path_cost(cost_map):
                        log.debug(f"Selected best path: {best_path} with cost {min_cost}")
                        routes.append((src, dst, best_path, min_cost))
                    else:
                        log.debug(f"Path cost {min_cost} exceeds threshold")
                else:
                    log.debug(f"No valid path found between {src} and {dst}")
                    
        log.info(f"Generated {len(routes)} candidate routes from {len(pins)} pins")
        return routes

    def _create_detour_path(self, src: Point, dst: Point, macro_indices: List, macro_tree: STRtree) -> Optional[LineString]:
        """Create a detour path around intersecting macros."""
        try:
            # Get actual macro geometries from indices
            macros = [macro_tree.geometries[i] for i in macro_indices]
            # Create a union of the macro geometries
            macro_union = unary_union(macros)
            
            # Try horizontal detour
            mid1 = Point(src.x, dst.y + 5)  # Go up
            mid2 = Point(dst.x, dst.y + 5)
            path = LineString([src, mid1, mid2, dst])
            
            if path.is_valid:
                return path
                
            # Try vertical detour
            mid1 = Point(src.x + 5, src.y)  # Go right
            mid2 = Point(src.x + 5, dst.y)
            path = LineString([src, mid1, mid2, dst])
            
            if path.is_valid:
                return path
                
            return None
        except Exception as e:
            log.error(f"Error creating detour path: {e}")
            return None

    def _build_and_solve_model(self, model, net, pins, junctions, edges) -> Optional[Topology]:
        """Build and solve the CP-SAT optimization model."""
        # Create variables
        edge_vars = {}
        for i, (src, dst, path, metrics) in enumerate(edges):
            edge_vars[(src, dst)] = model.NewBoolVar(f"edge_{i}")

        junction_vars = {j: model.NewBoolVar(f"junction_{i}") for i, j in enumerate(junctions)}
        label_vars = {p: model.NewBoolVar(f"label_{i}") for i, p in enumerate(pins)}

        log.info(f"{len(edge_vars)=}.")
        # Objective function
        obj_terms = []
        for edge_key, var in edge_vars.items():
            # Look up the full edge info from edges list
            src, dst, path, metrics = next((e for e in edges if (e[0], e[1]) == edge_key), None)
            if metrics:
                cost = (
                    DEFAULT_WEIGHTS["wirelength"] * metrics["length"]
                    + DEFAULT_WEIGHTS["congestion"] * metrics["congestion"]
                    + DEFAULT_WEIGHTS["halo"] * metrics["halo"]
                    + DEFAULT_WEIGHTS["crossing"] * metrics["crossing"]
                )
                obj_terms.append(var * cost)

        for p, var in label_vars.items():
            obj_terms.append(var * DEFAULT_WEIGHTS["label"])

        for j, var in junction_vars.items():
            obj_terms.append(var * DEFAULT_WEIGHTS["halo"])  # Junction halo cost

        log.info(f"{len(obj_terms)=}.")
        model.Minimize(sum(obj_terms))

        # Constraints
        # Each pin must be connected or labeled
        for p in pins:
            connected_edges = [var for (src, dst), var in edge_vars.items() if src == p or dst == p]
            model.Add(sum(connected_edges) + label_vars[p] >= 1)

        # Junction fanout <= 2
        for j in junctions:
            connected_edges = [var for (src, dst), var in edge_vars.items() if src == j or dst == j]
            model.Add(sum(connected_edges) <= 2)

        # Add junction activation constraints
        for j in junctions:
            for (src, dst), var in edge_vars.items():
                if src == j or dst == j:
                    model.AddImplication(var, junction_vars[j])

        # Configure solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        solver.parameters.num_search_workers = os.cpu_count()
        solver.parameters.random_seed = 42
        log.info(f"Solving CP-SAT model with {len(edge_vars)} edges and {len(junctions)} junctions")
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        # Extract solution
        active_junctions = {j: Junction((j.x, j.y)) for j in junctions if solver.Value(junction_vars[j])}
        topology = Topology(net=net, junctions=active_junctions)

        # Build topology tree
        for (src, dst), var in edge_vars.items():
            if solver.Value(var):
                if src in pins and dst in active_junctions:
                    active_junctions[dst].children.append(src)
                elif dst in pins and src in active_junctions:
                    active_junctions[src].children.append(dst)
                elif src in active_junctions and dst in active_junctions:
                    # Connect junctions
                    active_junctions[src].children.append(active_junctions[dst])

        # Record metrics
        topology.metrics = {
            "wirelength": sum(
                solver.Value(var) * next((e[3]["length"] for e in edges if (e[0], e[1]) == edge_key), 0)
                for edge_key, var in edge_vars.items()
            ),
            "num_junctions": len(active_junctions),
            "solver_runtime_s": solver.WallTime(),
        }
        return topology

    def _create_l_path(self, src: Point, dst: Point, direction: str) -> LineString:
        """Create an L-shaped path in specified direction."""
        try:
            x1, y1 = src.x, src.y
            x2, y2 = dst.x, dst.y
            
            # Add small offset to avoid exact overlaps
            offset = 0.1
            
            if direction == 'horizontal':
                mid = Point(x2, y1 + offset)
            else:
                mid = Point(x1 + offset, y2)
                
            path = LineString([src, mid, dst])
            if not path.is_valid:
                log.warning(f"Invalid L-path generated: {path}")
                return None
            return path
        except Exception as e:
            log.error(f"Error creating L-path: {e}")
            return None

    def _create_z_path(self, src: Point, dst: Point) -> Optional[LineString]:
        """Create a Z-shaped path if beneficial."""
        try:
            x1, y1 = src.x, src.y
            x2, y2 = dst.x, dst.y
            
            # Only create Z path if both coordinates differ significantly
            if abs(x1 - x2) > GRID_SPACING and abs(y1 - y2) > GRID_SPACING:
                mid1 = Point(x1 + (x2 - x1)/2, y1)
                mid2 = Point(x1 + (x2 - x1)/2, y2)
                path = LineString([src, mid1, mid2, dst])
                if not path.is_valid:
                    log.warning(f"Invalid Z-path generated: {path}")
                    return None
                return path
            return None
        except Exception as e:
            log.error(f"Error creating Z-path: {e}")
            return None

    def _sample_path_points(self, path: LineString, spacing: float = 1.0) -> List[Point]:
        """Sample points along a path at regular intervals.
        
        Args:
            path: LineString to sample
            spacing: Distance between sample points
            
        Returns:
            List of sampled points
        """
        points = []
        length = path.length
        distance = 0.0
        
        while distance <= length:
            point = path.interpolate(distance)
            points.append(point)
            distance += spacing
            
        return points

    def _calculate_path_cost(self, path: LineString, cost_map) -> float:
        """Calculate total cost for a path using the cost map."""
        # Sample points along the path
        samples = self._sample_path_points(path)
        
        # Accumulate costs
        total_cost = 0.0
        for pt in samples:
            x, y = int(pt.x), int(pt.y)
            if 0 <= x < cost_map.shape[1] and 0 <= y < cost_map.shape[0]:
                total_cost += cost_map[y, x]
                
        # Add length penalty
        total_cost += ROUTE_WEIGHTS['wirelength'] * path.length
        
        return total_cost

    def _create_cost_map(self, module, macro_tree, halo_tree) -> np.ndarray:
        """Create a 2D cost map based on macro and halo locations.
        
        Args:
            module: Module containing the net
            macro_tree: STRtree of macro geometries
            halo_tree: STRtree of halo geometries
            
        Returns:
            2D numpy array of costs
        """
        # Get bounding box of all pins
        pins = [p.draw.geom for p in module.get_all_pins().values()]
        if not pins:
            return np.zeros((1, 1))
            
        bbox = unary_union(pins).bounds
        min_x, min_y, max_x, max_y = bbox
        
        # Create grid with some padding
        width = int(max_x - min_x) + 10
        height = int(max_y - min_y) + 10
        cost_map = np.zeros((height, width))
        
        # Add macro costs
        for macro in macro_tree.geometries:
            x1, y1, x2, y2 = macro.bounds
            x1 = max(0, int(x1 - min_x))
            y1 = max(0, int(y1 - min_y))
            x2 = min(width, int(x2 - min_x))
            y2 = min(height, int(y2 - min_y))
            cost_map[y1:y2, x1:x2] += ROUTE_WEIGHTS['halo'] * 10  # High cost for macros
            
        # Add halo costs
        for halo in halo_tree.geometries:
            x1, y1, x2, y2 = halo.bounds
            x1 = max(0, int(x1 - min_x))
            y1 = max(0, int(y1 - min_y))
            x2 = min(width, int(x2 - min_x))
            y2 = min(height, int(y2 - min_y))
            cost_map[y1:y2, x1:x2] += ROUTE_WEIGHTS['halo']
            
        return cost_map

    def _form_junction_tree(self, net: Net, pins: List[Point], routes: List[Tuple]) -> Topology:
        """Form a junction tree from the generated routes.
        
        Args:
            net: The net being routed
            pins: List of pin points
            routes: List of (src, dst, path, cost) tuples
            
        Returns:
            Topology object with junctions and connections
        """
        # Create initial topology with all pins
        topology = Topology(net=net)
        
        # Create a junction tree that shares junctions between connections
        junctions = {}  # Map of location to junction
        
        def get_junction(location):
            """Get or create a junction at the given location."""
            if location not in junctions:
                junction = Junction(
                    name=f"J{len(junctions)}",
                    location=location
                )
                junctions[location] = junction
                topology.junctions.append(junction)
            return junctions[location]
        
        # Create junctions at pin locations
        for pin in net.pins.values():
            junction = get_junction((pin.draw.geom.x, pin.draw.geom.y))
            junction.children.add(pin)
            
        # Create intermediate junctions along routes
        for src, dst, path, cost in routes:
            # Get source and destination junctions
            src_junction = get_junction((src.x, src.y))
            dst_junction = get_junction((dst.x, dst.y))
            
            # Create intermediate junction at midpoint
            mid_point = path.interpolate(0.5, normalized=True)
            mid_junction = get_junction((mid_point.x, mid_point.y))
            
            # Connect junctions
            src_junction.children.add(mid_junction)
            dst_junction.children.add(mid_junction)
            mid_junction.children.add(src_junction)
            mid_junction.children.add(dst_junction)
            
        # Merge junctions that are too close
        merged = True
        while merged:
            merged = False
            junctions = topology.junctions
            
            for i, j1 in enumerate(junctions):
                for j2 in junctions[i+1:]:
                    if Point(j1.location).distance(Point(j2.location)) < GRID_SPACING * 2:
                        # Merge j2 into j1
                        j1.children.update(j2.children)
                        # Update any references to j2 to point to j1
                        for junction in topology.junctions:
                            if j2 in junction.children:
                                junction.children.remove(j2)
                                junction.children.add(j1)
                        del topology.junctions[i+1]
                        merged = True
                        break
                if merged:
                    break
                    
        return topology

    def _mean_path_cost(self, cost_map) -> float:
        """Calculate mean path cost from cost map."""
        return np.mean(cost_map[cost_map > 0])

    def _assemble_geometry(self, net, topology, cost_map=None):
        """Convert topology into drawable geometry."""
        lines = []
        for junction in topology.junctions:
            # Convert junction location to Point
            j_point = Point(junction.location)
            
            for child in junction.children:
                if isinstance(child, Pin):  # Pin connection
                    lines.append(LineString([j_point, child.draw.geom]))
                elif isinstance(child, Junction):  # Junction connection
                    lines.append(LineString([j_point, Point(child.location)]))

        if lines:
            try:
                # Post-process geometry
                merged = linemerge(lines)
                if merged.is_empty:
                    return
                    
                # Snap to grid
                if isinstance(merged, LineString):
                    snapped = snap(merged, Point(0,0).buffer(GRID_SPACING), GRID_SPACING)
                else:  # MultiLineString
                    snapped = MultiLineString([
                        snap(line, Point(0,0).buffer(GRID_SPACING), GRID_SPACING)
                        for line in merged.geoms
                    ])
                    
                net.draw.geom = snapped

                # Record final metrics
                topology.metrics.update({
                    "wirelength": sum(line.length for line in lines),
                    "num_junctions": len(topology.junctions)
                })
            except Exception as e:
                log.error(f"Error assembling geometry: {e}")
                return
