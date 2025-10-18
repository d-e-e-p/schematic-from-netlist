# Pattern Route-based global router

from __future__ import annotations

import logging as log
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shapely
from rtree import index
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box
from shapely.ops import linemerge, nearest_points, snap, unary_union
from shapely.strtree import STRtree
from tabulate import tabulate

from schematic_from_netlist.database.netlist_structures import Module, Net, Pin
from schematic_from_netlist.graph.geom_utils import Geom

# Pattern Route parameters
ROUTE_WEIGHTS = {"wirelength": 1.0, "congestion": 20.0, "halo": 300.0, "crossing": 5.0, "macro": 500.0}
MAX_PATH_COST = 100.0  # Mean path cost multiplier for pruning
GRID_SPACING = 1.0  # Track spacing for snapping
MAX_FANOUT = 15  # Maximum junction fanout


@dataclass
class Topology:
    net: Net
    junctions: List[Junction] = field(default_factory=list)
    metrics: Metrics | None = None


@dataclass
class Junction:
    name: str
    location: Point
    children: Set[Junction | Pin] = field(default_factory=set)

    def __hash__(self):
        return hash((self.name, self.location))


@dataclass
class Metrics:
    # Geometric parameters
    wirelength: float = 0.0
    macro_overlap: float = 0.0
    halo_overlap: float = 0.0
    intersecting_length: float = 0.0

    # Individual weighted costs
    cost_wirelength: float = 0.0
    cost_macro: float = 0.0
    cost_halo: float = 0.0
    cost_congestion: float = 0.0
    cost_macro_junction_penalty: float = 0.0
    cost_halo_junction_penalty: float = 0.0

    # Aggregate total
    total_cost: float = 0.0

    def to_dict(self):
        """Return the metrics as a plain dictionary."""
        return asdict(self)

    def __str__(self):
        """Return a short summary string for logging."""
        return (
            f"wl={self.cost_wirelength:.1f}, "
            f"macro={self.cost_macro:.1f}, "
            f"halo={self.cost_halo:.1f}, "
            f"cong={self.cost_congestion:.1f}, "
            f"pen={self.cost_macro_junction_penalty + self.cost_halo_junction_penalty:.1f}, "
            f"total={self.total_cost:.1f}"
        )


class GlobalRouter:
    def __init__(self, db):
        self.db = db
        self.junctions: Dict[Module, List[Topology]] = defaultdict(list)

    def insert_routing_junctions(self):
        # Process groups
        for module in self.db.design.modules.values():
            if module.is_leaf:
                continue
            sorted_nets = sorted(module.nets.values(), key=lambda net: net.num_conn)
            log.info(f"Processing module {module.name} with {len(sorted_nets)} nets")

            for net in sorted_nets:
                if 2 < net.num_conn < self.db.fanout_threshold:
                    log.info(f"Processing net {net.name} with {net.num_conn} connections")
                    topo = self.process_net(module, net)
                    if topo:
                        log.info(f"Created topology for net {net.name} with {len(topo.junctions)} junctions")
                        self.junctions[module].append(topo)
                    else:
                        log.warning(f"No topology created for net {net.name}")

        log.info(f"Total junctions created: {sum(len(v) for v in self.junctions.values())}")

        # Log detailed junction summary
        self._log_junction_summary()
        self._plot_junction_summary()

        return self.junctions

    def _get_macro_geometries(self, module: Module) -> Polygon:
        """Get all macro geometries in a module."""
        geoms = []
        for i in module.get_all_instances().values():
            if hasattr(i.draw, "geom") and i.draw.geom:
                if isinstance(i.draw.geom, Polygon):
                    geoms.append(i.draw.geom)
                elif isinstance(i.draw.geom, MultiPolygon):
                    geoms.extend(list(i.draw.geom.geoms))
        return unary_union(geoms) if geoms else Polygon()

    def _get_halo_geometries(self, macros: Polygon, buffer_dist: int = 10) -> Polygon:
        """Get halo geometries around macros."""
        if macros.is_empty:
            return Polygon()
        return macros.buffer(buffer_dist)

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

    def _generate_candidate_paths(self, p1: Point, p2: Point, halos: Polygon) -> List[LineString]:
        """Generate candidate L-shaped paths, including escape routes from halos."""
        paths = []

        # 1. Basic L-paths
        paths.extend(self._generate_l_paths(p1, p2))

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
            paths.extend(self._generate_l_paths(e1, p2))
        if e2:
            paths.extend(self._generate_l_paths(p1, e2))
        if e1 and e2:
            paths.extend(self._generate_l_paths(e1, e2))

        # Return unique paths
        unique_paths = []
        seen = set()
        for p in paths:
            if p.wkt not in seen:
                unique_paths.append(p)
                seen.add(p.wkt)
        return unique_paths

    def _generate_l_paths(self, p1: Point, p2: Point) -> List[LineString]:
        """Generate two L-shaped paths between two points."""
        if not all(isinstance(p, Point) for p in [p1, p2]):
            return []
        path1 = LineString([(p1.x, p1.y), (p1.x, p2.y), (p2.x, p2.y)])
        path2 = LineString([(p1.x, p1.y), (p2.x, p1.y), (p2.x, p2.y)])
        return [path1, path2]

    def _get_l_path_corner(self, path: LineString) -> Point:
        """Get the corner point of an L-shaped path."""
        return Point(path.coords[1])

    def _calculate_path_cost(
        self,
        path: LineString,
        macros: Polygon,
        halos: Polygon,
        congestion_idx: index.Index,
        other_nets_geoms: List[LineString],
    ) -> Metrics:
        """Calculate detailed routing cost metrics for a given path."""

        # Compute geometric parameters
        wirelength = path.length
        macro_overlap = path.intersection(macros).length
        halo_overlap = path.intersection(halos).length

        # Compute congestion overlap
        intersecting_length = 0.0
        possible_intersections = [other_nets_geoms[i] for i in congestion_idx.intersection(path.bounds)]
        for geom in possible_intersections:
            if path.intersects(geom):
                intersecting_length += path.intersection(geom).length

        # Weighted cost components
        cost_wirelength = ROUTE_WEIGHTS["wirelength"] * wirelength
        cost_macro = ROUTE_WEIGHTS["macro"] * macro_overlap
        cost_halo = ROUTE_WEIGHTS["halo"] * halo_overlap
        cost_congestion = ROUTE_WEIGHTS["congestion"] * intersecting_length

        # Junction penalties
        cost_macro_junction_penalty = 0.0
        cost_halo_junction_penalty = 0.0
        if len(path.coords) == 3:  # L-shaped path
            corner = self._get_l_path_corner(path)
            if corner.within(macros):
                cost_macro_junction_penalty = 1000.0
            if corner.within(halos):
                cost_halo_junction_penalty = 500.0

        # Total cost
        total_cost = (
            cost_wirelength + cost_macro + cost_halo + cost_congestion + cost_macro_junction_penalty + cost_halo_junction_penalty
        )

        metrics = Metrics(
            wirelength=wirelength,
            macro_overlap=macro_overlap,
            halo_overlap=halo_overlap,
            intersecting_length=intersecting_length,
            cost_wirelength=cost_wirelength,
            cost_macro=cost_macro,
            cost_halo=cost_halo,
            cost_congestion=cost_congestion,
            cost_macro_junction_penalty=cost_macro_junction_penalty,
            cost_halo_junction_penalty=cost_halo_junction_penalty,
            total_cost=total_cost,
        )

        log.info(
            f"cost: wl={cost_wirelength:.1f}, macro={cost_macro:.1f}, halo={cost_halo:.1f}, "
            f"congestion={cost_congestion:.1f}, penalties={cost_macro_junction_penalty + cost_halo_junction_penalty:.1f}, "
            f"total={total_cost:.1f}"
        )

        return metrics

    def process_net(self, module: Module, net: Net) -> Optional[Topology]:
        """Process a single net using Pattern Route-based routing.

        Args:
            module: Parent module containing the net
            net: Net to process

        Returns:
            Topology object with junctions and routes, or None if skipped
        """
        # 0. Remove existing geometry
        net.draw.geom = None

        pins = [p for p in net.pins.values() if hasattr(p.draw, "geom") and p.draw.geom is not None]
        if len(pins) < 3:
            return None

        log.debug(f"Routing net {net.name} with {len(pins)} pins.")

        # 1. Pre-computation
        macros = self._get_macro_geometries(module)
        halos = self._get_halo_geometries(macros)
        congestion_idx, other_nets_geoms = self._build_congestion_index(module, net)

        # 2. Initialize MST algorithm (Prim's)
        unconnected_pins = set(pins)
        start_pin = unconnected_pins.pop()

        tree_connection_points = {start_pin}  # Pins and Junctions in the current tree
        topology = Topology(net=net)

        # 3. Greedy merge loop
        while unconnected_pins:
            min_cost = float("inf")
            best_path = None
            best_new_pin = None
            best_connection_point = None

            for new_pin in unconnected_pins:
                for conn_point in tree_connection_points:
                    p1 = conn_point.draw.geom if isinstance(conn_point, Pin) else conn_point.location
                    p2 = new_pin.draw.geom
                    if not isinstance(p1, Point):
                        p1 = p1.centroid
                    if not isinstance(p2, Point):
                        p2 = p2.centroid

                    candidate_paths = self._generate_candidate_paths(p1, p2, halos)
                    for path_geom in candidate_paths:
                        metrics = self._calculate_path_cost(path_geom, macros, halos, congestion_idx, other_nets_geoms)
                        if metrics.total_cost < min_cost:
                            min_cost = metrics.total_cost
                            best_path = path_geom
                            best_new_pin = new_pin
                            best_connection_point = conn_point

            if best_path is None:
                log.warning(f"Could not find a path for net {net.name}, skipping.")
                return None

            # 4. Add new pin and junction to the tree
            unconnected_pins.remove(best_new_pin)

            corner = self._get_l_path_corner(best_path)
            junction_name = f"J_{net.name}_{len(topology.junctions)}"
            junction = Junction(name=junction_name, location=corner, children={best_new_pin, best_connection_point})

            topology.junctions.append(junction)
            tree_connection_points.add(junction)

            # If the connection point was a pin, remove it from the set of available connection points
            if isinstance(best_connection_point, Pin):
                tree_connection_points.remove(best_connection_point)

        # 5. Prune redundant junctions
        self._prune_redundant_junctions(topology)

        # 6. Post-process junction locations
        self._optimize_junction_locations(topology, macros, halos)

        # 7. Slide junctions to reduce cost
        self._slide_junctions(topology, macros, halos, congestion_idx, other_nets_geoms)

        # 8. Finalize net geometry
        self._rebuild_net_geometry(topology)

        return topology

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

    def _optimize_junction_locations(self, topology: Topology, macros: Polygon, halos: Polygon, passes: int = 3):
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
                    new_loc = Point(np.median(coords_x), np.median(coords_y))

                    # If the ideal location is restricted, find the nearest valid point on the boundary
                    if new_loc.within(macros) or new_loc.within(halos):
                        restricted_area = unary_union([macros, halos])
                        if not restricted_area.boundary.is_empty:
                            new_loc = nearest_points(new_loc, restricted_area.boundary)[1]

                    j.location = new_loc

    def _slide_junctions(
        self, topology: Topology, macros: Polygon, halos: Polygon, congestion_idx, other_nets_geoms, search_step_size: int = 4
    ):
        """
        Post-processing step to slide each junction in all directions to see if cost is reduced.
        Greedy search all surrounding spots, take a step and repeat until no reduction for any junction.
        """
        log.info(f"Starting junction sliding for net {topology.net.name}")
        max_iterations = 1000  # To prevent infinite loops
        for i in range(max_iterations):
            cost_reduced = False
            for junction in topology.junctions:
                initial_location = junction.location
                min_cost = self._calculate_total_cost(topology, macros, halos, congestion_idx, other_nets_geoms)
                best_location = initial_location

                # Explore surrounding spots
                for dx in range(-search_step_size, search_step_size + 1, search_step_size):
                    for dy in range(-search_step_size, search_step_size + 1, search_step_size):
                        if dx == 0 and dy == 0:
                            continue

                        new_location = Point(initial_location.x + dx, initial_location.y + dy)
                        junction.location = new_location

                        # Recalculate cost with the new junction location
                        current_cost = self._calculate_total_cost(topology, macros, halos, congestion_idx, other_nets_geoms)
                        log.info(f"  Trying {new_location} (cost {current_cost} / {min_cost})")

                        if current_cost < min_cost:
                            min_cost = current_cost
                            best_location = new_location
                        # self._plot_junction_summary(f"_slide_junctions_{i}_", str(current_cost))
                # If a better location is found, move the junction
                if best_location != initial_location:
                    junction.location = best_location
                    cost_reduced = True
                    log.info(f"  Moved junction {junction.name} to {best_location} (cost reduced)")

            if not cost_reduced:
                log.info(f"Junction sliding for net {topology.net.name} converged after {i + 1} iterations.")
                break
        else:
            log.warning(f"Junction sliding for net {topology.net.name} did not converge after {max_iterations} iterations.")

    def _calculate_total_cost(self, topology: Topology, macros: Polygon, halos: Polygon, congestion_idx, other_nets_geoms):
        """Calculate the total cost of all paths in a topology."""
        total_cost = 0.0
        for junction in topology.junctions:
            p1 = junction.location
            for child in junction.children:
                if isinstance(child, Pin):
                    p2 = child.draw.geom
                else:  # Junction
                    p2 = child.location

                if not isinstance(p1, Point):
                    p1 = p1.centroid
                if not isinstance(p2, Point):
                    p2 = p2.centroid

                # Create a simple line for cost calculation, assuming direct connection for simplicity
                path = LineString([p1, p2])
                metrics = self._calculate_path_cost(path, macros, halos, congestion_idx, other_nets_geoms)
                total_cost += metrics.total_cost
        return total_cost

    def _rebuild_net_geometry(self, topology: Topology):
        """Recreate the net's geometry from the final junction and pin locations."""
        new_geoms = []
        for j in topology.junctions:
            start_point = j.location
            for child in j.children:
                if isinstance(child, Pin):
                    end_point = child.draw.geom
                    if not isinstance(end_point, Point):
                        end_point = end_point.centroid
                    end_point = (end_point.x, end_point.y)
                else:  # Junction
                    end_point = child.location
                new_geoms.append(LineString([start_point, end_point]))

        if new_geoms:
            merged_geom = linemerge(unary_union(new_geoms))

            # Normalize to MultiLineString
            if isinstance(merged_geom, LineString):
                merged_geom = MultiLineString([merged_geom])
            elif not isinstance(merged_geom, MultiLineString):
                # handle any other unexpected geometry types
                merged_geom = MultiLineString([g for g in new_geoms if isinstance(g, LineString)])

            topology.net.draw.geom = merged_geom
        else:
            topology.net.draw.geom = None

    def _log_junction_summary(self):
        """Log detailed summary of inserted junctions."""
        for module, topos in self.junctions.items():
            log.info(f"module {module.name=} size {module.draw.geom}")
            macros = self._get_macro_geometries(module)
            if not macros.is_empty:
                log.info(f"  Macro blockages at: {macros.wkt}")
            for topo in topos:
                # Log detailed junction info
                for junction in topo.junctions:
                    log.info(f"Inserting {junction.name=} in {topo.net.name} at {junction.location}")
                    for child in junction.children:
                        if isinstance(child, Junction):
                            log.info(f"  Connected to junction {child.name} at {child.location}")
                        elif isinstance(child, Pin):
                            log.info(f"  Connected to pin {child.full_name} at {child.draw.geom}")
                        else:
                            log.info(f"  Connected to unknown child type {type(child)}")

        summary = []
        for module, topos in self.junctions.items():
            for topo in topos:
                # Count unique children
                unique_children = set()
                for junction in topo.junctions:
                    unique_children.update(
                        child for child in junction.children if isinstance(child, Pin)
                    )  # Only count pin connections

                row = [module.name, topo.net.name, topo.net.num_conn, len(topo.junctions), len(unique_children)]
                summary.append(row)

        # Log summary table
        headers = ["Module", "Net", "Connections", "Junctions", "Children"]
        log.info("Junction Insertion Summary:\n" + tabulate(summary, headers=headers, tablefmt="pipe"))

    def _plot_junction_summary(self, stage: str = "", title: str = ""):
        """
        Generate per-module schematic overview plots showing macros, pins, junctions, and existing net geometries.

        Args:
            junctions: Dict[Module, List[Topology]]
        """
        out_dir = "data/images"
        os.makedirs(out_dir, exist_ok=True)

        cmap = plt.get_cmap("tab20")  # color map for nets

        for module, topos in self.junctions.items():
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Module: {module.name} {title}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            # --- Draw macros ---
            macros = self._get_macro_geometries(module)
            if not macros.is_empty:
                if isinstance(macros, MultiPolygon):
                    for sub in macros.geoms:
                        x, y = sub.exterior.xy
                        ax.fill(x, y, color="lightgrey", alpha=0.6)
                else:
                    x, y = macros.exterior.xy
                    ax.fill(x, y, color="lightgrey", alpha=0.6)

            # --- Draw halos ---
            halos = self._get_halo_geometries(macros)
            if not halos.is_empty:
                if isinstance(halos, MultiPolygon):
                    for sub in halos.geoms:
                        x, y = sub.exterior.xy
                        ax.plot(x, y, color="blue", ls="--", lw=1)
                else:
                    x, y = halos.exterior.xy
                    ax.plot(x, y, color="blue", ls="--", lw=1)

            # --- Draw junctions, pins, and nets ---
            for idx, topo in enumerate(topos):
                color = cmap(idx % 20)  # assign color per net

                # Plot net geometry if exists
                if topo.net.draw.geom:
                    geom = topo.net.draw.geom
                    if isinstance(geom, LineString):
                        geom = [geom]
                    elif isinstance(geom, MultiLineString):
                        geom = list(geom.geoms)
                    for line in geom:
                        x, y = line.xy
                        ax.plot(x, y, color=color, lw=1.5)
                    # Label the net at first point
                    first_line = geom[0]
                    ax.text(first_line.coords[0][0], first_line.coords[0][1], topo.net.name, fontsize=8, color=color)

                # Plot junctions
                for junction in topo.junctions:
                    jx, jy = junction.location.x, junction.location.y
                    ax.scatter(jx, jy, c=color, s=80, marker="x")
                    ax.text(jx + 0.5, jy + 0.5, junction.name, fontsize=7, color=color)

                    # Draw connections to children
                    for child in junction.children:
                        if isinstance(child, Pin) and hasattr(child.draw, "geom") and child.draw.geom:
                            pgeom = child.draw.geom
                            if isinstance(pgeom, Point):
                                px, py = pgeom.x, pgeom.y
                            else:
                                px, py = pgeom.centroid.x, pgeom.centroid.y
                            ax.plot([jx, px], [jy, py], color=color, lw=1)
                            ax.scatter(px, py, c="black", s=20, marker="o")
                            ax.text(px + 0.5, py + 0.5, child.full_name, fontsize=6, color="black")
                        elif isinstance(child, Junction):
                            cx, cy = child.location.x, child.location.y
                            ax.plot([jx, cx], [jy, cy], color=color, lw=1, ls="--")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.tight_layout()

            # Save figure
            fname = os.path.join(out_dir, f"{stage}{module.name}_junctions.png")
            plt.savefig(fname, dpi=200)
            plt.close(fig)

            log.info(f"Saved schematic plot for module {module.name} → {fname}")
