from __future__ import annotations

import logging as log
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry

from schematic_from_netlist.database.netlist_structures import Pin
from schematic_from_netlist.graph.router_debug import RouterDebugger
from schematic_from_netlist.graph.routing_helpers import generate_l_paths, get_l_path_corner
from schematic_from_netlist.graph.routing_utils import Junction, Metrics, RoutingContext, Topology

# Pattern Route parameters
ROUTE_WEIGHTS = {"wirelength": 1.0, "congestion": 2.0, "halo": 5.0, "crossing": 5.0, "macro": 1000.0, "track": 20.0}
GRID_SPACING = 1.0  # Track spacing for snapping


class CostCalculator:
    def __init__(self, debugger: RouterDebugger):
        self._debugger = debugger

    def calculate_path_cost(
        self,
        path: LineString,
        context: RoutingContext,
        p1_macro_to_ignore: Optional[Polygon] = None,
        p2_macro_to_ignore: Optional[Polygon] = None,
    ) -> Tuple[Metrics, List[Point]]:
        """Calculate detailed routing cost metrics for a given path."""

        # Compute geometric parameters
        wirelength = path.length
        macro_overlap = path.intersection(context.macros).length
        halo_overlap = path.intersection(context.halos).length

        # Compute congestion overlap
        intersecting_length = 0.0
        possible_intersections = [
            context.other_nets_geoms[i] for i in context.congestion_idx.intersection(path.bounds)
        ]
        for geom in possible_intersections:
            if path.intersects(geom):
                intersecting_length += path.intersection(geom).length

        # Compute crossings
        intersecting_crossings = 0
        crossing_points = []
        for geom in possible_intersections:
            if path.intersects(geom):
                intersection = path.intersection(geom)
                if isinstance(intersection, Point):
                    intersecting_crossings += 1
                    crossing_points.append(intersection)
                elif isinstance(intersection, MultiPoint):
                    intersecting_crossings += len(intersection.geoms)
                    crossing_points.extend(list(intersection.geoms))
                elif isinstance(intersection, (LineString, MultiLineString)):
                    # Overlap detected. Penalize based on length.
                    intersecting_crossings += math.ceil(intersection.length / GRID_SPACING)

        # Compute track overlap
        track_overlap_length = 0.0
        path_coords = list(path.coords)
        for i in range(len(path_coords) - 1):
            p1 = Point(path_coords[i])
            p2 = Point(path_coords[i + 1])

            if abs(p1.y - p2.y) < 1e-9:  # Horizontal segment
                y = p1.y
                seg_x1, seg_x2 = sorted((p1.x, p2.x))
                if y in context.h_tracks:
                    for track_x1, track_x2 in context.h_tracks[y]:
                        overlap_start = max(seg_x1, track_x1)
                        overlap_end = min(seg_x2, track_x2)
                        if overlap_start < overlap_end:
                            track_overlap_length += overlap_end - overlap_start
            elif abs(p1.x - p2.x) < 1e-9:  # Vertical segment
                x = p1.x
                seg_y1, seg_y2 = sorted((p1.y, p2.y))
                if x in context.v_tracks:
                    for track_y1, track_y2 in context.v_tracks[x]:
                        overlap_start = max(seg_y1, track_y1)
                        overlap_end = min(seg_y2, track_y2)
                        if overlap_start < overlap_end:
                            track_overlap_length += overlap_end - overlap_start

        # Weighted cost components
        cost_wirelength = ROUTE_WEIGHTS["wirelength"] * wirelength
        cost_macro = ROUTE_WEIGHTS["macro"] * macro_overlap
        cost_halo = ROUTE_WEIGHTS["halo"] * halo_overlap
        cost_congestion = ROUTE_WEIGHTS["congestion"] * intersecting_length
        cost_crossing = ROUTE_WEIGHTS["crossing"] * intersecting_crossings
        cost_track_overlap = ROUTE_WEIGHTS["track"] * track_overlap_length

        # Junction penalties
        cost_macro_junction_penalty = 0.0
        cost_halo_junction_penalty = 0.0
        if len(path.coords) == 3:  # L-shaped path
            corner = get_l_path_corner(path)
            if corner.within(context.macros):
                cost_macro_junction_penalty = 1000.0
            if corner.within(context.halos):
                cost_halo_junction_penalty = 500.0

        # Zero out wirelength and macro overlap costs for segments inside their own macro
        for macro_to_ignore in [p1_macro_to_ignore, p2_macro_to_ignore]:
            if macro_to_ignore:
                internal_segment = path.intersection(macro_to_ignore)
                if internal_segment.length > 0:
                    internal_wirelength = internal_segment.length

                    # Subtract wirelength and macro overlap cost for the internal segment
                    cost_wirelength -= ROUTE_WEIGHTS["wirelength"] * internal_wirelength
                    cost_macro -= ROUTE_WEIGHTS["macro"] * internal_wirelength

                    # Update raw metrics as well for logging consistency
                    wirelength -= internal_wirelength
                    macro_overlap -= internal_wirelength

        # Total cost
        total_cost = (
            cost_wirelength
            + cost_macro
            + cost_halo
            + cost_congestion
            + cost_crossing
            + cost_track_overlap
            + cost_macro_junction_penalty
            + cost_halo_junction_penalty
        )

        metrics = Metrics(
            wirelength=wirelength,
            macro_overlap=macro_overlap,
            halo_overlap=halo_overlap,
            intersecting_length=intersecting_length,
            intersecting_crossings=intersecting_crossings,
            cost_wirelength=cost_wirelength,
            cost_macro=cost_macro,
            cost_halo=cost_halo,
            cost_congestion=cost_congestion,
            cost_crossing=cost_crossing,
            cost_track_overlap=cost_track_overlap,
            cost_macro_junction_penalty=cost_macro_junction_penalty,
            cost_halo_junction_penalty=cost_halo_junction_penalty,
            total_cost=total_cost,
        )

        log.debug(
            f"cost: wl={cost_wirelength:.1f}, macro={cost_macro:.1f}, halo={cost_halo:.1f}, "
            f"congestion={cost_congestion:.1f}, crossing={cost_crossing:.1f}, "
            f"penalties={cost_macro_junction_penalty + cost_halo_junction_penalty:.1f}, "
            f"total={total_cost:.1f}"
        )

        return metrics, crossing_points

    def calculate_total_cost(
        self,
        topology: Topology,
        context: RoutingContext,
        debug_plot: bool = False,
        plot_filename_prefix: str | None = None,
    ):
        """
        Calculates the total cost of a given topology and can optionally generate a debug plot.
        """
        total_cost = 0.0
        paths_with_metrics = []
        crossing_points = []

        # Build adjacency list for the topology
        adj = defaultdict(set)
        all_nodes: Set[Junction | Pin] = set(topology.junctions)
        for j in topology.junctions:
            for child in j.children:
                adj[j].add(child)
                adj[child].add(j)
                all_nodes.add(child)

        processed_edges = set()
        for u in all_nodes:
            if u not in adj:
                continue
            for v in adj[u]:
                if tuple(sorted((id(u), id(v)))) in processed_edges:
                    continue
                processed_edges.add(tuple(sorted((id(u), id(v)))))

                p1 = u.location if isinstance(u, Junction) else u.draw.geom
                if p1 and not isinstance(p1, Point):
                    p1 = p1.centroid

                p2 = v.location if isinstance(v, Junction) else v.draw.geom
                if p2 and not isinstance(p2, Point):
                    p2 = p2.centroid

                if p1 is None or p2 is None:
                    continue

                p1_macro_to_ignore = context.pin_macros.get(u) if isinstance(u, Pin) else None
                p2_macro_to_ignore = context.pin_macros.get(v) if isinstance(v, Pin) else None

                candidate_paths = generate_l_paths(p1, p2)

                if not candidate_paths:
                    path = LineString([p1, p2])
                    metrics, new_crossings = self.calculate_path_cost(
                        path,
                        context,
                        p1_macro_to_ignore,
                        p2_macro_to_ignore,
                    )
                    total_cost += metrics.total_cost
                    if debug_plot:
                        paths_with_metrics.append((path, metrics))
                        crossing_points.extend(new_crossings)
                    continue

                metrics1, crossings1 = self.calculate_path_cost(
                    candidate_paths[0],
                    context,
                    p1_macro_to_ignore,
                    p2_macro_to_ignore,
                )
                metrics2, crossings2 = self.calculate_path_cost(
                    candidate_paths[1],
                    context,
                    p1_macro_to_ignore,
                    p2_macro_to_ignore,
                )

                if metrics1.total_cost < metrics2.total_cost:
                    best_path, best_metrics, best_crossings = candidate_paths[0], metrics1, crossings1
                else:
                    best_path, best_metrics, best_crossings = candidate_paths[1], metrics2, crossings2

                total_cost += best_metrics.total_cost
                if debug_plot:
                    paths_with_metrics.append((best_path, best_metrics))
                    crossing_points.extend(best_crossings)

        if debug_plot:
            self._debugger.plot_cost_calculation(
                topology, paths_with_metrics, crossing_points, context, plot_filename_prefix
            )

        return total_cost
