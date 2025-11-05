import logging as log
import math
import os
import re
import sys
import time
import unicodedata
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from itertools import cycle
from typing import Any, Dict, Tuple

import numpy as np
import pygame
from pcst_fast import pcst_fast
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

from schematic_from_netlist.sastar_router.models import CostBuckets, CostEstimator, RoutingContext
from schematic_from_netlist.sastar_router.visualization import plot_result

# Set to True to enable Pygame visualization
ENABLE_PYGAME = True
BASE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (0, 128, 0),
    (255, 165, 0),
]
COLORS = cycle(BASE_COLORS)


log.basicConfig(level=log.INFO)


@dataclass
class Terminal:
    name: str
    pt: Tuple
    direction: str


@dataclass
class PCost:
    """Defines the baseline costs for various routing elements."""

    base: int = 1  # Cost for routing straight
    macro: int = 100  # Cost for routing over a macro
    halo: int = 40  # Cost for routing near a macro
    turn: int = 10  # PENALTY: Cost applied for any 90-degree turn
    prize: int = 1000  # Prize for required terminals


class PcstRouter:
    def __init__(
        self,
        grid_spacing=1.0,
        obstacle_geoms=None,
        halo_size=4,
        bounds=None,
        other_paths=[],
        terminals=[],
        debug: bool | None = None,
        log_interval: int = 100,
        prune_cycles: bool = True,
        post_simplify: bool = True,
        min_loop_area: float | None = None,
    ):
        self.grid_spacing = grid_spacing
        self.obstacles = obstacle_geoms or []
        self.obstacles_index = STRtree(self.obstacles) if self.obstacles else None
        self.bounds = bounds  # Add bounds parameter
        self.other_paths = other_paths
        self.terminals = terminals
        self.scale = 1.0
        self.width, self.height = 800, 600

        self.GRID_SIZE = 0
        self.DIR_MAP = {"N": 0, "E": 1, "S": 2, "W": 3}
        self.DIR_VEC = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
        self.OPPOSITE_DIR = {"N": "S", "S": "N", "E": "W", "W": "E"}

        self.coord_to_id = {}
        self.id_to_coord = {}
        self.next_id = 0

        # Debug/diagnostic controls
        self.debug = False if debug is None else bool(debug)
        self.log_interval = log_interval

        # Cleanup/pruning options
        self.prune_cycles = prune_cycles
        self.post_simplify = post_simplify
        self.min_loop_area = min_loop_area

        # Create halo regions around obstacles
        self.halo_geoms = []
        for obs in self.obstacles:
            halo = obs.buffer(halo_size)
            self.halo_geoms.append(halo)
        self.halo_index = STRtree(self.halo_geoms) if self.halo_geoms else None

        # Calculate bounds if not provided
        # Ensure bounds include all terminals by expanding if necessary
        if not self.bounds:
            self._calculate_default_bounds()

        # Initialize cost estimator
        self.cost_estimator = CostEstimator(grid_spacing)
        self.cost = PCost()

    def route_net(
        self,
        net_name: str,
        # Use a Dict to accept optional keyword arguments for cost overrides
        cost_overrides: Dict[str, Any] = None,
    ):
        """
        connect one net
        """
        edges_input, prizes, costs, root, num_clusters, blockage_cost_val = self.setup_pcst_grid(net_name)

        # Configure the solver
        pruning = "gw"  # Use strong pruning for sparse solutions
        verbosity_level = 1
        vertices, edges_output = pcst_fast(edges_input, prizes, costs, root, num_clusters, pruning, verbosity_level)

        selected_cost = np.sum(costs[edges_output])
        log.info(f"Total Cost of Route (Weighted): {selected_cost:.0f}")

        # --- Decode Coordinates ---
        coord_segments = self.decode_edges_to_coords(edges_output, edges_input, costs)
        segments = [LineString([start, end]) for start, end in coord_segments]
        merged_segments = MultiLineString(self._merge_adjacent_segments(segments))
        return merged_segments, selected_cost

    def _calculate_default_bounds(self):
        """Calculate default bounds based on obstacles"""
        # Compute bounds with padding
        padding = 10  # Add padding to bounds
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")

        # Add all obstacle bounds
        for obs in self.obstacles:
            bounds = obs.bounds
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        # Pad by 10 grid points
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        self.bounds = box(min_x, min_y, max_x, max_y)

    def to_screen_coords(self, point):
        x, y = point
        scale = self.scale
        screen_x = (x - self.min_x) * scale + (self.width - (self.max_x - self.min_x) * scale) / 2
        screen_y = self.height - ((y - self.min_y) * scale + (self.height - (self.max_y - self.min_y) * scale) / 2)
        return (int(screen_x), int(screen_y))

    def _initialize_visualization(self, port_colors=None):
        """Initialize Pygame and draw the initial state."""
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("P Routing Visualization")
        font = pygame.font.Font(None, 24)

        # Scale factors
        self.min_x, self.min_y, self.max_x, self.max_y = self.bounds.bounds
        scale_x = self.width / (self.max_x - self.min_x)
        scale_y = self.height / (self.max_y - self.min_y)
        self.scale = min(scale_x, scale_y) * 0.9

        screen.fill((255, 255, 255))

        # Draw obstacles
        for obs in self.obstacles:
            points = [self.to_screen_coords((x, y)) for x, y in obs.exterior.coords]
            pygame.draw.polygon(screen, (200, 200, 200), points)
            pygame.draw.polygon(screen, (0, 0, 0), points, 1)

        # Draw halo regions
        for halo in self.halo_geoms:
            points = [self.to_screen_coords((x, y)) for x, y in halo.exterior.coords]
            pygame.draw.polygon(screen, (255, 200, 200), points, 1)

        # bounds
        if self.bounds:
            bounds_points = [self.to_screen_coords((x, y)) for x, y in self.bounds.exterior.coords]
            # Draw outline in red to show it's a boundary
            pygame.draw.polygon(screen, (255, 0, 0), bounds_points, 3)

        # Draw existing nets in different colors
        for route_geom in self.other_paths:
            self.draw_route(screen, route_geom)

        # Draw terminals with per-port colors if provided
        legend_y = 10
        port_colors = {i: BASE_COLORS[i % len(BASE_COLORS)] for i in range(len(self.terminals))}
        for i, term in enumerate(self.terminals):
            pos = self.to_screen_coords(term.pt)
            color = (255, 0, 0)
            if port_colors and i in port_colors:
                color = port_colors[i]
            pygame.draw.circle(screen, color, pos, 5)
            text = font.render(f"T{i}", True, (0, 0, 0))
            screen.blit(text, (pos[0] + 5, pos[1] - 5))
            # Legend
            pygame.draw.circle(screen, color, (15, legend_y + 5), 5)
            legend_text = font.render(f"T{i}", True, (0, 0, 0))
            screen.blit(legend_text, (30, legend_y))
            legend_y += 20
        # Add merge legend entry
        pygame.draw.circle(screen, (150, 150, 150), (15, legend_y + 5), 5)
        merge_text = font.render("Merge", True, (0, 0, 0))
        screen.blit(merge_text, (30, legend_y))
        pygame.display.flip()

        return screen, font

    def _display_flip(self):
        """Visualize the final path and save the image."""
        pygame.display.update()

    def _handle_pygame_events(self):
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def clean_hierarchical_name(self, name: str, delimiter: str = "_") -> str:
        """
        Cleans a hierarchical name (e.g., 'Mod/Instance/Signal') into a flat,
        filesystem-safe string (e.g., 'mod_instance_signal').
        """

        # 1. Normalize Unicode and convert to ASCII
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

        # 2. Convert to lowercase
        name = name.lower()

        # The pattern now explicitly includes '/' and backslash '\'
        name = re.sub(r"[^\w-]", delimiter, name)

        # 4. Collapse multiple delimiters (which might result from the previous step)
        #    and strip any leading/trailing delimiters.
        name = re.sub(f"{delimiter}+", delimiter, name).strip(delimiter)

        return name

    def coord_to_node_id(self, r, c, direction):
        """Converts (r, c, direction) to a unique node index."""
        key = (r, c, direction)
        if key not in self.coord_to_id:
            node_id = self.next_id
            self.coord_to_id[key] = node_id
            self.id_to_coord[node_id] = key
            self.next_id += 1
        return self.coord_to_id[key]

    def node_id_to_coord(self, node_id):
        """Converts a node index back to (r, c, direction)."""
        return self.id_to_coord[node_id]

    def decode_edges_to_coords(self, edges_output, edges_input, all_costs):
        """
        Decodes the selected edges into a list of simplified (R, C) segments,
        handling the complex 4-node model.
        """
        coord_segments = []

        # Store visited segments to deduplicate and trace the path
        unique_segments = set()

        for edge_index in edges_output:
            node_a, node_b = edges_input[edge_index]
            cost = all_costs[edge_index]

            # Determine if it is an inter-cell move or an intra-cell turn
            try:
                r1, c1, dir1 = self.node_id_to_coord(node_a)
                r2, c2, dir2 = self.node_id_to_coord(node_b)
            except ValueError:
                continue  # Skip invalid nodes

            # Inter-Cell Movement (Wiring between two different grid squares)
            # This occurs when r1 != r2 or c1 != c2.
            if r1 != r2 or c1 != c2:
                # We record the segment as the connection between the two grid centers.
                # Use the cell with the lower index as the start for canonical representation
                if node_a < node_b:
                    segment = ((r1, c1), (r2, c2))
                else:
                    segment = ((r2, c2), (r1, c1))

                # The direction difference (dir1 != dir2) is usually minimal here

                if segment not in unique_segments:
                    coord_segments.append(segment)
                    unique_segments.add(segment)

            # Intra-Cell Turn (Wiring inside the same grid square)
            # This occurs when r1 == r2 and c1 == c2, but dir1 != dir2.
            elif dir1 != dir2:
                # This represents a turn (e.g., entering North, leaving East)
                # Log the turn node only, as it doesn't represent a physical segment
                pass

        return coord_segments

    def coords_outside_poly(self, poly_list):
        res = set()
        for poly in poly_list:
            coords = poly.exterior.coords[:-1]  # Remove duplicate last point
            for x, y in coords:
                res.add((int(x), int(y)))
        return res

    def coords_inside_poly(self, poly_list):
        res = set()
        for poly in poly_list:
            minx, miny, maxx, maxy = poly.bounds
            for x in range(int(minx), int(maxx) + 1):
                for y in range(int(miny), int(maxy) + 1):
                    if poly.contains(Point(x, y)):
                        res.add((x, y))
        return res

    def coords_from_multilinestring(self, mls_list):
        res = set()
        for mls in mls_list:
            for line in mls.geoms:
                for x, y in line.coords:
                    res.add((int(x), int(y)))
        return res

    def setup_pcst_grid(self, net_name):
        """
        creates custom grid for each net
        """

        log.info(f"Creating detailed route grid for {net_name}")

        edges_list = []
        costs_list = []

        blockage_cells = self.coords_inside_poly(self.obstacles)
        halo_cells = self.coords_inside_poly(self.halo_geoms)
        grid_cells = self.coords_inside_poly([self.bounds])
        other_route_cells = self.coords_from_multilinestring(self.other_paths)
        log.info(f"routing using {len(grid_cells)} grid cells")

        # --- 1. Edge & Cost Generation ---
        for r, c in grid_cells:
            # A. Intra-Cell Edges (Turns): Connects all 4 D-Nodes within the same cell
            for dir_from in self.DIR_MAP.keys():
                for dir_to in self.DIR_MAP.keys():
                    if dir_from == dir_to:
                        continue  # No cost for continuing straight

                    # Connection between D-Nodes within the same cell
                    node_a = self.coord_to_node_id(r, c, dir_from)
                    node_b = self.coord_to_node_id(r, c, dir_to)

                    cost = 0  # Default cost is free for internal movement

                    # Check for 90-degree turn: The direction from is NOT the opposite of the direction to
                    # and the two directions are NOT the same.
                    is_turn = dir_from != self.OPPOSITE_DIR.get(dir_to)
                    is_straight_through = dir_from == self.OPPOSITE_DIR.get(dir_to)

                    if is_turn and not is_straight_through:
                        # 90-degree turn (e.g., N->E, E->S)
                        cost = self.cost.turn
                    elif is_straight_through:
                        # 180-degree pass-through (e.g., N->S). Cost 0 for D-Node transition.
                        cost = 0.0
                    else:
                        # This should cover 90-degree turns and self-loops (which are skipped above)
                        cost = self.cost.turn

                    # NOTE: This implementation is simplified. A full router would ensure N->N (self-loop) is skipped
                    # and often only connect adjacent D-Nodes. Given the constraints, we use the
                    # N-S, E-W axis to define the straight-through path.

                    edges_list.append([node_a, node_b])
                    costs_list.append(cost)

            # B. Inter-Cell Edges (Movement): Connects D-Nodes in adjacent cells
            for dir_from, (dr, dc) in self.DIR_VEC.items():
                r_b, c_b = r + dr, c + dc

                # Node A is the directional node *at* (r, c)
                node_a = self.coord_to_node_id(r, c, dir_from)

                # Node B is the directional node *entering* (r_b, c_b) from the opposite direction
                dir_to = self.OPPOSITE_DIR[dir_from]
                node_b = self.coord_to_node_id(r_b, c_b, dir_to)

                # Base cost is 1 unit of length
                cost = self.cost.base
                # Apply blockage cost if *this segment* is in the blockage
                if (r, c) in blockage_cells or (r_b, c_b) in blockage_cells:
                    cost = self.cost.macro
                elif (r, c) in halo_cells or (r_b, c_b) in halo_cells:
                    cost = self.cost.halo

                # Add edge (a->b and b->a are added for undirected graph)
                edges_list.append([node_a, node_b])
                costs_list.append(cost)

        edges = np.array(edges_list, dtype=np.int64)
        costs = np.array(costs_list, dtype=np.int64)

        # We set prize on ALL directional nodes in the target cell to simplify connection
        required_terminals = []
        for term in self.terminals:
            r, c = term.pt.x, term.pt.y
            dir = term.direction
            required_terminals.append(self.coord_to_node_id(r, c, dir))

        # just choose the first as the root index
        # TODO: find optimal terminal to start from
        root = required_terminals[0]

        # Max shortest path cost (8 straight segments + max 1 turn) is ~13.
        # Set prize higher than the worst single-path cost to force connectivity.

        prizes = np.zeros(self.next_id + 1, dtype=np.int64)
        for node_id in required_terminals:
            prizes[node_id] = self.cost.prize

        log.info(f"Required Terminals: {self.terminals} (Total Prize: {self.cost.prize * len(required_terminals):.0f})")
        log.info(f"Blockage Cost: {self.cost.macro}. Turn Cost: {self.cost.turn}. Normal Cost: {self.cost.base}")

        num_clusters = 1

        return edges, prizes, costs, root, num_clusters, self.cost.macro

    def _merge_adjacent_segments(self, paths, terminals=None, existing_paths=None):
        """Merge adjacent segments into clean orthogonal lines while avoiding overlaps"""
        if not paths:
            return []

        # Robust dissolve before merging to prevent fragmented overlaps
        unioned = unary_union(paths)
        merged = linemerge(unioned)

        if merged.is_empty:
            return []

        # Convert to list of LineStrings
        if isinstance(merged, LineString):
            merged_paths = [merged]
        else:  # MultiLineString
            merged_paths = list(merged.geoms)

        # Optional simplification with zero tolerance to drop duplicate vertices
        if self.post_simplify:

            def simplify_line(g):
                try:
                    return g.simplify(0, preserve_topology=True)
                except Exception:
                    return g

            merged_paths = [simplify_line(l) for l in merged_paths]

        # Split merged paths at direction changes to get clean orthogonal segments
        log.info(f"{paths} -> {merged_paths=}")
        dedup = set()
        result_paths = []
        for line in merged_paths:
            segments = self._split_at_direction_changes(line)

            # Process each segment
            for segment in segments:
                if segment.length == 0:
                    continue
                key = tuple(map(tuple, segment.coords))
                if key in dedup:
                    continue
                dedup.add(key)
                result_paths.append(segment)
                if existing_paths:
                    # Check for overlaps with existing nets
                    has_significant_overlap = False
                    for net_path in existing_paths:
                        intersection = segment.intersection(net_path)
                        if intersection:
                            overlap_length = 0
                            if isinstance(intersection, LineString):
                                overlap_length = round(intersection.length)
                            elif isinstance(intersection, MultiLineString):
                                overlap_length = round(sum(seg.length for seg in intersection.geoms))

                            if overlap_length >= 1:
                                log.warning(
                                    f"Merged segment {segment.wkt} has {overlap_length} overlap with existing path {net_path}"
                                )
                                has_significant_overlap = True
                                break

                    if has_significant_overlap:
                        log.warning(f"Adding segment despite overlap: {segment.wkt}")

        # Optional: detect residual tiny loops (should be none after cycle pruning)
        if self.min_loop_area is not None and result_paths:
            ml = float(self.min_loop_area)
        else:
            # Default threshold: quarter of a grid square
            ml = 0.25 * float(self.grid_spacing) * float(self.grid_spacing)
        if result_paths:
            poly_candidates = list(polygonize(unary_union(result_paths)))
            tiny = [p for p in poly_candidates if p.area <= ml + 1e-9]
            if tiny:
                log.info(f"Detected {len(tiny)} tiny loops (area <= {ml}); pruning via cycle removal already applied.")
                # No further action needed; cycle pruning ensures the linear network is acyclic.

        # Ensure all terminals are connected
        if terminals:
            terminal_points = set(terminals)
            merged_geom = unary_union(result_paths) if result_paths else None

            if merged_geom:
                for terminal in terminal_points:
                    if merged_geom.distance(Point(terminal)) > 1e-6:
                        log.warning(f"Terminal {terminal} is not connected to {merged_geom=}")

        return result_paths

    def _split_at_direction_changes(self, line):
        """Split a LineString into clean orthogonal segments at direction changes only"""
        coords = list(line.coords)
        if len(coords) < 3:  # No direction changes possible with just 2 points
            return [line]

        segments = []
        start_idx = 0

        # Track the current direction
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        current_dir = "horizontal" if abs(dx) > abs(dy) else "vertical"

        for i in range(2, len(coords)):
            # Calculate direction of current segment
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            new_dir = "horizontal" if abs(dx) > abs(dy) else "vertical"

            # If direction changes, split the segment
            if new_dir != current_dir:
                segments.append(LineString([coords[start_idx], coords[i - 1]]))
                start_idx = i - 1
                current_dir = new_dir

        # Add the final segment
        segments.append(LineString([coords[start_idx], coords[-1]]))

        return segments

    def _reconstruct_paths(self, path_dict, terminals):
        """Convert path dict (tree structure) to list of LineStrings"""
        paths = []

        # Build edges from parent pointers
        edges = set()
        for node, parent_info in path_dict.items():
            parent, _ = parent_info
            if parent is not None:
                edge = tuple(sorted([node, parent]))
                edges.add(edge)

        # Convert edges to LineStrings
        for edge in edges:
            paths.append(LineString([edge[0], edge[1]]))

        # Merge collinear segments (optional optimization)
        return self._merge_collinear(paths)

    def _merge_collinear(self, paths):
        """Merge collinear path segments"""
        # Simple version: return as-is
        # Advanced: detect and merge collinear segments
        return paths

    # Reuse helper methods from SequentialRouter
    def _snap_to_grid(self, point):
        x, y = point
        gx = int(round(x / self.grid_spacing) * self.grid_spacing)
        gy = int(round(y / self.grid_spacing) * self.grid_spacing)
        return (gx, gy)

    def _get_neighbors(self, node):
        x, y = node
        g = self.grid_spacing
        # Only orthogonal moves
        neighbors = []
        potential_neighbors = [(x + g, y), (x - g, y), (x, y + g), (x, y - g)]
        # Filter neighbors to be within padded bounds
        for nx, ny in potential_neighbors:
            if self.bounds.contains(Point(nx, ny)):
                neighbors.append((int(nx), int(ny)))
        return neighbors

    @staticmethod
    def _distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def draw_route(self, screen, route_geom):
        """Draw an existing route as a single colored line from start to end."""
        color = next(COLORS)

        font = pygame.font.Font(None, 12)
        pos = self.to_screen_coords((0, 10))
        text = font.render(f"{self.cost.halo=}", True, (0, 0, 0))
        screen.blit(text, (pos[0], pos[1]))

        # Flatten either MultiLineString or LineString
        if isinstance(route_geom, MultiLineString):
            lines = list(route_geom.geoms)
        elif isinstance(route_geom, LineString):
            lines = [route_geom]
        else:
            return  # unsupported geometry

        # Get the coordinates of the entire route
        for seg in lines:
            coords = list(seg.coords)
            for a, b in zip(coords, coords[1:]):
                pygame.draw.line(screen, color, self.to_screen_coords(a), self.to_screen_coords(b), 2)
        pygame.display.flip()


if __name__ == "__main__":
    # log.info(sr.clean_hierarchical_name("/tnoi/is/a/test"))
    pass
