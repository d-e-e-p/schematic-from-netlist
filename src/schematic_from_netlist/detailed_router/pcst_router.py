import logging as log
import math
import os
import re
import sys
import time
import unicodedata
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from heapq import heappop, heappush
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
from pcst_fast import pcst_fast
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

from schematic_from_netlist.detailed_router.pcst_debug import PCSTGridDebugger
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


@dataclass
class PcstInputs:
    """All inputs required to run pcst_fast()."""

    edges: Optional[List[Tuple[int, int]]] = field(default=None)
    prizes: Optional[List[float]] = field(default=None)
    costs: Optional[List[float]] = field(default=None)
    root: Optional[int] = None
    num_clusters: int = 1
    pruning: str = "gw"
    verbosity_level: int = 1

    def run(self):
        """Run pcst_fast using current attributes."""
        if self.edges is None or self.prizes is None or self.costs is None:
            raise ValueError("edges, prizes, and costs must be set before running pcst_fast.")

        edges_arr = np.array(self.edges, dtype=int)
        prizes_arr = np.array(self.prizes, dtype=float)
        costs_arr = np.array(self.costs, dtype=float)

        vertices, edges = pcst_fast(
            edges_arr,
            prizes_arr,
            costs_arr,
            self.root,
            self.num_clusters,
            self.pruning,
            self.verbosity_level,
        )
        return vertices, edges


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

        self.GRID_SIZE = 1
        self.DIR_MAP = {"N": 0, "E": 1, "S": 2, "W": 3}
        self.DIR_VEC = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
        self.OPPOSITE_DIR = {"N": "S", "S": "N", "E": "W", "W": "E"}
        # Direction to axis mapping
        # TODO: hunt down C dir pins
        self.DIR_TO_AXIS = {"C": "EW", "N": "NS", "S": "NS", "E": "EW", "W": "EW"}
        self.AXIS_TO_DIRS = {"NS": ["N", "S"], "EW": ["E", "W"]}

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
        pcst = self.setup_pcst_grid(net_name)

        # Configure the solver
        vertices, edges_output = pcst.run()

        debugger = PCSTGridDebugger(self, pcst)
        debugger.set_solution(vertices, edges_output)
        debugger.print_debug_stats(net_name)
        debugger.visualize_grid(net_name)

        selected_cost = np.sum(pcst.costs[edges_output])
        log.info(f"Total Cost of Route (Weighted): {selected_cost:.0f}")

        # --- Decode Coordinates ---
        coord_segments = self.decode_edges_to_coords(edges_output, pcst.edges, pcst.costs)
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

    def coord_to_node_id(self, r, c, axis_or_direction):
        """
        Converts (r, c, axis) to a unique node index.

        Args:
            r, c: Grid coordinates
            axis_or_direction: Either 'NS', 'EW' (axis) or 'N', 'S', 'E', 'W' (direction)

        Returns:
            node_id: Unique integer node identifier
        """
        # Convert direction to axis if needed
        if axis_or_direction in self.DIR_TO_AXIS:
            axis = self.DIR_TO_AXIS[axis_or_direction]
        else:
            axis = axis_or_direction

        key = (r, c, axis)
        if key not in self.coord_to_id:
            node_id = self.next_id
            self.coord_to_id[key] = node_id
            self.id_to_coord[node_id] = key
            self.next_id += 1
        return self.coord_to_id[key]

    def node_id_to_coord(self, node_id):
        """
        Converts a node index back to (r, c, axis).

        Returns:
            (r, c, axis): Tuple with row, column, and axis ('NS' or 'EW')
        """
        return self.id_to_coord[node_id]

    def setup_pcst_grid(self, net_name):
        """
        Creates optimized cost grid with merged opposite-direction nodes.

        Graph structure:
        - 2 nodes per cell: NS (North-South axis) and EW (East-West axis)
        - Intra-cell edges: NS ↔ EW with turn cost
        - Inter-cell edges: Axis-aligned connections with base/blockage/halo costs

        Returns:
            edges: np.array of [node_a, node_b] pairs
            prizes: np.array of prize values per node
            costs: np.array of edge costs
            root: Root node ID for PCST
            num_clusters: Number of clusters (always 1)
        """

        log.info(f"Creating optimized route grid for {net_name}")

        edges_list = []
        costs_list = []

        blockage_cells = set(self.coords_inside_poly(self.obstacles))
        halo_cells = set(self.coords_inside_poly(self.halo_geoms))
        grid_cells = list(self.coords_inside_poly([self.bounds]))

        log.info(f"routing using {len(grid_cells)} grid cells")

        # --- 1. Create nodes and intra-cell edges (turns) ---
        for r, c in grid_cells:
            # Create two merged nodes per cell
            node_ns = self.coord_to_node_id(r, c, "NS")
            node_ew = self.coord_to_node_id(r, c, "EW")

            # Intra-cell edge: Turn between NS and EW axes
            # This represents a 90-degree turn
            edges_list.append([node_ns, node_ew])
            costs_list.append(self.cost.turn)

        # --- 2. Inter-cell edges (straight movement between adjacent cells) ---
        grid_cells_set = set(grid_cells)

        for r, c in grid_cells:
            # North-South axis connections
            # Connect NS node at (r,c) to NS nodes in adjacent cells (above/below)
            for dr in [-1, 1]:  # North and South
                r_b, c_b = r + dr, c
                if (r_b, c_b) in grid_cells_set:
                    node_a = self.coord_to_node_id(r, c, "NS")
                    node_b = self.coord_to_node_id(r_b, c_b, "NS")

                    # Determine cost based on blockage/halo
                    cost = self.cost.base
                    if (r, c) in blockage_cells or (r_b, c_b) in blockage_cells:
                        cost = self.cost.macro
                    elif (r, c) in halo_cells or (r_b, c_b) in halo_cells:
                        cost = self.cost.halo

                    edges_list.append([node_a, node_b])
                    costs_list.append(cost)

            # East-West axis connections
            # Connect EW node at (r,c) to EW nodes in adjacent cells (left/right)
            for dc in [-1, 1]:  # West and East
                r_b, c_b = r, c + dc
                if (r_b, c_b) in grid_cells_set:
                    node_a = self.coord_to_node_id(r, c, "EW")
                    node_b = self.coord_to_node_id(r_b, c_b, "EW")

                    # Determine cost based on blockage/halo
                    cost = self.cost.base
                    if (r, c) in blockage_cells or (r_b, c_b) in blockage_cells:
                        cost = self.cost.macro
                    elif (r, c) in halo_cells or (r_b, c_b) in halo_cells:
                        cost = self.cost.halo

                    edges_list.append([node_a, node_b])
                    costs_list.append(cost)

        edges = np.array(edges_list, dtype=np.int64)
        costs = np.array(costs_list, dtype=np.int64)

        # --- 3. Map terminals to merged nodes ---
        required_terminals = []
        for term in self.terminals:
            r, c = term.pt.x, term.pt.y
            direction = term.direction

            # Convert direction to axis (handles both 'N'/'S' and 'E'/'W')
            axis = self.DIR_TO_AXIS.get(direction)
            if axis is None:
                raise ValueError(f"Unknown terminal direction: {direction}")

            node_id = self.coord_to_node_id(r, c, axis)
            required_terminals.append(node_id)

        # Choose first terminal as root
        root = required_terminals[0]

        # Set prizes on terminal nodes
        prizes = np.zeros(self.next_id, dtype=np.int64)
        for node_id in required_terminals:
            prizes[node_id] = self.cost.prize

        # Calculate theoretical savings
        theoretical_nodes = len(grid_cells) * 4
        actual_nodes = self.next_id

        log.info(f"Optimized Grid Stats:")
        log.info(f"  Grid cells: {len(grid_cells)}")
        log.info(f"  Nodes (4-dir design): {theoretical_nodes}")
        log.info(f"  Nodes (optimized): {actual_nodes} ({actual_nodes / theoretical_nodes * 100:.1f}%)")
        log.info(f"  Edges: {len(edges)}")
        log.info(f"  Required Terminals: {len(required_terminals)} (Total Prize: {self.cost.prize * len(required_terminals):.0f})")
        log.info(f"  Costs: Base={self.cost.base}, Turn={self.cost.turn}, Halo={self.cost.halo}, Macro={self.cost.macro}")
        pcst = PcstInputs(edges, prizes, costs, root)
        return pcst

    def decode_edges_to_coords(self, edges_output, edges_input, all_costs):
        """
        Decodes the selected edges into a list of (r, c) coordinate segments.

        With merged nodes:
        - Inter-cell edges represent physical wire segments
        - Intra-cell edges represent turns (no physical segment)

        Args:
            edges_output: List of edge indices selected by PCST
            edges_input: Array of [node_a, node_b] edge pairs
            all_costs: Array of edge costs

        Returns:
            coord_segments: List of ((r1, c1), (r2, c2)) tuples representing wire segments
        """
        coord_segments = []
        unique_segments = set()
        turn_points = []  # Track where turns occur

        for edge_index in edges_output:
            node_a, node_b = edges_input[edge_index]
            cost = all_costs[edge_index]

            try:
                r1, c1, axis1 = self.node_id_to_coord(node_a)
                r2, c2, axis2 = self.node_id_to_coord(node_b)
            except (KeyError, ValueError):
                log.warning(f"Invalid node ID in edge {edge_index}: {node_a} or {node_b}")
                continue

            # Inter-Cell Movement (Physical wire segment)
            # This occurs when moving between different cells
            if r1 != r2 or c1 != c2:
                # Create canonical segment representation
                if (r1, c1) < (r2, c2):
                    segment = ((r1, c1), (r2, c2))
                else:
                    segment = ((r2, c2), (r1, c1))

                if segment not in unique_segments:
                    coord_segments.append(segment)
                    unique_segments.add(segment)

            # Intra-Cell Turn (Axis change within same cell)
            # This occurs when switching between NS and EW axes
            elif axis1 != axis2:
                # Record turn location for debugging/visualization
                turn_points.append((r1, c1, axis1, axis2))
                # No physical segment added - turn is implicit at this cell
                log.debug(f"Turn at ({r1},{c1}): {axis1} → {axis2}")

        log.info(f"Decoded {len(coord_segments)} wire segments and {len(turn_points)} turns")

        return coord_segments

    def get_path_directions(self, edges_output, edges_input):
        """
        Extract the sequence of directions taken by the path.
        Useful for debugging and visualization.

        Returns:
            path_info: List of (r, c, axis, cost) tuples in path order
        """
        # Build adjacency for path reconstruction
        from collections import defaultdict

        graph = defaultdict(list)

        for edge_index in edges_output:
            node_a, node_b = edges_input[edge_index]
            try:
                coord_a = self.node_id_to_coord(node_a)
                coord_b = self.node_id_to_coord(node_b)
                graph[coord_a].append(coord_b)
                graph[coord_b].append(coord_a)
            except (KeyError, ValueError):
                continue

        # Simple path extraction (assumes single path)
        path_info = []
        visited = set()

        # Start from a terminal
        if self.terminals:
            r, c = self.terminals[0].pt.x, self.terminals[0].pt.y
            axis = self.DIR_TO_AXIS[self.terminals[0].direction]
            start = (r, c, axis)

            # DFS to build path
            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                path_info.append(node)

                for neighbor in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor)

            dfs(start)

        return path_info

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
