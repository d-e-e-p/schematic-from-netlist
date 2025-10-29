import logging as log
import math
import sys
import time
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import cycle

import numpy as np
import pygame
from models import CostBuckets, CostEstimator, RoutingContext
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree
from visualization import plot_result

# Set to True to enable Pygame visualization
ENABLE_PYGAME = False
COLORS = cycle(
    [
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
)


log.basicConfig(level=log.INFO)


class SimultaneousRouter:
    def __init__(self, grid_spacing=1.0, obstacle_geoms=None, halo_size=4, bounds=None):
        self.grid_spacing = grid_spacing
        self.obstacles = obstacle_geoms or []
        self.obstacles_index = STRtree(self.obstacles) if self.obstacles else None
        self.bounds = bounds  # Add bounds parameter
        self.scale = 1.0
        self.width, self.height = 800, 600

        # Create halo regions around obstacles
        self.halo_geoms = []
        for obs in self.obstacles:
            halo = obs.buffer(halo_size)
            self.halo_geoms.append(halo)
        self.halo_index = STRtree(self.halo_geoms) if self.halo_geoms else None

        # Initialize cost estimator
        self.cost_estimator = CostEstimator(grid_spacing)

        # Calculate bounds if not provided
        if not self.bounds:
            self._calculate_default_bounds()

        # Ensure bounds include all terminals by expanding if necessary
        # We'll update this when routing with actual terminals

    def _calculate_default_bounds(self):
        """Calculate default bounds based on obstacles"""
        # Compute bounds with padding
        self.min_x = float("inf")
        self.max_x = float("-inf")
        self.min_y = float("inf")
        self.max_y = float("-inf")

        # Add all obstacle bounds
        for obs in self.obstacles:
            bounds = obs.bounds
            self.min_x = min(self.min_x, bounds[0])
            self.min_y = min(self.min_y, bounds[1])
            self.max_x = max(self.max_x, bounds[2])
            self.max_y = max(self.max_y, bounds[3])

        # Pad by 20 grid points
        padding = 20 * self.grid_spacing
        self.min_x -= padding
        self.max_x += padding
        self.min_y -= padding
        self.max_y += padding

    def to_screen_coords(self, point):
        x, y = point
        scale = self.scale
        screen_x = (x - self.min_x) * scale + (self.width - (self.max_x - self.min_x) * scale) / 2
        screen_y = self.height - ((y - self.min_y) * scale + (self.height - (self.max_y - self.min_y) * scale) / 2)
        return (int(screen_x), int(screen_y))

    def route_net(self, net, existing_paths=None):
        """
        More efficient approach: track which terminals are connected to each node
        """
        terminals = net.pins
        if len(terminals) < 2:
            return [], 0, {}  # Return empty paths, zero cost, and empty step costs

        # If bounds were provided, use them
        if self.bounds:
            self.min_x, self.min_y, self.max_x, self.max_y = self.bounds
            padding = 5 * self.grid_spacing
            self.min_x -= padding
            self.max_x += padding
            self.min_y -= padding
            self.max_y += padding

        # Snap terminals to grid
        terminals = [self._snap_to_grid(t) for t in terminals]
        terminal_set = set(terminals)

        # Create routing context
        context = RoutingContext(
            obstacles_index=self.obstacles_index,
            obstacles=self.obstacles,
            existing_paths_index=STRtree(existing_paths) if existing_paths else None,
            existing_paths=existing_paths or [],
            halo_geoms=self.halo_geoms,
            halo_index=self.halo_index,
            terminal_set=terminal_set,
        )

        # Initialize pygame if enabled
        if ENABLE_PYGAME:
            pygame.init()
            screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("A* Routing Visualization")
            font = pygame.font.Font(None, 24)

            # Scale factors
            scale_x = self.width / (self.max_x - self.min_x)
            scale_y = self.height / (self.max_y - self.min_y)
            self.scale = min(scale_x, scale_y) * 0.9

        if ENABLE_PYGAME:
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

            # Draw existing nets in different colors
            for route_geom in existing_paths:
                self.draw_route(screen, route_geom)

            # Draw terminals
            for i, term in enumerate(terminals):
                pos = self.to_screen_coords(term)
                pygame.draw.circle(screen, (255, 0, 0), pos, 5)
                text = font.render(f"T{i}", True, (0, 0, 0))
                screen.blit(text, (pos[0] + 5, pos[1] - 5))
        # Update bounds
        min_x_term = min(t[0] for t in terminals)
        max_x_term = max(t[0] for t in terminals)
        min_y_term = min(t[1] for t in terminals)
        max_y_term = max(t[1] for t in terminals)
        self.min_x = min(self.min_x, min_x_term)
        self.max_x = max(self.max_x, max_x_term)
        self.min_y = min(self.min_y, min_y_term)
        self.max_y = max(self.max_y, max_y_term)

        # Initialize from all terminals
        # Priority queue: (cost, node, parent, direction, connected_terminals)
        # Use a bitmask for connected terminals to save space

        # Map each terminal to a bit
        terminal_to_bit = {term: 1 << i for i, term in enumerate(terminals)}
        all_terminals_mask = (1 << len(terminals)) - 1

        # Start from each terminal
        open_set = []
        # Track best cost for (node, mask)
        best_costs = {}
        # Track parent information
        parents = {}

        # Initialize step costs for the net being routed
        step_costs = {}
        final_step_costs = {}  # Track only the final path's costs

        # Initialize all terminals
        for i, term in enumerate(terminals):
            mask = terminal_to_bit[term]
            state = (term, mask)
            best_costs[state] = 0
            parents[state] = (None, None)  # (parent_node, parent_mask)
            heappush(open_set, (0, term, mask, None, None))  # cost, node, mask, parent_node, incoming_direction

        final_cost = 0  # Initialize final cost

        while open_set:
            # Handle pygame events if visualization is enabled
            if ENABLE_PYGAME:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            cost, current_node, current_mask, parent_node, incoming_direction = heappop(open_set)

            # Check if all terminals are connected
            if current_mask == all_terminals_mask:
                final_cost = cost  # Store the final cost
                if ENABLE_PYGAME:
                    # Calculate and display additional cost information before quitting
                    h_value = self._heuristic_mst(current_node, current_mask, terminal_set)
                    f_score = cost + h_value
                    # Display cost information
                    cost_text = f"Final cost: {cost:.2f}, h: {h_value:.2f}, f: {f_score:.2f}"
                    cost_surface = font.render(cost_text, True, (0, 0, 0))
                    screen.blit(cost_surface, (10, 100))
                    f_score_text = f"f_score: {f_score:.2f}"
                    f_score_surface = font.render(f_score_text, True, (0, 0, 0))
                    screen.blit(f_score_surface, (10, 130))
                    pygame.display.flip()
                    # pygame.image.save(pygame.display.get_surface(), "data/images/saroute/screenshot.png")
                    pygame.image.save(screen, f"data/images/sastar/{net.name}_cost_explored.png")
                    # Wait a moment to see the final state
                    pygame.quit()
                # Verify that all terminals are actually reachable in the parent structure
                tree = self._build_steiner_tree(parents, (current_node, current_mask), terminals, existing_paths)
                # Additional check: ensure all terminals are in the tree
                tree_geom = unary_union(tree) if tree else None
                if tree_geom:
                    for terminal in terminals:
                        if tree_geom.distance(Point(terminal)) > 1e-6:
                            log.warning(f"Terminal {terminal} is not in the built tree despite mask being complete")
                            # Add direct connection
                            nearest_point = tree_geom.interpolate(tree_geom.project(Point(terminal)))
                            tree.append(LineString([terminal, (nearest_point.x, nearest_point.y)]))
                mls = MultiLineString(tree_geom)
                return mls, final_cost

            # Explore neighbors
            for neighbor in self._get_neighbors(current_node):
                # Calculate move cost using the cost estimator
                move_cost = self.cost_estimator.get_move_cost(current_node, neighbor, parent_node, context)

                # Store cost at segment midpoint
                midpoint = ((current_node[0] + neighbor[0]) / 2, (current_node[1] + neighbor[1]) / 2)
                step_costs[midpoint] = move_cost

                new_cost = cost + move_cost

                # Update mask if neighbor is a terminal
                new_mask = current_mask
                if neighbor in terminal_set:
                    new_mask |= terminal_to_bit[neighbor]

                new_state = (neighbor, new_mask)

                if new_state not in best_costs or new_cost < best_costs[new_state]:
                    best_costs[new_state] = new_cost
                    parents[new_state] = (current_node, current_mask)
                    heappush(open_set, (new_cost, neighbor, new_mask, current_node, None))

                    # Visualization if enabled

                    if ENABLE_PYGAME:
                        # Avoid zero or negative costs by shifting
                        shift = 1e-6
                        log_costs = [math.log(c + shift) for c in step_costs.values()]

                        log_min = min(log_costs)
                        log_max = max(log_costs)

                        for (mx, my), cost in step_costs.items():
                            log_cost = math.log(cost + shift)
                            norm_cost = (log_cost - log_min) / (log_max - log_min + 1e-12)

                            color_value = int(255 * norm_cost)
                            color = (color_value, 255 - color_value, 128)  # same gradient

                            rect_size = int(1 * self.scale)
                            screen_rect = pygame.Rect(0, 0, rect_size, rect_size)
                            screen_rect.center = self.to_screen_coords((mx, my))

                            surf = pygame.Surface((rect_size, rect_size), pygame.SRCALPHA)
                            surf.fill((*color, 100))
                            screen.blit(surf, screen_rect.topleft)

                        """
                        # Draw current path tree by reconstructing from best_costs
                        for state, parent_info in parents.items():
                            node, mask = state
                            parent_node, parent_mask = parent_info
                            if parent_node is not None:
                                start_pos = self.to_screen_coords(parent_node)
                                end_pos = self.to_screen_coords(node)
                                pygame.draw.line(screen, (0, 0, 255), start_pos, end_pos, 2)
                        """

                        # Draw current node
                        current_pos = self.to_screen_coords(current_node)
                        pygame.draw.circle(screen, (10, 10, 10), current_pos, 4)

                        # Draw reached terminals info
                        reached_count = bin(current_mask).count("1")
                        info_text = f"Reached: {reached_count}/{len(terminals)}"
                        text_surface = font.render(info_text, True, (0, 0, 0))
                        screen.blit(text_surface, (10, 10))

                        # Display cost information
                        cost_text = f"Current cost: {cost:.0f}"
                        cost_surface = font.render(cost_text, True, (0, 0, 0))
                        screen.blit(cost_surface, (10, 40))

                        # Display g_score
                        g_score_text = f"g_score: {best_costs.get((current_node, current_mask), 0):.0f}"
                        g_score_surface = font.render(g_score_text, True, (0, 0, 0))
                        screen.blit(g_score_surface, (10, 70))

                        pygame.display.flip()

        if ENABLE_PYGAME:
            pygame.quit()
        return [], 0, {}  # Failed to route, return empty list, zero cost, and empty step costs

    def _extract_final_path_costs(self, tree, context, terminals):
        """Extract step costs only for the final chosen path"""
        final_step_costs = {}

        # Skip if tree is empty
        if not tree:
            return final_step_costs

        # Extract all coordinates from the tree
        all_coords = []
        for path in tree:
            if isinstance(path, LineString):
                all_coords.extend(list(path.coords))
            elif isinstance(path, MultiLineString):
                for line in path.geoms:
                    all_coords.extend(list(line.coords))

        # Remove duplicates while preserving order
        unique_coords = []
        seen = set()
        for coord in all_coords:
            # Round to avoid floating point issues
            rounded = (round(coord[0], 6), round(coord[1], 6))
            if rounded not in seen:
                seen.add(rounded)
                unique_coords.append(coord)

        # Calculate costs for each segment in the final path
        parent_node = None
        for i in range(len(unique_coords) - 1):
            current_node = unique_coords[i]
            neighbor_node = unique_coords[i + 1]

            # Skip zero-length segments
            if current_node == neighbor_node:
                continue

            # Calculate move cost
            move_cost = self.cost_estimator.get_move_cost(current_node, neighbor_node, parent_node, context)

            # Store cost at segment midpoint
            midpoint = ((current_node[0] + neighbor_node[0]) / 2, (current_node[1] + neighbor_node[1]) / 2)
            final_step_costs[midpoint] = move_cost

            # Update parent for next iteration
            parent_node = current_node

        return final_step_costs

    def _heuristic_mst(self, current_node, reached_mask, all_terminals):
        """
        Admissible heuristic: MST cost to connect remaining terminals
        """
        # Convert bitmask to set of reached terminals
        reached_terminals = set()
        for i, terminal in enumerate(all_terminals):
            if reached_mask & (1 << i):
                reached_terminals.add(terminal)

        unreached = all_terminals - reached_terminals

        if not unreached:
            return 0

        # Simple heuristic: sum of distances to nearest reached terminal
        # (More accurate: compute actual MST)
        total = 0

        for terminal in unreached:
            min_dist = float("inf")

            # Distance to current node
            min_dist = min(min_dist, self._distance(terminal, current_node))

            # Distance to any reached terminal
            for reached in reached_terminals:
                min_dist = min(min_dist, self._distance(terminal, reached))

            total += min_dist

        return total

    def _build_steiner_tree(self, parents, goal_state, terminals, existing_paths=None):
        """Build the Steiner tree from the goal state using parent pointers"""
        all_nodes = set()

        def trace_state_chain(start_state):
            """Follow parent pointers up to a root, collecting real nodes only."""
            current_state = start_state
            visited = set()
            while current_state in parents:
                if current_state in visited:
                    log.warning(f"Cycle detected in parents: {current_state}")
                    break
                visited.add(current_state)

                node, mask = current_state
                if node is not None:
                    all_nodes.add(node)

                parent_info = parents.get(current_state)
                if not parent_info or parent_info == (None, None):
                    break

                parent_node, parent_mask = parent_info

                # Mask must strictly lose terminals → prevents infinite cycles
                if parent_mask & mask == mask and parent_mask != mask:
                    log.error(f"Invalid parent mask transition: {current_state} → {(parent_node, parent_mask)}")
                    break

                current_state = (parent_node, parent_mask)

        # Collect all nodes reachable from terminals
        for terminal in terminals:
            for state, (pnode, _) in parents.items():
                node, _ = state
                if node == terminal:
                    trace_state_chain(state)
                    break

        # Collect tree nodes from goal state also
        trace_state_chain(goal_state)

        # Build valid edges from collected nodes
        edges = set()
        for state, parent_info in parents.items():
            node, _ = state
            if parent_info and parent_info != (None, None):
                parent_node, _ = parent_info
                if node in all_nodes and parent_node in all_nodes:
                    edges.add(tuple(sorted([node, parent_node])))

        # Convert edges to LineStrings
        line_segments = [LineString([a, b]) for (a, b) in edges]

        # Merge segments collinearly and avoid double counting existing paths
        merged_paths = self._merge_adjacent_segments(line_segments, terminals, existing_paths)

        # Fix any dropped terminal connections
        if merged_paths:
            tree_geom = unary_union(merged_paths)
            for terminal in terminals:
                tpt = Point(terminal)
                if tree_geom.distance(tpt) > 1e-6:
                    log.warning(f"Forcing terminal {terminal} into tree")
                    nearest = tree_geom.interpolate(tree_geom.project(tpt))
                    merged_paths.append(LineString([terminal, (nearest.x, nearest.y)]))

        return merged_paths

    def old_build_steiner_tree(self, parents, goal_state, terminals, existing_paths=None):
        """Build the Steiner tree from the goal state using parent pointers"""
        # Collect all nodes that are part of the tree
        all_nodes = set()
        # Start from all terminals and traverse up to collect all nodes
        for terminal in terminals:
            # Find states that include this terminal
            for state in parents:
                node, mask = state
                if node == terminal:
                    # Traverse up the parent hierarchy
                    current_state = state
                    while current_state in parents:
                        current_node, current_mask = current_state
                        all_nodes.add(current_node)
                        parent_info = parents.get(current_state)
                        if parent_info is None or parent_info == (None, None):
                            break
                        parent_node, parent_mask = parent_info
                        current_state = (parent_node, parent_mask)
                    break

        # Also include nodes from the goal state
        current_state = goal_state
        while current_state in parents:
            current_node, current_mask = current_state
            all_nodes.add(current_node)
            parent_info = parents.get(current_state)
            if parent_info is None or parent_info == (None, None):
                break
            parent_node, parent_mask = parent_info
            current_state = (parent_node, parent_mask)

        # Now build edges between parent-child relationships
        edges = set()
        for state, parent_info in parents.items():
            node, mask = state
            if parent_info is None or parent_info == (None, None):
                continue
            parent_node, parent_mask = parent_info
            if parent_node is not None:
                # Ensure both nodes are in our collected nodes
                if node in all_nodes and parent_node in all_nodes:
                    edge = tuple(sorted([node, parent_node]))
                    edges.add(edge)

        # Convert edges to LineStrings
        paths = []
        for edge in edges:
            paths.append(LineString([edge[0], edge[1]]))

        # Merge adjacent collinear segments to reduce fragmentation
        # Pass existing_paths to avoid creating overlapping segments
        merged_paths = self._merge_adjacent_segments(paths, terminals, existing_paths)

        # Verify that all terminals are connected in the tree
        if merged_paths:
            tree_geom = unary_union(merged_paths)
            for terminal in terminals:
                if tree_geom.distance(Point(terminal)) > 1e-6:
                    log.warning(f"Terminal {terminal} is not connected to the tree")
                    # Add a direct connection to the nearest point in the tree
                    nearest_point = tree_geom.interpolate(tree_geom.project(Point(terminal)))
                    connection = LineString([terminal, (nearest_point.x, nearest_point.y)])
                    merged_paths.append(connection)

        # Ensure we always return a list of LineStrings (can be converted to MultiLineString later)
        return merged_paths

    def _merge_adjacent_segments(self, paths, terminals=None, existing_paths=None):
        """Merge adjacent segments into clean orthogonal lines while avoiding overlaps"""
        if not paths:
            return []

        # First merge all connected segments
        merged = linemerge(paths)

        if merged.is_empty:
            return []

        # Convert to list of LineStrings
        if isinstance(merged, LineString):
            merged_paths = [merged]
        else:  # MultiLineString
            merged_paths = list(merged.geoms)

        # Split merged paths at direction changes to get clean orthogonal segments
        result_paths = []
        for line in merged_paths:
            segments = self._split_at_direction_changes(line)

            # Process each segment
            for segment in segments:
                if existing_paths:
                    # Check for overlaps with existing nets
                    has_significant_overlap = False
                    for net_path in existing_paths:
                        intersection = segment.intersection(net_path)
                        if not intersection.is_empty:
                            overlap_length = 0
                            if isinstance(intersection, LineString):
                                overlap_length = round(intersection.length)
                            elif isinstance(intersection, MultiLineString):
                                overlap_length = round(sum(seg.length for seg in intersection.geoms))

                            if overlap_length >= 1:
                                # This segment has a significant overlap
                                log.warning(f"Merged segment has overlap with existing net: {overlap_length=}")
                                has_significant_overlap = True
                                break

                    # Always add the segment, even if it has overlaps
                    # This ensures connectivity, and we'll handle overlaps in post-processing
                    result_paths.append(segment)

                    # If there's a significant overlap, log it for debugging
                    if has_significant_overlap:
                        log.warning(f"Adding segment despite overlap: {segment.wkt}")
                else:
                    result_paths.append(segment)

        # Ensure all terminals are connected
        if terminals:
            terminal_points = set(terminals)
            merged_geom = unary_union(result_paths) if result_paths else None

            if merged_geom:
                for terminal in terminal_points:
                    if merged_geom.distance(Point(terminal)) > 1e-6:
                        # Add connection to nearest point
                        nearest = merged_geom.interpolate(merged_geom.project(Point(terminal)))
                        result_paths.append(LineString([terminal, (nearest.x, nearest.y)]))

        return result_paths

    def _try_reroute_segment(self, segment, existing_paths, obstacles):
        """Try to reroute a segment to avoid overlaps with existing nets"""
        # Get the endpoints of the segment
        start = segment.coords[0]
        end = segment.coords[-1]

        # Create a temporary router with a smaller grid spacing
        router = SimultaneousRouter(grid_spacing=1.0, obstacle_geoms=obstacles, halo_size=2)

        # Try to route between the endpoints while avoiding existing nets
        paths, _ = router.route_net([start, end], existing_paths)

        if paths:
            # Successfully rerouted
            return paths
        else:
            # Failed to reroute, return the original segment
            return [segment]

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
        gx = round(x / self.grid_spacing) * self.grid_spacing
        gy = round(y / self.grid_spacing) * self.grid_spacing
        return (gx, gy)

    def _get_neighbors(self, node):
        x, y = node
        g = self.grid_spacing
        # Only orthogonal moves
        neighbors = []
        potential_neighbors = [(x + g, y), (x - g, y), (x, y + g), (x, y - g)]
        # Filter neighbors to be within padded bounds
        for nx, ny in potential_neighbors:
            if self.min_x <= nx <= self.max_x and self.min_y <= ny <= self.max_y:
                neighbors.append((nx, ny))
        return neighbors

    @staticmethod
    def _distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def draw_route(self, screen, route_geom):
        """Draw an existing route as a single colored line from start to end."""
        color = next(COLORS)

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
