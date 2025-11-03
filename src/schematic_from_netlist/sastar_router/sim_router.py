import logging as log
import math
import os
import re
import sys
import time
import unicodedata
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import cycle

import numpy as np
import pygame
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box
from shapely.ops import linemerge, unary_union
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


class SimultaneousRouter:
    def __init__(
        self,
        grid_spacing=1.0,
        obstacle_geoms=None,
        halo_size=4,
        bounds=None,
        debug: bool | None = None,
        log_interval: int = 100,
    ):
        self.grid_spacing = grid_spacing
        self.obstacles = obstacle_geoms or []
        self.obstacles_index = STRtree(self.obstacles) if self.obstacles else None
        self.bounds = bounds  # Add bounds parameter
        self.scale = 1.0
        self.width, self.height = 800, 600

        # Debug/diagnostic controls
        self.debug = False
        self.log_interval = log_interval

        # Create halo regions around obstacles
        self.halo_geoms = []
        for obs in self.obstacles:
            halo = obs.buffer(halo_size)
            self.halo_geoms.append(halo)
        self.halo_index = STRtree(self.halo_geoms) if self.halo_geoms else None

        # Initialize cost estimator
        self.cost_estimator = CostEstimator(grid_spacing)

        # Calculate bounds if not provided
        # Ensure bounds include all terminals by expanding if necessary
        if not self.bounds:
            self._calculate_default_bounds()

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

    def _initialize_visualization(self, terminals, existing_paths, port_colors=None):
        """Initialize Pygame and draw the initial state."""
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("A* Routing Visualization")
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
        if existing_paths:
            for route_geom in existing_paths:
                self.draw_route(screen, route_geom)

        # Draw terminals with per-port colors if provided
        legend_y = 10
        for i, term in enumerate(terminals):
            pos = self.to_screen_coords(term)
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
        return screen, font

    def _handle_pygame_events(self):
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def _visualize_search_progress(
        self,
        screen,
        font,
        step_costs,
        parents,
        current_node,
        current_mask,
        terminals,
        cost,
        best_costs,
        per_source_costs,
        port_colors,
    ):
        """Visualize the progress of the A* search.

        Instead of a cost gradient, color the explored wavefront by originating terminal.
        """
        # Draw explored nodes per source with their assigned port color
        if not per_source_costs:
            return

        # Determine a reasonable pixel size for grid markers
        rect_size = max(2, int(0.8 * self.scale))

        # First pass: collect overlap counts to highlight merge regions
        overlap_counts = {}
        for src_id, nodes in per_source_costs.items():
            for node in nodes.keys():
                overlap_counts[node] = overlap_counts.get(node, 0) + 1

        # Draw per-source explored nodes
        for src_id, nodes in per_source_costs.items():
            color = (255, 0, 0)
            if port_colors and src_id in port_colors:
                color = port_colors[src_id]
            for mx, my in nodes.keys():
                screen_rect = pygame.Rect(0, 0, rect_size, rect_size)
                screen_rect.center = self.to_screen_coords((mx, my))
                # Semi-transparent fill for the wave from this source
                surf = pygame.Surface((rect_size, rect_size), pygame.SRCALPHA)
                surf.fill((*color, 110))
                screen.blit(surf, screen_rect.topleft)

        # Overlay merge regions (visited by multiple sources) in neutral gray
        merge_color = (150, 150, 150, 170)
        for (mx, my), cnt in overlap_counts.items():
            if cnt > 1:
                screen_rect = pygame.Rect(0, 0, rect_size, rect_size)
                screen_rect.center = self.to_screen_coords((mx, my))
                surf = pygame.Surface((rect_size, rect_size), pygame.SRCALPHA)
                surf.fill(merge_color)
                screen.blit(surf, screen_rect.topleft)

        # Draw current node marker
        current_pos = self.to_screen_coords(current_node)
        pygame.draw.circle(screen, (10, 10, 10), current_pos, 2)

        # Draw reached terminals info
        reached_count = bin(current_mask).count("1")
        info_text = f"Reached: {reached_count}/{len(terminals)}"
        text_surface = font.render(info_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))

        # Display cost information (g-score at current state)
        g = best_costs.get((current_node, current_mask), cost)
        cost_text = f"g: {g:.0f}"
        cost_surface = font.render(cost_text, True, (0, 0, 0))
        screen.blit(cost_surface, (10, 40))

        pygame.display.flip()

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

    def _visualize_final_path(self, screen, font, net, cost, current_node, current_mask, terminal_set):
        """Visualize the final path and save the image."""
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
        pygame.image.save(screen, f"data/images/sastar/{self.clean_hierarchical_name(net.name)}_cost_explored.png")
        # Wait a moment to see the final state
        pygame.quit()

    def route_net(self, net, existing_paths=None):
        """
        connect one net
        """
        terminals = [(pin.draw.geom.x, pin.draw.geom.y) for pin in net.pins.values() if pin.draw.geom]
        if len(terminals) < 2:
            return [], 0

        # Snap terminals to grid
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

        # Initialize pygame
        screen, font = None, None
        if ENABLE_PYGAME:
            # Assign distinct colors per terminal
            port_colors = {i: BASE_COLORS[i % len(BASE_COLORS)] for i in range(len(terminals))}
            screen, font = self._initialize_visualization(terminals, context.existing_paths, port_colors)

        # Debug: start summary
        if self.debug:
            log.debug(
                "[START routing] net=%s terminals=%d obstacles=%d existing_paths=%d bounds=%s",
                getattr(net, "name", None),
                len(terminals),
                len(self.obstacles),
                len(context.existing_paths),
                str(self.bounds.bounds if self.bounds else None),
            )

        # Initialize from all terminals
        # Priority queue: (cost, node, mask, parent_node, incoming_direction, origin_id)
        # Use a bitmask for connected terminals to save space
        # Map each terminal to a bit
        terminal_to_bit = {term: 1 << i for i, term in enumerate(terminals)}
        all_terminals_mask = (1 << len(terminals)) - 1

        open_set = []  # Start from each terminal
        best_costs = {}  # Track best cost for (node, mask)
        parents = {}  # Track parent information
        step_costs = {}  # Initialize step costs for the net being routed
        visited = set()  # Track visited (node, mask) states
        # Per-node best masks to enable wave meeting and merging at a node
        node_masks: dict[tuple[float, float], dict[int, float]] = {}
        # Per-source cumulative explored costs for visualization
        per_source_costs: dict[int, dict[tuple[float, float], float]] = {i: {} for i in range(len(terminals))}

        # Diagnostics counters and helpers
        cnt_pops = cnt_pushes = cnt_prune_cost = cnt_prune_visited = 0
        cnt_merge_attempt = cnt_merge_improve = cnt_early_stop = 0
        cnt_prune_child = 0
        t0 = time.time()
        pop_idx = 0

        def mask_info(mask: int) -> str:
            return f"mask=0b{mask:b}({mask.bit_count()} of {len(terminals)})"

        def maybe_log_summary():
            if not self.debug:
                return
            if self.log_interval and pop_idx % self.log_interval == 0 and pop_idx > 0:
                log.debug(
                    "[SUM] pops=%d pushes=%d prune_curr=%d prune_child=%d prune_visited=%d merges(attempt=%d, improve=%d) early_stop=%d open=%d visited=%d elapsed=%.2fs",
                    cnt_pops,
                    cnt_pushes,
                    cnt_prune_cost,
                    cnt_prune_child,
                    cnt_prune_visited,
                    cnt_merge_attempt,
                    cnt_merge_improve,
                    cnt_early_stop,
                    len(open_set),
                    len(visited),
                    time.time() - t0,
                )

        # Initialize all terminals
        for i, term in enumerate(terminals):
            mask = terminal_to_bit[term]
            state = (term, mask)
            best_costs[state] = 0.0
            parents[state] = (None, None)  # (parent_node, parent_mask)
            heappush(open_set, (0.0, term, mask, None, None, i))  # cost, node, mask, parent_node, incoming_direction, origin
            # Seed per-node mask table
            node_masks.setdefault(term, {})[mask] = 0.0
            per_source_costs[i][term] = 0.0

        final_cost = 0.0  # Initialize final cost

        while open_set:
            # Handle pygame events if visualization is enabled
            if ENABLE_PYGAME:
                self._handle_pygame_events()

            cost, current_node, current_mask, parent_node, incoming_direction, origin_id = heappop(open_set)
            cnt_pops += 1
            pop_idx += 1
            if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                log.debug(
                    "[POP %d] cost=%.0f node=%s %s origin=%s open=%d",
                    pop_idx,
                    cost,
                    current_node,
                    mask_info(current_mask),
                    str(origin_id),
                    len(open_set),
                )
            maybe_log_summary()

            # Update mask for current node if it's a terminal
            current_mask_updated = current_mask
            if current_node in terminal_set:
                prev_mask = current_mask_updated
                current_mask_updated |= terminal_to_bit[current_node]
                if self.debug and current_mask_updated != prev_mask:
                    log.debug(
                        "[MASK] Reached terminal at %s: %s -> %s",
                        current_node,
                        mask_info(prev_mask),
                        mask_info(current_mask_updated),
                    )

            current_state = (current_node, current_mask_updated)

            # Periodic heuristic diagnostics
            if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                try:
                    h = self._heuristic_mst(current_node, current_mask_updated, terminal_set)
                    g = cost
                    f = g + h
                    log.debug("[GHF] g=%.0f h=%.0f f=%.0f at node=%s %s", g, h, f, current_node, mask_info(current_mask_updated))
                except Exception as e:
                    log.debug("[GHF] heuristic error: %s", e)

            # Record per-source explored cost at the popped node
            if ENABLE_PYGAME and "per_source_costs" in locals() and origin_id is not None and origin_id >= 0:
                # Only keep the best known cost per node per source
                prev = per_source_costs[origin_id].get(current_node)
                if prev is None or cost < prev - 1e-9:
                    per_source_costs[origin_id][current_node] = cost

            # If we already have a better cost for this state, skip
            if current_state in best_costs and cost > best_costs[current_state] + 1e-9:
                cnt_prune_cost += 1
                if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                    log.debug(
                        "[SKIP worse] state=%s best=%.0f curr=%.0f delta=%.0f",
                        (current_node, current_mask_updated),
                        best_costs[current_state],
                        cost,
                        cost - best_costs[current_state],
                    )
                continue

            # Record best cost at this node/mask and attempt to merge with other waves meeting here
            node_entry = node_masks.setdefault(current_node, {})
            prev_best_here = node_entry.get(current_mask_updated)
            if prev_best_here is None or cost + 1e-9 < prev_best_here:
                node_entry[current_mask_updated] = cost

                # Try to merge with other disjoint masks at the same node (waves meet)
                for other_mask, other_cost in list(node_entry.items()):
                    if other_mask == current_mask_updated:
                        continue
                    if (other_mask & current_mask_updated) == 0:
                        cnt_merge_attempt += 1
                        merged_mask = other_mask | current_mask_updated
                        merged_cost = other_cost + cost
                        merged_state = (current_node, merged_mask)
                        improved = merged_state not in best_costs or merged_cost < best_costs[merged_state] - 1e-9
                        if improved:
                            cnt_merge_improve += 1
                            best_costs[merged_state] = merged_cost
                            # Parent points to the newer component; the other side is already chained via its own parents
                            parents[merged_state] = (current_node, current_mask_updated)
                            heappush(open_set, (merged_cost, current_node, merged_mask, current_node, None, -1))
                            cnt_pushes += 1
                            if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                                log.debug(
                                    "[MERGE curr] node=%s %s + %s -> %s merged_cost=%.0f open=%d",
                                    current_node,
                                    mask_info(current_mask_updated),
                                    mask_info(other_mask),
                                    mask_info(merged_mask),
                                    merged_cost,
                                    len(open_set),
                                )
                        else:
                            if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                                log.debug(
                                    "[MERGE curr skip] node=%s %s + %s -> %s no improvement (best=%.0f <= new=%.0f)",
                                    current_node,
                                    mask_info(current_mask_updated),
                                    mask_info(other_mask),
                                    mask_info(merged_mask),
                                    best_costs.get(merged_state, float("inf")),
                                    merged_cost,
                                )

                        # Early stop if all terminals are connected when waves meet
                        if merged_mask == all_terminals_mask:
                            cnt_early_stop += 1
                            if self.debug:
                                log.debug(
                                    "[EARLY STOP via merge] node=%s %s cost=%.0f", current_node, mask_info(merged_mask), merged_cost
                                )
                            final_cost = merged_cost
                            if ENABLE_PYGAME:
                                self._visualize_final_path(screen, font, net, merged_cost, current_node, merged_mask, terminal_set)
                            tree = self._build_steiner_tree(parents, (current_node, merged_mask), terminals, existing_paths)
                            tree_geom = unary_union(tree) if tree else None
                            if isinstance(tree_geom, LineString):
                                mls = MultiLineString([tree_geom])
                            elif isinstance(tree_geom, MultiLineString):
                                mls = tree_geom
                            elif tree_geom is None:
                                mls = MultiLineString()
                            else:
                                mls = MultiLineString([g for g in getattr(tree_geom, "geoms", []) if isinstance(g, LineString)])
                            if ENABLE_PYGAME:
                                pygame.quit()
                            return mls, final_cost

            # Mark visited after potential merging
            if current_state in visited:
                cnt_prune_visited += 1
                if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                    log.debug("[SKIP visited] state=%s", (current_node, current_mask_updated))
                continue
            visited.add(current_state)
            if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                log.debug("[VISIT] state=%s visited=%d", (current_node, current_mask_updated), len(visited))

            # Check if we've connected all terminals via this state specifically
            if current_mask_updated == all_terminals_mask:
                cnt_early_stop += 1
                if self.debug:
                    log.debug("[EARLY STOP via state] node=%s %s cost=%.0f", current_node, mask_info(current_mask_updated), cost)
                final_cost = cost
                if ENABLE_PYGAME:
                    self._visualize_final_path(screen, font, net, cost, current_node, current_mask_updated, terminal_set)
                tree = self._build_steiner_tree(parents, (current_node, current_mask_updated), terminals, existing_paths)
                tree_geom = unary_union(tree) if tree else None
                if isinstance(tree_geom, LineString):
                    mls = MultiLineString([tree_geom])
                elif isinstance(tree_geom, MultiLineString):
                    mls = tree_geom
                elif tree_geom is None:
                    mls = MultiLineString()
                else:
                    mls = MultiLineString([g for g in getattr(tree_geom, "geoms", []) if isinstance(g, LineString)])
                if ENABLE_PYGAME:
                    pygame.quit()
                return mls, final_cost

            # Explore neighbors - use the updated mask
            for neighbor in self._get_neighbors(current_node):
                move_cost = self.cost_estimator.get_move_cost(parent_node, current_node, neighbor, context)
                step_costs[neighbor] = move_cost
                new_cost = cost + move_cost

                # Start with the updated current mask, then add neighbor's terminal bit if applicable
                new_mask = current_mask_updated
                if neighbor in terminal_set:
                    new_mask |= terminal_to_bit[neighbor]

                new_state = (neighbor, new_mask)

                if new_state not in best_costs or new_cost < best_costs[new_state] - 1e-9:
                    best_costs[new_state] = new_cost
                    parents[new_state] = (current_node, current_mask_updated)  # Store updated mask
                    heappush(open_set, (new_cost, neighbor, new_mask, current_node, None, origin_id))
                    cnt_pushes += 1
                    if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                        log.debug(
                            "[PUSH] to=%s move=%.0f new_cost=%.0f %s origin=%s open=%d",
                            neighbor,
                            move_cost,
                            new_cost,
                            mask_info(new_mask),
                            str(origin_id),
                            len(open_set),
                        )
                    # Track per-source explored cost for neighbor
                    if ENABLE_PYGAME and origin_id is not None and origin_id >= 0:
                        prev = per_source_costs[origin_id].get(neighbor)
                        if prev is None or new_cost < prev - 1e-9:
                            per_source_costs[origin_id][neighbor] = new_cost

                    # Record into node_masks for potential merging at neighbor and try immediate merges
                    n_entry = node_masks.setdefault(neighbor, {})
                    prev_best_n = n_entry.get(new_mask)
                    if prev_best_n is None or new_cost + 1e-9 < prev_best_n:
                        n_entry[new_mask] = new_cost
                        for other_mask, other_cost in list(n_entry.items()):
                            if other_mask == new_mask:
                                continue
                            if (other_mask & new_mask) == 0:
                                cnt_merge_attempt += 1
                                merged_mask = other_mask | new_mask
                                merged_cost = other_cost + new_cost
                                merged_state = (neighbor, merged_mask)
                                improved = merged_state not in best_costs or merged_cost < best_costs[merged_state] - 1e-9
                                if improved:
                                    cnt_merge_improve += 1
                                    best_costs[merged_state] = merged_cost
                                    parents[merged_state] = (neighbor, new_mask)
                                    heappush(open_set, (merged_cost, neighbor, merged_mask, neighbor, None, -1))
                                    cnt_pushes += 1
                                    if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                                        log.debug(
                                            "[MERGE nei] node=%s %s + %s -> %s merged_cost=%.0f open=%d",
                                            neighbor,
                                            mask_info(new_mask),
                                            mask_info(other_mask),
                                            mask_info(merged_mask),
                                            merged_cost,
                                            len(open_set),
                                        )
                                else:
                                    if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                                        log.debug(
                                            "[MERGE nei skip] node=%s %s + %s -> %s no improvement (best=%.0f <= new=%.0f)",
                                            neighbor,
                                            mask_info(new_mask),
                                            mask_info(other_mask),
                                            mask_info(merged_mask),
                                            best_costs.get(merged_state, float("inf")),
                                            merged_cost,
                                        )
                                if merged_mask == all_terminals_mask:
                                    cnt_early_stop += 1
                                    if self.debug:
                                        log.debug(
                                            "[EARLY STOP via nei-merge] node=%s %s cost=%.0f",
                                            neighbor,
                                            mask_info(merged_mask),
                                            merged_cost,
                                        )
                                    final_cost = merged_cost
                                    if ENABLE_PYGAME:
                                        self._visualize_final_path(
                                            screen, font, net, merged_cost, neighbor, merged_mask, terminal_set
                                        )
                                    tree = self._build_steiner_tree(parents, (neighbor, merged_mask), terminals, existing_paths)
                                    tree_geom = unary_union(tree) if tree else None
                                    if isinstance(tree_geom, LineString):
                                        mls = MultiLineString([tree_geom])
                                    elif isinstance(tree_geom, MultiLineString):
                                        mls = tree_geom
                                    elif tree_geom is None:
                                        mls = MultiLineString()
                                    else:
                                        mls = MultiLineString(
                                            [g for g in getattr(tree_geom, "geoms", []) if isinstance(g, LineString)]
                                        )
                                    if ENABLE_PYGAME:
                                        pygame.quit()
                                    return mls, final_cost
                else:
                    cnt_prune_child += 1
                    if self.debug and (self.log_interval and pop_idx % self.log_interval == 0):
                        log.debug(
                            "[PRUNE child worse] to=%s new_cost=%.0f best=%.0f %s",
                            neighbor,
                            new_cost,
                            best_costs[new_state],
                            mask_info(new_mask),
                        )

                    # Visualization if enabled
                    if ENABLE_PYGAME:
                        self._visualize_search_progress(
                            screen,
                            font,
                            step_costs,
                            parents,
                            current_node,
                            current_mask_updated,
                            terminals,
                            cost,
                            best_costs,
                            per_source_costs,
                            port_colors,
                        )

        if ENABLE_PYGAME:
            pygame.quit()

        # Failed to route, return empty list
        log.warning(f"Failed to route net {net.name} - could not connect all terminals")
        return [], float("inf")

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

            # Store cost at neighbor_node
            final_step_costs[neighbor_node] = move_cost

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
        log.info(f"{goal_state=}")
        log.info(f"{parents=}")
        log.info(f"{terminals=}")

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
        log.info(f"{edges=}")

        # Convert edges to LineStrings
        line_segments = [LineString([a, b]) for (a, b) in edges]
        log.info(f"{line_segments=}")

        # Merge segments collinearly and avoid double counting existing paths
        merged_paths = self._merge_adjacent_segments(line_segments, terminals, existing_paths)
        log.info(f"{merged_paths=}")

        # Fix any dropped terminal connections
        if merged_paths:
            tree_geom = unary_union(merged_paths)
            for terminal in terminals:
                tpt = Point(terminal)
                if tree_geom.distance(tpt) > 1e-6:
                    log.warning(f"mising terminal {terminal} in tree")
                    nearest = tree_geom.interpolate(tree_geom.project(tpt))
                    # merged_paths.append(LineString([terminal, (nearest.x, nearest.y)]))

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
        log.info(f"{paths} -> {merged_paths=}")
        for line in merged_paths:
            segments = self._split_at_direction_changes(line)

            # Process each segment
            for segment in segments:
                result_paths.append(segment)
                if existing_paths:
                    # Check for overlaps with existing nets
                    has_significant_overlap = False
                    for net_path in existing_paths:
                        intersection = segment.intersection(net_path)
                        # if not intersection.is_empty:
                        if intersection:
                            overlap_length = 0
                            if isinstance(intersection, LineString):
                                overlap_length = round(intersection.length)
                            elif isinstance(intersection, MultiLineString):
                                overlap_length = round(sum(seg.length for seg in intersection.geoms))

                            if overlap_length >= 1:
                                # This segment has a significant overlap
                                log.warning(
                                    f"Merged segment {segment.wkt} has {overlap_length} overlap with existing path {net_path}"
                                )
                                has_significant_overlap = True
                                break

                    # Always add the segment, even if it has overlaps
                    # This ensures connectivity, and we'll handle overlaps in post-processing

                    # If there's a significant overlap, log it for debugging
                    if has_significant_overlap:
                        log.warning(f"Adding segment despite overlap: {segment.wkt}")

        # Ensure all terminals are connected
        if terminals:
            terminal_points = set(terminals)
            merged_geom = unary_union(result_paths) if result_paths else None

            if merged_geom:
                for terminal in terminal_points:
                    if merged_geom.distance(Point(terminal)) > 1e-6:
                        log.warning(f"Terminal {terminal} is not connected to {merged_geom=}")
                        # Add connection to nearest point
                        # nearest = merged_geom.interpolate(merged_geom.project(Point(terminal)))
                        # result_paths.append(LineString([terminal, (nearest.x, nearest.y)]))

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


if __name__ == "__main__":
    log.info("Running Simultaneous Router")
    sr = SimultaneousRouter()
    log.info(sr.clean_hierarchical_name("/tnoi/is/a/test"))
