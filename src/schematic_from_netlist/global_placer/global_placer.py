import logging as log
import os
import sys
import time
from collections import deque, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass, field
from heapq import heappop, heappush

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree

from schematic_from_netlist.database.netlist_structures import Instance, Module, Net
from schematic_from_netlist.interfaces.symbol_library import SymbolLibrary

try:
    import pygame
except ImportError:
    log.warning("Pygame not found. Visualization will be disabled.")
    pygame = None

log.basicConfig(level=log.INFO)


@dataclass
class PlacerConfig:
    max_iterations: int = 50000
    step_size: float = 0.01
    # Base wirelength weight
    lambda_wire: float = 1.0
    # Penalties and weights
    lambda_overlap: float = 10.0
    lambda_region: float = 5.0  # kept for backward compatibility; prefer lambda_bound
    lambda_bound: float = 10.0  # recommended ~10x lambda_wire (soft wall)
    lambda_shape: float = 1.0
    lambda_rigid: float = 5.0
    lambda_anchor: float = 0.01  # recommended 1e-4 to 1e-2 x lambda_wire to prevent drift
    # Optional stabilizer: weak pull to region centroid for unconnected/loose cells
    w_dummy: float = 0.05  # typical ~0.05 × avg_net_weight
    # Optimization controls
    use_line_search: bool = True
    armijo_c: float = 1e-4
    backtrack_beta: float = 0.5
    max_backtracks: int = 20
    max_grad_norm: float = 100.0
    convergence_tol: float = 1e-5
    grid_size: int = 1
    visualize: bool = False
    visualization_interval: int = 10
    cost_logging_interval: int = 100
    macro_padding: float = 4.0


@dataclass
class PlacementRegion:
    name: str
    instances: list[int]
    geom: Polygon = field(default_factory=Polygon)


@dataclass
class PlacementState:
    """Encapsulates all placement state for optimization."""

    instances: list[Instance]
    instance_map: dict[str, int]
    nets: list[Net]
    pos: np.ndarray
    sizes: np.ndarray
    is_soft: np.ndarray
    areas: np.ndarray
    aspect_ratios: np.ndarray
    r_min: np.ndarray
    r_max: np.ndarray
    r_pref: np.ndarray
    regions: list[PlacementRegion]
    instance_to_region: np.ndarray
    parent_indices: np.ndarray
    template_groups: dict
    # Anchoring: store initial positions to prevent drift
    anchor_pos: np.ndarray | None = None

    @property
    def num_instances(self) -> int:
        return len(self.instances)


class Visualizer:
    def __init__(self, width=800, height=600, enabled=True, padding=50):
        self.enabled = enabled and pygame is not None
        if not self.enabled:
            return

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Macro Placer Visualization")
        self.font = pygame.font.SysFont(None, 24)
        self.padding = padding
        self.colors = {
            "background": (255, 255, 255),
            "hard_macro": (100, 100, 200),
            "soft_macro": (100, 200, 100),
            "net": (200, 100, 100),
            "region": (200, 200, 100, 100),
        }

    def _transform(self, x, y, scale, offset_x, offset_y, design_min_x, design_min_y):
        return (x - design_min_x) * scale + offset_x, (y - design_min_y) * scale + offset_y

    def draw(self, instances: list[Instance], nets: list[Net], regions: list[PlacementRegion] = None):
        if not self.enabled:
            return

        self.screen.fill(self.colors["background"])

        # Calculate bounding box of all objects
        all_geoms = [inst.draw.geom for inst in instances if inst.draw.geom]
        if regions:
            all_geoms.extend([r.geom for r in regions if r.geom])

        if not all_geoms:
            pygame.display.flip()
            return

        overall_bounds = unary_union(all_geoms).bounds
        design_min_x, design_min_y, design_max_x, design_max_y = overall_bounds
        design_width = design_max_x - design_min_x
        design_height = design_max_y - design_min_y

        if design_width == 0 or design_height == 0:
            return

        # Calculate scale to fit design with padding
        screen_width, screen_height = self.screen.get_size()
        scale_x = (screen_width - 2 * self.padding) / design_width
        scale_y = (screen_height - 2 * self.padding) / design_height
        scale = min(scale_x, scale_y)

        # Calculate offset to center the design
        scaled_width = design_width * scale
        scaled_height = design_height * scale
        offset_x = (screen_width - scaled_width) / 2
        offset_y = (screen_height - scaled_height) / 2

        # Draw regions
        if regions:
            for region in regions:
                if region.geom:
                    x_min, y_min, x_max, y_max = region.geom.bounds
                    sx_min, sy_min = self._transform(x_min, y_min, scale, offset_x, offset_y, design_min_x, design_min_y)
                    sx_max, sy_max = self._transform(x_max, y_max, scale, offset_x, offset_y, design_min_x, design_min_y)
                    s = pygame.Surface((sx_max - sx_min, sy_max - sy_min), pygame.SRCALPHA)
                    s.fill(self.colors["region"])
                    self.screen.blit(s, (sx_min, sy_min))

        # Draw instances
        for inst in instances:
            if inst.draw.geom:
                x_min, y_min, x_max, y_max = inst.draw.geom.bounds
                sx_min, sy_min = self._transform(x_min, y_min, scale, offset_x, offset_y, design_min_x, design_min_y)
                sx_max, sy_max = self._transform(x_max, y_max, scale, offset_x, offset_y, design_min_x, design_min_y)

                rect = pygame.Rect(sx_min, sy_min, sx_max - sx_min, sy_max - sy_min)
                color = self.colors["hard_macro"] if inst.draw.fixed_size else self.colors["soft_macro"]
                pygame.draw.rect(self.screen, color, rect, 2)
                text = self.font.render(inst.name, True, (0, 0, 0))
                self.screen.blit(text, (sx_min, sy_min))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()

    def quit(self):
        if self.enabled:
            pygame.quit()
            self.enabled = False


class GlobalPlacer:
    def __init__(self, db, config: PlacerConfig = PlacerConfig()):
        self.db = db
        self.config = config
        self.symbol_outlines = SymbolLibrary().get_symbol_outlines()
        self.visualizer = Visualizer(enabled=self.config.visualize)

    def place_design(self):
        """Main entry point for placement optimization."""
        log.info("Placing design")

        # Initialize placement state
        state = self._initialize_placement_state()

        # Log initial cost
        cost = self._calculate_total_cost_from_state(state)
        self._log_cost(cost, iteration=0, prefix="Initial")

        # Run optimization
        self._optimize_placement(state)

        # Legalize final placement
        self._legalize(state.instances, state.pos, state.sizes, state.nets, state.regions)

    def _initialize_placement_state(self) -> PlacementState:
        """Initialize all data structures for placement."""
        components = set(self.symbol_outlines.keys())
        module = self.db.design.flat_module

        instances = list(module.instances.values())
        instance_map = {inst.full_name: i for i, inst in enumerate(instances)}
        nets = list(module.nets.values())

        # Initialize numpy arrays
        num_instances = len(instances)
        state_dict = {
            "instances": instances,
            "instance_map": instance_map,
            "nets": nets,
            "pos": np.zeros((num_instances, 2)),
            "sizes": np.zeros((num_instances, 2)),
            "is_soft": np.zeros(num_instances, dtype=bool),
            "areas": np.zeros(num_instances),
            "aspect_ratios": np.ones(num_instances),
            "r_min": np.full(num_instances, 0.2),
            "r_max": np.full(num_instances, 5.0),
            "r_pref": np.ones(num_instances),
            "instance_to_region": np.full(num_instances, -1, dtype=int),
        }

        # Populate instance data
        self._populate_instance_data(instances, components, state_dict)

        # Create regions
        state_dict["regions"] = self._create_regions(instances, state_dict)

        # Anchor positions to initial placement to prevent drift
        state_dict["anchor_pos"] = np.copy(state_dict["pos"])  # used by anchor penalty

        # Pre-compute rigidity data
        parent_indices, template_groups = self._compute_rigidity_data(instances, instance_map, num_instances)
        state_dict["parent_indices"] = parent_indices
        state_dict["template_groups"] = template_groups

        return PlacementState(**state_dict)

    def _populate_instance_data(self, instances, components, state_dict):
        """Extract geometry and properties from instances."""
        for i, inst in enumerate(instances):
            if inst.draw.geom:
                x_min, y_min, x_max, y_max = inst.draw.geom.bounds
                state_dict["pos"][i] = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                width = x_max - x_min
                height = y_max - y_min
                state_dict["sizes"][i] = [width, height]
                state_dict["areas"][i] = width * height
                if height > 0:
                    state_dict["aspect_ratios"][i] = width / height

            # Determine if soft or hard macro
            inst.draw.fixed_size = inst.module.name in components
            state_dict["is_soft"][i] = not inst.draw.fixed_size

    def _create_regions(self, instances, state_dict) -> list[PlacementRegion]:
        """Create placement regions based on hierarchical modules."""
        regions = []
        hier_modules = {}

        for i, inst in enumerate(instances):
            if inst.hier_module.name not in hier_modules:
                hier_modules[inst.hier_module.name] = len(regions)
                regions.append(PlacementRegion(name=inst.hier_module.name, instances=[i]))
            else:
                region_idx = hier_modules[inst.hier_module.name]
                regions[region_idx].instances.append(i)
            state_dict["instance_to_region"][i] = hier_modules[inst.hier_module.name]

        # Calculate initial region geometries
        for region in regions:
            region_pos = state_dict["pos"][region.instances]
            region_sizes = state_dict["sizes"][region.instances]
            min_coords = np.min(region_pos - region_sizes / 2, axis=0)
            max_coords = np.max(region_pos + region_sizes / 2, axis=0)
            region.geom = box(min_coords[0], min_coords[1], max_coords[0], max_coords[1])

        return regions

    def _compute_rigidity_data(self, instances, instance_map, num_instances):
        """Pre-compute parent-child relationships for rigidity penalty."""
        parent_indices = np.full(num_instances, -1, dtype=int)
        template_groups = {}

        for i, inst in enumerate(instances):
            if inst.hier_prefix:
                parent_name = "/".join(inst.full_name.split("/")[:-1])
                if parent_name in instance_map:
                    parent_idx = instance_map[parent_name]
                    parent_indices[i] = parent_idx

                    template_key = (inst.hier_module.name, inst.name)
                    if template_key not in template_groups:
                        template_groups[template_key] = []
                    template_groups[template_key].append(i)

        return parent_indices, template_groups

    def _optimize_placement(self, state: PlacementState):
        """Run the main gradient descent optimization loop.

        Tracks and restores the lowest-cost state reached during optimization.
        """
        cost_history = deque(maxlen=5)

        # Initialize best-known state from the initial placement
        initial_cost = self._calculate_total_cost_from_state(state)
        best_cost = initial_cost["total"]
        best_pos = np.copy(state.pos)
        best_ar = np.copy(state.aspect_ratios)

        last_iteration = -1
        for iteration in range(self.config.max_iterations):
            last_iteration = iteration
            # Compute all gradients
            gradients = self._compute_all_gradients(state)

            # Update positions and aspect ratios
            self._apply_gradient_update(state, gradients)

            # Update derived quantities
            self._update_soft_macro_sizes(state)
            self._update_region_centroids(state)
            self._update_instance_geometries(state)

            # Track best state by cost every iteration
            current_cost = self._calculate_total_cost_from_state(state)
            if current_cost["total"] < best_cost:
                best_cost = current_cost["total"]
                best_pos = np.copy(state.pos)
                best_ar = np.copy(state.aspect_ratios)

            # Periodic cost logging and convergence check
            if iteration % self.config.cost_logging_interval == 0:
                if self._check_convergence(state, cost_history, iteration):
                    break

            # Visualization and user interrupt handling
            if not self._handle_visualization_and_events(state, iteration):
                break

        # Restore best-known state before exiting optimization
        state.pos = best_pos
        state.aspect_ratios = best_ar
        self._update_soft_macro_sizes(state)
        self._update_region_centroids(state)
        self._update_instance_geometries(state)
        # Log final best cost once
        self._log_cost(self._calculate_total_cost_from_state(state), max(last_iteration, 0), prefix="Best")

        if self.visualizer.enabled:
            self.visualizer.quit()

    def _compute_all_gradients(self, state: PlacementState) -> dict:
        """Compute all gradient components."""
        # Create consistent instance_map for gradient calculations
        instance_map = {inst.name: i for i, inst in enumerate(state.instances)}

        grad_hpwl_pos, grad_hpwl_r = self._calculate_hpwl_gradient(
            state.pos, state.sizes, state.aspect_ratios, state.areas, state.is_soft, state.nets, instance_map
        )
        grad_overlap_pos, grad_overlap_r = self._calculate_overlap_gradient(
            state.pos, state.sizes, state.aspect_ratios, state.areas, state.is_soft
        )
        grad_shape_r = self._calculate_shape_gradient(state.aspect_ratios, state.r_min, state.r_max, state.r_pref, state.is_soft)
        grad_bound_pos = self._calculate_region_gradient(state.pos, state.sizes, state.regions, state.instance_to_region)
        grad_rigidity_pos = self._calculate_rigidity_gradient(state.pos, state.parent_indices, state.template_groups)
        grad_anchor_pos = self._calculate_anchor_gradient(state.pos, state.anchor_pos, state.nets, instance_map)
        grad_dummy_pos = self._calculate_dummy_gradient(state.pos, state.regions, state.instance_to_region)

        # Combine gradients with weights
        grad_pos = (
            self.config.lambda_wire * grad_hpwl_pos
            + self.config.lambda_overlap * grad_overlap_pos
            + self.config.lambda_bound * grad_bound_pos
            + self.config.lambda_rigid * grad_rigidity_pos
            + self.config.lambda_anchor * grad_anchor_pos
            + self.config.w_dummy * grad_dummy_pos
        )
        grad_r = (
            self.config.lambda_wire * grad_hpwl_r
            + self.config.lambda_overlap * grad_overlap_r
            + self.config.lambda_shape * grad_shape_r
        )

        return {"pos": grad_pos, "aspect_ratio": grad_r}

    def _apply_gradient_update(self, state: PlacementState, gradients: dict):
        """Apply one optimization step with optional Armijo backtracking line search and gradient clipping.

        This function does NOT update sizes/regions/geometries; the caller updates derived
        quantities after a step. If line search fails to find a decreasing step, no update is applied.
        """
        grad_pos = gradients["pos"]
        grad_r = gradients["aspect_ratio"]

        # Gradient clipping to improve stability
        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
            gnorm_sq = float(np.sum(grad_pos**2) + np.sum(grad_r**2))
            gnorm = np.sqrt(gnorm_sq)
            if gnorm > self.config.max_grad_norm:
                scale = self.config.max_grad_norm / (gnorm + 1e-12)
                grad_pos = grad_pos * scale
                grad_r = grad_r * scale
                gnorm_sq = float(np.sum(grad_pos**2) + np.sum(grad_r**2))
        else:
            gnorm_sq = float(np.sum(grad_pos**2) + np.sum(grad_r**2))

        # Base step size
        alpha = self.config.step_size

        if not self.config.use_line_search:
            # Plain GD update
            state.pos = state.pos - alpha * grad_pos
            state.aspect_ratios = state.aspect_ratios - alpha * grad_r
            state.aspect_ratios = np.clip(state.aspect_ratios, state.r_min, state.r_max)
            return

        # Armijo backtracking line search
        f0 = self._calculate_total_cost_from_state(state)["total"]
        c = self.config.armijo_c
        beta = self.config.backtrack_beta
        max_bt = self.config.max_backtracks

        accepted = False
        pos0 = state.pos
        r0 = state.aspect_ratios

        for _ in range(max_bt):
            pos_trial = pos0 - alpha * grad_pos
            r_trial = np.clip(r0 - alpha * grad_r, state.r_min, state.r_max)
            f_trial = self._evaluate_cost_trial(state, pos_trial, r_trial)["total"]
            if f_trial <= f0 - c * alpha * gnorm_sq:
                # Accept step
                state.pos = pos_trial
                state.aspect_ratios = r_trial
                accepted = True
                break
            alpha *= beta

        if not accepted:
            # No update; keep current state to avoid cost blow-up
            log.debug("Line search failed to find decreasing step; skipping update this iteration")

    def _update_soft_macro_sizes(self, state: PlacementState):
        """Update sizes for soft macros based on aspect ratios."""
        soft_indices = np.where(state.is_soft)[0]
        if len(soft_indices) > 0:
            r = state.aspect_ratios[soft_indices]
            A = state.areas[soft_indices]
            state.sizes[soft_indices, 0] = np.sqrt(A * r)  # Width
            state.sizes[soft_indices, 1] = np.sqrt(A / r)  # Height

    def _update_region_centroids(self, state: PlacementState):
        """Update region geometries to follow instance centroids."""
        for region in state.regions:
            region_pos = state.pos[region.instances]
            centroid = np.mean(region_pos, axis=0)
            x_min, y_min, x_max, y_max = region.geom.bounds
            w, h = x_max - x_min, y_max - y_min
            region.geom = box(centroid[0] - w / 2, centroid[1] - h / 2, centroid[0] + w / 2, centroid[1] + h / 2)

    def _update_instance_geometries(self, state: PlacementState):
        """Update instance geometries based on current positions and sizes."""
        for i, inst in enumerate(state.instances):
            x, y = state.pos[i]
            w, h = state.sizes[i]
            inst.draw.geom = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def _check_convergence(self, state: PlacementState, cost_history: deque, iteration: int) -> bool:
        """Log cost and check for convergence."""
        cost = self._calculate_total_cost_from_state(state)
        self._log_cost(cost, iteration)

        cost_history.append(cost["total"])
        if len(cost_history) == cost_history.maxlen:
            rel_change = (max(cost_history) - min(cost_history)) / (np.mean(cost_history) + 1e-6)
            if rel_change < self.config.convergence_tol:
                log.info(f"Converged after {iteration} iterations.")
                return True
        return False

    def _handle_visualization_and_events(self, state: PlacementState, iteration: int) -> bool:
        """Handle visualization and pygame events. Returns False if should quit."""
        if self.config.visualize and iteration % self.config.visualization_interval == 0:
            self.visualizer.draw(state.instances, state.nets, state.regions)

        if self.visualizer.enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.visualizer.quit()
                    return False

        return self.visualizer.enabled

    def _log_cost(self, cost: dict, iteration: int, prefix: str = "Iter"):
        """Log cost breakdown in a consistent format."""
        # Support both legacy 'region' and new 'bound' keys
        region_cost = cost.get("bound", cost.get("region", 0.0))
        anchor_cost = cost.get("anchor", 0.0)
        log.info(
            f"{prefix} {iteration}: cost: {cost['total']:.2f} "
            f"(HPWL: {cost['hpwl']:.2f}, Overlap: {cost['overlap']:.2f}, "
            f"Bound: {region_cost:.2f}, Shape: {cost['shape']:.2f}, "
            f"Rigid: {cost['rigid']:.2f}, Anchor: {anchor_cost:.2f})"
        )

    def _calculate_total_cost_from_state(self, state: PlacementState) -> dict:
        """Convenience wrapper for calculating cost from PlacementState."""
        instance_map = {inst.name: i for i, inst in enumerate(state.instances)}
        return self._calculate_total_cost(
            state.pos,
            state.sizes,
            state.aspect_ratios,
            state.areas,
            state.is_soft,
            state.r_min,
            state.r_max,
            state.r_pref,
            state.nets,
            instance_map,
            state.regions,
            state.instance_to_region,
            state.parent_indices,
            state.template_groups,
            state.anchor_pos,
        )

    def _evaluate_cost_trial(self, state: PlacementState, pos_trial: np.ndarray, r_trial: np.ndarray) -> dict:
        """Evaluate cost for a trial (pos, aspect_ratio) without mutating state."""
        # Build trial sizes: copy current, update soft macros from areas and r_trial
        sizes_trial = np.copy(state.sizes)
        soft_indices = np.where(state.is_soft)[0]
        if len(soft_indices) > 0:
            r = r_trial[soft_indices]
            A = state.areas[soft_indices]
            sizes_trial[soft_indices, 0] = np.sqrt(A * r)
            sizes_trial[soft_indices, 1] = np.sqrt(A / r)
        instance_map = {inst.name: i for i, inst in enumerate(state.instances)}
        return self._calculate_total_cost(
            pos_trial,
            sizes_trial,
            r_trial,
            state.areas,
            state.is_soft,
            state.r_min,
            state.r_max,
            state.r_pref,
            state.nets,
            instance_map,
            state.regions,
            state.instance_to_region,
            state.parent_indices,
            state.template_groups,
            state.anchor_pos,
        )

    def _calculate_total_cost(
        self,
        pos,
        sizes,
        aspect_ratios,
        areas,
        is_soft,
        r_min,
        r_max,
        r_pref,
        nets,
        instance_map,
        regions,
        instance_to_region,
        parent_indices,
        template_groups,
        anchor_pos,
    ):
        """Calculate total placement cost including all penalty terms."""
        num_insts = pos.shape[0]
        # HPWL
        cost_hpwl = 0.0
        degrees = np.zeros(num_insts, dtype=float)
        for net in nets:
            if len(net.pins) < 2:
                continue
            pin_inst_indices = [instance_map[pin.instance.name] for pin in net.pins.values() if pin.instance.name in instance_map]
            if not pin_inst_indices:
                continue
            net_pos = pos[pin_inst_indices]
            min_coords = np.min(net_pos, axis=0)
            max_coords = np.max(net_pos, axis=0)
            cost_hpwl += np.sum(max_coords - min_coords)
            # Degree accumulation for anchor weighting
            for idx in pin_inst_indices:
                degrees[idx] += 1.0

        # Overlap
        cost_overlap = 0.0
        padding = self.config.macro_padding
        for i in range(num_insts):
            for j in range(i + 1, num_insts):
                overlap_x = max(0, (sizes[i, 0] + padding + sizes[j, 0] + padding) / 2 - abs(pos[i, 0] - pos[j, 0]))
                overlap_y = max(0, (sizes[i, 1] + padding + sizes[j, 1] + padding) / 2 - abs(pos[i, 1] - pos[j, 1]))
                cost_overlap += overlap_x * overlap_y

        # Boundary soft wall ("bound")
        cost_bound = 0.0
        for i in range(num_insts):
            region = regions[instance_to_region[i]]
            x_min, y_min, x_max, y_max = region.geom.bounds
            ix_min, iy_min, ix_max, iy_max = (
                pos[i, 0] - sizes[i, 0] / 2,
                pos[i, 1] - sizes[i, 1] / 2,
                pos[i, 0] + sizes[i, 0] / 2,
                pos[i, 1] + sizes[i, 1] / 2,
            )
            cost_bound += max(0, x_min - ix_min) ** 2 + max(0, ix_max - x_max) ** 2
            cost_bound += max(0, y_min - iy_min) ** 2 + max(0, iy_max - y_max) ** 2

        # Shape
        cost_shape = 0.0
        soft_indices = np.where(is_soft)[0]
        if len(soft_indices) > 0:
            r, r_p, r_mi, r_ma = (
                aspect_ratios[soft_indices],
                r_pref[soft_indices],
                r_min[soft_indices],
                r_max[soft_indices],
            )
            cost_shape = np.sum(((r - r_p) / (r_ma - r_mi)) ** 2)

        # Rigidity
        cost_rigid = 0.0
        for _, group_indices in template_groups.items():
            if len(group_indices) < 2:
                continue
            local_pos = np.array([pos[i] - pos[parent_indices[i]] for i in group_indices if parent_indices[i] != -1])
            if len(local_pos) > 0:
                avg_local_pos = np.mean(local_pos, axis=0)
                cost_rigid += np.sum(np.linalg.norm(local_pos - avg_local_pos, axis=1) ** 2)

        # Anchor: pull towards initial positions, stronger for low-degree instances
        cost_anchor = 0.0
        if anchor_pos is not None:
            # Weight per instance: 1 / (1 + degree)
            w = 1.0 / (1.0 + degrees)
            d = pos - anchor_pos
            cost_anchor = float(np.sum(w[:, None] * (d**2)))

        # Optional dummy: weak pull to region centroid
        cost_dummy = 0.0
        if self.config.w_dummy > 0.0:
            for region in regions:
                region_pos = pos[region.instances]
                if len(region.instances) == 0:
                    continue
                centroid = np.mean(region_pos, axis=0)
                diffs = region_pos - centroid
                cost_dummy += float(np.sum(diffs**2))

        total_cost = (
            self.config.lambda_wire * cost_hpwl
            + self.config.lambda_overlap * cost_overlap
            + self.config.lambda_bound * cost_bound
            + self.config.lambda_shape * cost_shape
            + self.config.lambda_rigid * cost_rigid
            + self.config.lambda_anchor * cost_anchor
            + self.config.w_dummy * cost_dummy
        )

        return {
            "total": total_cost,
            "hpwl": cost_hpwl,
            "overlap": cost_overlap,
            "bound": cost_bound,
            "region": cost_bound,  # legacy key for compatibility
            "shape": cost_shape,
            "rigid": cost_rigid,
            "anchor": cost_anchor,
            "dummy": cost_dummy,
        }

    def _legalize(self, instances, pos, sizes, nets, regions):
        """Legalize placement by snapping to grid and removing overlaps."""
        log.info("Legalizing placement")
        # Snap to grid
        pos = np.round(pos / self.config.grid_size) * self.config.grid_size

        # Greedy overlap removal
        for _ in range(10):  # 10 iterations of overlap removal
            overlap_found = False
            for i in range(len(instances)):
                for j in range(i + 1, len(instances)):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]

                    w_sum = (sizes[i, 0] + self.config.macro_padding + sizes[j, 0] + self.config.macro_padding) / 2
                    h_sum = (sizes[i, 1] + self.config.macro_padding + sizes[j, 1] + self.config.macro_padding) / 2

                    overlap_x = w_sum - abs(dx)
                    overlap_y = h_sum - abs(dy)

                    if overlap_x > 0 and overlap_y > 0:
                        overlap_found = True
                        # Move them apart
                        move_x = overlap_x / 2 * np.sign(dx) if dx != 0 else overlap_x / 2
                        move_y = overlap_y / 2 * np.sign(dy) if dy != 0 else overlap_y / 2
                        pos[i, 0] += move_x
                        pos[i, 1] += move_y
                        pos[j, 0] -= move_x
                        pos[j, 1] -= move_y
            if not overlap_found:
                break

        # Update final geometries
        for i, inst in enumerate(instances):
            x, y = pos[i]
            w, h = sizes[i]
            inst.draw.geom = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)

        log.info("Legalization finished.")
        if self.config.visualize:
            self.visualizer = Visualizer(enabled=True)
            self.visualizer.draw(instances, nets, regions)

            # Save final placement image
            output_dir = "data/images/placer"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"final_placement_{self.db.design.name}.png")
            pygame.image.save(self.visualizer.screen, output_path)
            log.info(f"Saved final placement image to {output_path}")

            time.sleep(2)
            self.visualizer.quit()

    def _calculate_rigidity_gradient(self, pos, parent_indices, template_groups):
        """Calculate gradient of rigidity penalty term."""
        grad_pos = np.zeros_like(pos)
        for template_key, group_indices in template_groups.items():
            if len(group_indices) < 2:
                continue

            local_pos = np.zeros((len(group_indices), 2))
            for i, inst_idx in enumerate(group_indices):
                parent_idx = parent_indices[inst_idx]
                if parent_idx != -1:
                    local_pos[i] = pos[inst_idx] - pos[parent_idx]

            avg_local_pos = np.mean(local_pos, axis=0)

            for i, inst_idx in enumerate(group_indices):
                parent_idx = parent_indices[inst_idx]
                if parent_idx != -1:
                    grad = 2 * (local_pos[i] - avg_local_pos)
                    grad_pos[inst_idx] += grad
                    grad_pos[parent_idx] -= grad
        return grad_pos

    def _calculate_region_gradient(self, pos, sizes, regions, instance_to_region):
        """Calculate gradient of region constraint penalty term."""
        grad_pos = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            region = regions[instance_to_region[i]]
            x_min, y_min, x_max, y_max = region.geom.bounds

            ix_min = pos[i, 0] - sizes[i, 0] / 2
            iy_min = pos[i, 1] - sizes[i, 1] / 2
            ix_max = pos[i, 0] + sizes[i, 0] / 2
            iy_max = pos[i, 1] + sizes[i, 1] / 2

            if ix_min < x_min:
                grad_pos[i, 0] += -2 * (x_min - ix_min)
            if ix_max > x_max:
                grad_pos[i, 0] += -2 * (x_max - ix_max)
            if iy_min < y_min:
                grad_pos[i, 1] += -2 * (y_min - iy_min)
            if iy_max > y_max:
                grad_pos[i, 1] += -2 * (y_max - iy_max)
        return grad_pos

    def _calculate_anchor_gradient(self, pos, anchor_pos, nets, instance_map):
        """Gradient of anchor term pulling instances toward initial positions.
        Weight per instance is 1/(1+degree) to anchor mostly unconnected/loose cells.
        """
        grad_pos = np.zeros_like(pos)
        if anchor_pos is None:
            return grad_pos
        # Compute degrees per instance
        degrees = np.zeros(pos.shape[0], dtype=float)
        for net in nets:
            if len(net.pins) < 1:
                continue
            pin_inst_indices = [instance_map[pin.instance.name] for pin in net.pins.values() if pin.instance.name in instance_map]
            for idx in pin_inst_indices:
                degrees[idx] += 1.0
        w = 1.0 / (1.0 + degrees)
        grad_pos = 2.0 * (w[:, None] * (pos - anchor_pos))
        return grad_pos

    def _calculate_dummy_gradient(self, pos, regions, instance_to_region):
        """Gradient of dummy stabilizer: weak pull to each region's centroid."""
        grad_pos = np.zeros_like(pos)
        if self.config.w_dummy <= 0.0:
            return grad_pos
        # For each region, compute centroid and apply 2*(xi - c)
        for region in regions:
            idxs = region.instances
            if not idxs:
                continue
            region_pos = pos[idxs]
            centroid = np.mean(region_pos, axis=0)
            grad_pos[idxs] += 2.0 * (region_pos - centroid)
        return grad_pos

    def _calculate_hpwl_gradient(self, pos, sizes, aspect_ratios, areas, is_soft, nets, instance_map):
        """Calculate gradient of Half-Perimeter Wire Length (HPWL) cost."""
        grad_pos = np.zeros_like(pos)
        grad_r = np.zeros_like(aspect_ratios)
        # For soft macros, pin positions depend on size, which depends on r.
        # p_x = x + alpha * W = x + alpha * sqrt(A*r)
        # p_y = y + beta * H = y + beta * sqrt(A/r)
        # For now, we assume pins are at the center (alpha=0, beta=0), so HPWL gradient w.r.t. r is 0.
        # This will be updated when we have actual pin positions.
        for net in nets:
            if len(net.pins) < 2:
                continue

            pin_inst_indices = [instance_map[pin.instance.name] for pin in net.pins.values() if pin.instance.name in instance_map]
            if not pin_inst_indices:
                continue

            net_pos = pos[pin_inst_indices]
            min_coords = np.min(net_pos, axis=0)
            max_coords = np.max(net_pos, axis=0)

            for i, inst_idx in enumerate(pin_inst_indices):
                if net_pos[i, 0] == min_coords[0]:
                    grad_pos[inst_idx, 0] -= 1
                if net_pos[i, 0] == max_coords[0]:
                    grad_pos[inst_idx, 0] += 1
                if net_pos[i, 1] == min_coords[1]:
                    grad_pos[inst_idx, 1] -= 1
                if net_pos[i, 1] == max_coords[1]:
                    grad_pos[inst_idx, 1] += 1
        return grad_pos, grad_r

    def _calculate_overlap_gradient(self, pos, sizes, aspect_ratios, areas, is_soft):
        """Calculate gradient of overlap penalty term."""
        grad_pos = np.zeros_like(pos)
        grad_r = np.zeros_like(aspect_ratios)
        num_instances = pos.shape[0]
        padding = self.config.macro_padding
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                dx = abs(pos[i, 0] - pos[j, 0])
                dy = abs(pos[i, 1] - pos[j, 1])

                w_sum = (sizes[i, 0] + padding + sizes[j, 0] + padding) / 2
                h_sum = (sizes[i, 1] + padding + sizes[j, 1] + padding) / 2

                overlap_x = max(0, w_sum - dx)
                overlap_y = max(0, h_sum - dy)

                if overlap_x > 0 and overlap_y > 0:
                    # Gradient w.r.t. position
                    grad_x_pos = -overlap_x * np.sign(pos[i, 0] - pos[j, 0])
                    grad_y_pos = -overlap_y * np.sign(pos[i, 1] - pos[j, 1])

                    grad_pos[i, 0] += grad_x_pos
                    grad_pos[i, 1] += grad_y_pos
                    grad_pos[j, 0] -= grad_x_pos
                    grad_pos[j, 1] -= grad_y_pos

                    # Gradient w.r.t. aspect ratio
                    # d(overlap)/dr = d(overlap)/d(size) * d(size)/dr
                    # d(overlap_x)/dW_i = 0.5, d(overlap_y)/dH_i = 0.5
                    # dW/dr = 0.5 * np.sqrt(A/r), dH/dr = -0.5 * np.sqrt(A/r^3)
                    if is_soft[i]:
                        r_i = aspect_ratios[i]
                        A_i = areas[i]
                        dW_dr_i = 0.5 * np.sqrt(A_i / r_i)
                        dH_dr_i = -0.5 * np.sqrt(A_i / (r_i**3))
                        grad_r[i] += 0.5 * overlap_x * dW_dr_i + 0.5 * overlap_y * dH_dr_i
                    if is_soft[j]:
                        r_j = aspect_ratios[j]
                        A_j = areas[j]
                        dW_dr_j = 0.5 * np.sqrt(A_j / r_j)
                        dH_dr_j = -0.5 * np.sqrt(A_j / (r_j**3))
                        grad_r[j] += 0.5 * overlap_x * dW_dr_j + 0.5 * overlap_y * dH_dr_j

        return grad_pos, grad_r

    def _calculate_shape_gradient(self, aspect_ratios, r_min, r_max, r_pref, is_soft):
        """Calculate gradient of shape penalty term for soft macros."""
        grad_r = np.zeros_like(aspect_ratios)
        soft_indices = np.where(is_soft)[0]
        if len(soft_indices) == 0:
            return grad_r

        r = aspect_ratios[soft_indices]
        r_p = r_pref[soft_indices]
        r_mi = r_min[soft_indices]
        r_ma = r_max[soft_indices]

        # C_shape = sum(((r - r_p) / (r_ma - r_mi))^2)
        # dC/dr = 2 * (r - r_p) / (r_ma - r_mi)^2
        grad_r[soft_indices] = 2 * (r - r_p) / ((r_ma - r_mi) ** 2)
        return grad_r


if __name__ == "__main__":
    from schematic_from_netlist.sastar_router.test_cases import create_hard_test_case

    log.basicConfig(level=log.INFO)

    # Create test cases assuming db is not populated
    db = create_hard_test_case("precision")
    placer_config = PlacerConfig()
    global_placer = GlobalPlacer(db, config=placer_config)
    global_placer.place_design()
