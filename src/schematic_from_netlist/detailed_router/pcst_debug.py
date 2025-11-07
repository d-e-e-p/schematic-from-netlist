import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class PCSTGridDebugger:
    """Debug and visualize PCST directional grid with merged NS/EW nodes"""

    def __init__(self, router, pcst_inputs):
        """
        Args:
            router: Your router object with node_id_to_coord method and DIR_TO_AXIS mapping
            pcst_inputs: PcstInputs dataclass with edges, prizes, costs, root
        """
        self.router = router
        self.pcst_inputs = pcst_inputs

        # Get PCST solution if available
        self.vertices = None
        self.edges_output = None

    def set_solution(self, vertices, edges_output):
        """Set the PCST solution for visualization"""
        self.vertices = vertices
        self.edges_output = edges_output

    def visualize_grid(self, net_name, grid_size=8, output_dir="data/images/droute"):
        """
        Create visualization focused on each terminal showing:
        - Grid cells around terminal
        - All nodes (NS/EW per cell)
        - All edges (with costs)
        - Selected path (if solution provided)

        Args:
            net_name: Name of the net
            grid_size: Size of grid to show around each terminal (e.g., 8x8)
            output_dir: Directory to save images
        """

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pcst_inputs = self.pcst_inputs
        edges = np.array(pcst_inputs.edges)
        prizes = np.array(pcst_inputs.prizes)
        costs = np.array(pcst_inputs.costs)
        root = pcst_inputs.root

        # Get selected edges if solution exists
        selected_edges_set = set()
        if self.edges_output is not None:
            selected_edges_set = set(self.edges_output)

        # Process each terminal
        for term in self.router.terminals:
            self._visualize_terminal_region(term, net_name, edges, prizes, costs, root, selected_edges_set, grid_size, output_path)

        log.info(f"Saved {len(self.router.terminals)} debug images to {output_path}")

    def _visualize_terminal_region(
        self, terminal, net_name, edges, prizes, costs, root, selected_edges_set, grid_size, output_path
    ):
        """Create focused visualization around a single terminal"""

        term_r, term_c = terminal.pt.x, terminal.pt.y
        term_axis = self.router.DIR_TO_AXIS[terminal.direction]

        # Calculate grid bounds around terminal
        half_size = grid_size // 2
        r_min, r_max = term_r - half_size, term_r + half_size
        c_min, c_max = term_c - half_size, term_c + half_size

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))

        # Get all cells and context
        all_grid_cells = set(self.router.coords_inside_poly([self.router.bounds]))
        blockage_cells = set(self.router.coords_inside_poly(self.router.obstacles))
        halo_cells = set(self.router.coords_inside_poly(self.router.halo_geoms))

        # Filter cells in view
        view_cells = [(r, c) for r, c in all_grid_cells if r_min <= r <= r_max and c_min <= c <= c_max]

        # --- 1. Draw grid cells ---
        for r, c in view_cells:
            # Determine cell color
            if (r, c) in blockage_cells:
                color = "red"
                alpha = 0.3
            elif (r, c) in halo_cells:
                color = "orange"
                alpha = 0.2
            else:
                color = "lightgray"
                alpha = 0.1

            rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor="gray", facecolor=color, alpha=alpha)
            ax.add_patch(rect)

            # Label cell coordinates
            ax.text(c, r, f"{r},{c}", fontsize=6, ha="center", va="center", color="gray", alpha=0.5)

        # --- 2. Draw nodes (NS and EW per cell) ---
        node_positions = {}  # node_id -> (x, y) for edge drawing

        # Offsets for NS and EW nodes within cell
        axis_offsets = {
            "NS": (0, 0.25),  # Slightly above center
            "EW": (0, -0.25),  # Slightly below center
        }

        for r, c in view_cells:
            for axis in ["NS", "EW"]:
                try:
                    node_id = self.router.coord_to_node_id(r, c, axis)
                except (KeyError, AttributeError):
                    continue

                dx, dy = axis_offsets[axis]
                x, y = c + dx, r + dy
                node_positions[node_id] = (x, y)

                # Determine node appearance
                has_prize = node_id < len(prizes) and prizes[node_id] > 0
                is_root = node_id == root
                is_in_solution = self.vertices is not None and node_id in self.vertices

                # Node styling
                if is_root:
                    marker, size, color = "*", 200, "gold"
                    edgecolor, linewidth = "red", 3
                elif has_prize:
                    marker, size, color = "o", 150, "gold"
                    edgecolor, linewidth = "darkgoldenrod", 2
                elif is_in_solution:
                    marker, size, color = "o", 100, "lightgreen"
                    edgecolor, linewidth = "darkgreen", 2
                else:
                    marker, size, color = "o", 80, "lightblue"
                    edgecolor, linewidth = "blue", 1

                ax.scatter(x, y, marker=marker, s=size, c=color, edgecolors=edgecolor, linewidths=linewidth, zorder=5)

                # Label node
                label = f"{axis}\n({int(r)},{int(c)})"
                if has_prize:
                    label += f"\n★{int(prizes[node_id])}"

                ax.text(
                    x + 0.10,
                    y + 0.10,
                    label,
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=edgecolor, linewidth=0.5),
                    zorder=6,
                )

        # --- 3. Draw edges ---
        for edge_idx, (node_a, node_b) in enumerate(edges):
            # Check if nodes are in view
            if node_a not in node_positions or node_b not in node_positions:
                continue

            x1, y1 = node_positions[node_a]
            x2, y2 = node_positions[node_b]

            # Get edge info
            try:
                r1, c1, axis1 = self.router.node_id_to_coord(node_a)
                r2, c2, axis2 = self.router.node_id_to_coord(node_b)
            except (KeyError, ValueError):
                continue

            is_selected = edge_idx in selected_edges_set
            cost = costs[edge_idx]

            # Determine edge type
            is_inter_cell = r1 != r2 or c1 != c2
            is_turn = not is_inter_cell and axis1 != axis2

            # Edge styling
            if is_selected:
                color = "green"
                linewidth = 3
                alpha = 1.0
                zorder = 4
            elif is_turn:
                color = "red"
                linewidth = 1.5
                alpha = 0.6
                zorder = 2
            elif cost == self.router.cost.base:
                color = "blue"
                linewidth = 1
                alpha = 0.3
                zorder = 1
            elif cost == self.router.cost.halo:
                color = "orange"
                linewidth = 1.5
                alpha = 0.5
                zorder = 2
            else:  # macro/blockage
                color = "red"
                linewidth = 2
                alpha = 0.5
                zorder = 2

            # Draw edge
            if is_turn:
                # Curved arrow for turns
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=linewidth, alpha=alpha, connectionstyle="arc3,rad=0.3"),
                    zorder=zorder,
                )
            else:
                # Straight line for inter-cell movement
                ax.plot([x1, x2], [y1, y2], "-", color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

            # Label edge cost if selected or turn
            # if is_selected or is_turn:
            if True:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    f"{int(cost)}",
                    fontsize=6,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor=color, linewidth=1),
                    zorder=7,
                )

        # --- 4. Highlight terminal ---
        ax.scatter(
            term_c, term_r, marker="s", s=400, facecolors="none", edgecolors="green", linewidths=4, zorder=10, label="Terminal"
        )

        term_label = f"{terminal.name}\n({term_r},{term_c})\n{terminal.direction}→{term_axis}"
        ax.text(
            term_c - 1,
            term_r - 1,
            term_label,
            fontsize=10,
            ha="center",
            fontweight="bold",
            color="green",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="green", linewidth=2),
            zorder=11,
        )

        # --- 5. Setup plot ---
        ax.set_xlim(c_min - 0.5, c_max + 0.5)
        ax.set_ylim(r_min - 0.5, r_max + 0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("Column (c)", fontsize=10)
        ax.set_ylabel("Row (r)", fontsize=10)

        # Title
        solution_status = "WITH SOLUTION" if self.edges_output is not None else "NO SOLUTION"
        ax.set_title(
            f"PCST Debug: {net_name} - Terminal: {terminal.name} ({solution_status})\n"
            f"Grid: {grid_size}×{grid_size} around ({term_r},{term_c})",
            fontsize=12,
            fontweight="bold",
        )

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gold",
                markersize=15,
                markeredgecolor="red",
                markeredgewidth=2,
                label="Root Terminal",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gold",
                markersize=12,
                markeredgecolor="darkgoldenrod",
                markeredgewidth=2,
                label="Prize Node",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lightgreen",
                markersize=10,
                markeredgecolor="darkgreen",
                markeredgewidth=2,
                label="In Solution",
            ),
            Line2D([0], [0], color="green", linewidth=3, label="Selected Edge"),
            Line2D([0], [0], color="red", linewidth=2, label="Turn Edge"),
            Line2D([0], [0], color="blue", linewidth=1, alpha=0.5, label="Normal Edge"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

        # Save
        filename = f"pcst_debug_{net_name}_{terminal.name}"
        filename = self.router.clean_hierarchical_name(filename) + ".svg"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        log.info(f"  Saved: {filepath}")

    def print_debug_stats(self, net_name):
        """Print detailed statistics about the grid"""
        pcst_inputs = self.pcst_inputs
        edges = pcst_inputs.edges
        prizes = pcst_inputs.prizes
        costs = pcst_inputs.costs
        root = pcst_inputs.root

        log.info(f"\n{'=' * 60}")
        log.info(f"PCST Grid Debug Stats for: {net_name}")
        log.info(f"{'=' * 60}")

        log.info(f"\n📊 Graph Structure:")
        log.info(f"  Total Nodes: {len(prizes)}")
        log.info(f"  Total Edges: {len(edges)}")

        try:
            root_coord = self.router.node_id_to_coord(root)
            log.info(f"  Root Node: {root} -> {root_coord}")
        except (KeyError, ValueError):
            log.info(f"  Root Node: {root}")

        log.info(f"\n💰 Cost Distribution:")
        log.info(f"  Base  cost:  {self.router.cost.base}")
        log.info(f"  Turn  cost:  {self.router.cost.turn}")
        log.info(f"  Halo  cost:  {self.router.cost.halo}")
        log.info(f"  Macro cost:  {self.router.cost.macro}")

        costs_array = np.array(costs)
        unique_costs, counts = np.unique(costs_array, return_counts=True)
        log.info(f"\n  Edge cost breakdown:")
        for cost, count in zip(unique_costs, counts):
            percentage = count / len(costs) * 100
            log.info(f"    Cost {int(cost):4d}: {count:6d} edges ({percentage:5.1f}%)")

        log.info(f"\n🎯 Terminals:")
        prizes_array = np.array(prizes)
        terminal_nodes = np.where(prizes_array > 0)[0]
        log.info(f"  Prize per terminal: {self.router.cost.prize}")
        log.info(f"  Total prize nodes: {len(terminal_nodes)}")
        log.info(f"  Total prize value: {int(prizes_array.sum())}")
        log.info(f"  Terminal details:")
        for term in self.router.terminals:
            try:
                axis = self.router.DIR_TO_AXIS[term.direction]
                node_id = self.router.coord_to_node_id(term.pt.x, term.pt.y, axis)
                prize_val = prizes_array[node_id] if node_id < len(prizes_array) else 0
                log.info(
                    f"    {term.name:15s} @ ({term.pt.x},{term.pt.y}) "
                    f"{term.direction}→{axis}  Node:{node_id:5d}  Prize:{int(prize_val):4d}"
                )
            except (KeyError, ValueError, AttributeError):
                log.error(f"    {term.name:15s} @ ({term.pt.x},{term.pt.y}) {term.direction} [ERROR]")
                breakpoint()

        log.info(f"\n🔍 Edge Type Breakdown:")
        intra_cell = 0
        inter_cell = 0

        for node_a, node_b in edges:
            try:
                r_a, c_a, axis_a = self.router.node_id_to_coord(node_a)
                r_b, c_b, axis_b = self.router.node_id_to_coord(node_b)

                if r_a == r_b and c_a == c_b:
                    intra_cell += 1
                else:
                    inter_cell += 1
            except (KeyError, ValueError):
                continue

        total = len(edges)
        log.info(f"  Intra-cell (turn) edges: {intra_cell:6d} ({intra_cell / total * 100:5.1f}%)")
        log.info(f"  Inter-cell (move) edges: {inter_cell:6d} ({inter_cell / total * 100:5.1f}%)")

        # Analyze turn costs
        turn_edges = sum(1 for c in costs if c == self.router.cost.turn)
        straight_edges = sum(1 for c in costs if c == self.router.cost.base)
        log.info(f"\n🔄 Turn Analysis:")
        log.info(f"  Base-cost edges:  {straight_edges:6d} ({straight_edges / total * 100:5.1f}%)")
        log.info(f"  Turn-cost edges:  {turn_edges:6d} ({turn_edges / total * 100:5.1f}%)")

        # Solution stats if available
        if self.edges_output is not None:
            log.info(f"\n✅ Solution Statistics:")
            log.info(f"  Selected edges: {len(self.edges_output)}")

            if self.vertices is not None:
                log.info(f"  Selected nodes: {len(self.vertices)}")

            # Calculate total cost
            total_cost = sum(costs[i] for i in self.edges_output)
            log.info(f"  Total path cost: {int(total_cost)}")

            # Count turns in solution
            solution_turns = sum(1 for i in self.edges_output if costs[i] == self.router.cost.turn)
            log.info(f"  Turns in path: {solution_turns}")

        log.info(f"\n{'=' * 60}\n")


# Usage in route_net method:
"""
def route_net(self, net_name: str, cost_overrides: Dict[str, Any] = None):
    '''Connect one net'''
    
    # Setup PCST grid
    pcst = self.setup_pcst_grid(net_name)
    
    # Run PCST solver
    vertices, edges_output = pcst.run()
    
    # Debug and visualize
    debugger = PCSTGridDebugger(self, pcst)
    debugger.set_solution(vertices, edges_output)  # Provide solution
    debugger.print_debug_stats(net_name)
    debugger.visualize_grid(net_name, grid_size=8)  # Creates images per terminal
    
    return vertices, edges_output
"""

