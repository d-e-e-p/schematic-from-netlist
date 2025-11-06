import logging as log
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class PCSTGridDebugger:
    """Debug and visualize PCST directional grid"""

    def __init__(self, router, pcst_inputs):
        """
        Args:
            router: Your router object with node_id_to_coord method
            pcst_inputs: PcstInputs dataclass with edges, prizes, costs, root
        """
        self.router = router
        self.pcst_inputs = pcst_inputs

    def visualize_grid(self, net_name, save_path=None):
        """Create comprehensive visualization of the PCST grid"""

        pcst_inputs = self.pcst_inputs
        edges = np.array(pcst_inputs.edges)
        prizes = np.array(pcst_inputs.prizes)
        costs = np.array(pcst_inputs.costs)
        root = pcst_inputs.root

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Cell-level grid with D-nodes
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cell_dnodes(ax1, prizes, root)

        # 2. Edge cost heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cost_heatmap(ax2, edges, costs)

        # 3. Intra-cell connections (turns)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_intra_cell_edges(ax3, edges, costs)

        # 4. Inter-cell connections (movement)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_inter_cell_edges(ax4, edges, costs)

        # 5. Terminal prizes
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_terminal_prizes(ax5, prizes, root)

        # 6. Graph structure
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_graph_structure(ax6, edges, costs, prizes)

        plt.suptitle(f"PCST Grid Debug: {net_name}", fontsize=16, fontweight="bold")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_cell_dnodes(self, ax, prizes, root):
        """Plot grid cells with 4 directional nodes each"""
        grid_cells = self.router.coords_inside_poly([self.router.bounds])
        blockage_cells = self.router.coords_inside_poly(self.router.obstacles)
        halo_cells = self.router.coords_inside_poly(self.router.halo_geoms)

        # Convert to sets for fast lookup
        blockage_set = set(blockage_cells)
        halo_set = set(halo_cells)

        # Draw cells
        for r, c in grid_cells:
            color = "lightgray"
            if (r, c) in blockage_set:
                color = "red"
            elif (r, c) in halo_set:
                color = "orange"

            rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor="black", facecolor=color, alpha=0.3)
            ax.add_patch(rect)

            # Draw 4 D-nodes per cell
            d_offsets = {"N": (0, -0.3), "E": (0.3, 0), "S": (0, 0.3), "W": (-0.3, 0)}
            for dir_name, (dx, dy) in d_offsets.items():
                node_id = self.router.coord_to_node_id(r, c, dir_name)

                # Highlight prize nodes
                if node_id < len(prizes) and prizes[node_id] > 0:
                    marker = "*" if node_id == root else "o"
                    color_node = "gold"
                    size = 8 if node_id == root else 6
                else:
                    marker = "o"
                    color_node = "blue"
                    size = 4

                ax.plot(c + dx, r + dy, marker, markersize=size, color=color_node)
                ax.text(c + dx + 0.1, r + dy + 0.1, f"{dir_name}", fontsize=5, ha="left", va="bottom", color="darkblue")

        # Mark terminals
        for term in self.router.terminals:
            r, c = term.pt.x, term.pt.y
            ax.plot(c, r, "s", markersize=12, color="green", markeredgewidth=2, fillstyle="none", label="Terminal")
            ax.text(c, r - 0.6, f"{term.direction}", fontsize=8, ha="center", color="green", fontweight="bold")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title("Cell Grid with D-Nodes\n(Gold = Prize, Blue = Normal)")
        ax.legend()

    def _plot_cost_heatmap(self, ax, edges, costs):
        """Show cost distribution across the grid"""
        grid_cells = list(self.router.coords_inside_poly([self.router.bounds]))

        if not grid_cells:
            ax.text(0.5, 0.5, "No grid cells", ha="center", va="center")
            return

        # Create cost matrix
        rows = [r for r, c in grid_cells]
        cols = [c for r, c in grid_cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        cost_matrix = np.zeros((max_r - min_r + 1, max_c - min_c + 1))
        count_matrix = np.zeros((max_r - min_r + 1, max_c - min_c + 1))

        # Aggregate costs per cell
        for i, (node_a, node_b) in enumerate(edges):
            r_a, c_a, dir_a = self.router.node_id_to_coord(node_a)
            r_b, c_b, dir_b = self.router.node_id_to_coord(node_b)

            # Add cost to both cells involved
            if min_r <= r_a <= max_r and min_c <= c_a <= max_c:
                cost_matrix[r_a - min_r, c_a - min_c] += costs[i]
                count_matrix[r_a - min_r, c_a - min_c] += 1

            if min_r <= r_b <= max_r and min_c <= c_b <= max_c:
                cost_matrix[r_b - min_r, c_b - min_c] += costs[i]
                count_matrix[r_b - min_r, c_b - min_c] += 1

        # Average the costs
        with np.errstate(divide="ignore", invalid="ignore"):
            cost_matrix = np.divide(cost_matrix, count_matrix)
            cost_matrix[~np.isfinite(cost_matrix)] = 0

        im = ax.imshow(cost_matrix, cmap="RdYlGn_r", aspect="auto", origin="lower")
        ax.set_title("Average Edge Cost per Cell")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, label="Avg Cost")

    def _plot_intra_cell_edges(self, ax, edges, costs):
        """Visualize turn costs within cells"""
        # Sample a few cells to show detail
        grid_cells = list(self.router.coords_inside_poly([self.router.bounds]))

        # Choose up to 4 representative cells
        num_samples = min(4, len(grid_cells))
        if num_samples == 0:
            ax.text(0.5, 0.5, "No grid cells", ha="center", va="center")
            ax.axis("off")
            return

        sample_cells = grid_cells[:num_samples]

        for idx, (r, c) in enumerate(sample_cells):
            row_idx = idx // 2
            col_idx = idx % 2
            ax_sub = plt.subplot(2, 2, idx + 1)

            # Draw D-nodes
            d_offsets = {"N": (0, -0.3), "E": (0.3, 0), "S": (0, 0.3), "W": (-0.3, 0)}
            node_pos = {}
            for dir_name, (dx, dy) in d_offsets.items():
                node_id = self.router.coord_to_node_id(r, c, dir_name)
                node_pos[node_id] = (dx, dy)
                ax_sub.plot(dx, dy, "o", markersize=10, color="blue")
                ax_sub.text(dx, dy, dir_name, fontsize=8, ha="center", va="center", color="white", fontweight="bold")

            # Draw intra-cell edges
            for i, (node_a, node_b) in enumerate(edges):
                r_a, c_a, dir_a = self.router.node_id_to_coord(node_a)
                r_b, c_b, dir_b = self.router.node_id_to_coord(node_b)

                # Only intra-cell edges for this cell
                if r_a == r and c_a == c and r_b == r and c_b == c:
                    if node_a in node_pos and node_b in node_pos:
                        x1, y1 = node_pos[node_a]
                        x2, y2 = node_pos[node_b]

                        # Color by cost
                        if costs[i] == 0:
                            color = "green"
                            alpha = 0.4
                        elif costs[i] == self.router.cost.turn:
                            color = "red"
                            alpha = 0.7
                        else:
                            color = "orange"
                            alpha = 0.6

                        # Draw curved arrow
                        ax_sub.annotate(
                            "",
                            xy=(x2, y2),
                            xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, lw=2, connectionstyle="arc3,rad=0.3"),
                        )

                        # Label cost at midpoint
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ax_sub.text(
                            mid_x,
                            mid_y,
                            f"{int(costs[i])}",
                            fontsize=6,
                            color=color,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                        )

            ax_sub.set_xlim(-0.5, 0.5)
            ax_sub.set_ylim(-0.5, 0.5)
            ax_sub.set_aspect("equal")
            ax_sub.set_title(f"Cell ({r},{c}) - Turn Costs\nGreen=0, Red=Turn", fontsize=8)
            ax_sub.grid(True, alpha=0.3)

        ax.axis("off")

    def _plot_inter_cell_edges(self, ax, edges, costs):
        """Show movement between cells"""
        grid_cells = self.router.coords_inside_poly([self.router.bounds])
        blockage_cells = set(self.router.coords_inside_poly(self.router.obstacles))
        halo_cells = set(self.router.coords_inside_poly(self.router.halo_geoms))

        # Draw cells
        for r, c in grid_cells:
            color = "lightgray"
            if (r, c) in blockage_cells:
                color = "red"
            elif (r, c) in halo_cells:
                color = "orange"

            rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor="gray", facecolor=color, alpha=0.2)
            ax.add_patch(rect)
            ax.plot(c, r, "o", markersize=4, color="blue")

        # Draw inter-cell edges
        edge_types = {"base": [], "halo": [], "macro": []}

        for i, (node_a, node_b) in enumerate(edges):
            r_a, c_a, dir_a = self.router.node_id_to_coord(node_a)
            r_b, c_b, dir_b = self.router.node_id_to_coord(node_b)

            # Only inter-cell edges
            if not (r_a == r_b and c_a == c_b):
                if costs[i] == self.router.cost.base:
                    edge_types["base"].append(((c_a, r_a), (c_b, r_b)))
                elif costs[i] == self.router.cost.halo:
                    edge_types["halo"].append(((c_a, r_a), (c_b, r_b)))
                else:
                    edge_types["macro"].append(((c_a, r_a), (c_b, r_b)))

        # Plot edges by type
        for (x1, y1), (x2, y2) in edge_types["base"]:
            ax.plot([x1, x2], [y1, y2], "-", color="green", alpha=0.3, linewidth=1)
        for (x1, y1), (x2, y2) in edge_types["halo"]:
            ax.plot([x1, x2], [y1, y2], "-", color="orange", alpha=0.5, linewidth=1.5)
        for (x1, y1), (x2, y2) in edge_types["macro"]:
            ax.plot([x1, x2], [y1, y2], "-", color="red", alpha=0.5, linewidth=2)

        # Create legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="green", lw=2, label=f"Normal ({len(edge_types['base'])})"),
            Line2D([0], [0], color="orange", lw=2, label=f"Halo ({len(edge_types['halo'])})"),
            Line2D([0], [0], color="red", lw=2, label=f"Blockage ({len(edge_types['macro'])})"),
        ]

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title("Inter-Cell Movement Edges")
        ax.legend(handles=legend_elements, loc="best")

    def _plot_terminal_prizes(self, ax, prizes, root):
        """Show prize distribution"""
        grid_cells = self.router.coords_inside_poly([self.router.bounds])

        # Draw cells faintly
        for r, c in grid_cells:
            rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=0.5, edgecolor="gray", facecolor="white", alpha=0.5)
            ax.add_patch(rect)

        # Draw all D-nodes with prizes
        d_offsets = {"N": (0, -0.3), "E": (0.3, 0), "S": (0, 0.3), "W": (-0.3, 0)}

        for node_id in range(len(prizes)):
            if prizes[node_id] > 0:
                r, c, dir_name = self.router.node_id_to_coord(node_id)
                dx, dy = d_offsets[dir_name]

                marker = "*" if node_id == root else "o"
                size = 15 if node_id == root else 10

                ax.plot(c + dx, r + dy, marker, markersize=size, color="gold", markeredgecolor="red", markeredgewidth=2)
                ax.text(c + dx, r + dy - 0.2, f"{int(prizes[node_id])}", fontsize=7, ha="center", color="red", fontweight="bold")
                ax.text(c + dx, r + dy + 0.2, f"{dir_name}\n({r},{c})", fontsize=6, ha="center", color="blue")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Terminal Prizes (* = Root node {root})")

    def _plot_graph_structure(self, ax, edges, costs, prizes):
        """Show graph connectivity using NetworkX"""
        G = nx.Graph()

        # Add edges with costs
        for i, (node_a, node_b) in enumerate(edges):
            G.add_edge(node_a, node_b, weight=costs[i])

        # Node colors based on prizes
        node_colors = []
        for n in G.nodes():
            if n < len(prizes) and prizes[n] > 0:
                node_colors.append("gold")
            else:
                node_colors.append("lightblue")

        # Use spring layout for visualization (sample if too large)
        if G.number_of_nodes() > 200:
            # Sample subgraph around terminals
            terminal_nodes = [i for i in range(len(prizes)) if prizes[i] > 0]
            subgraph_nodes = set(terminal_nodes)
            for term in terminal_nodes:
                neighbors = list(G.neighbors(term))[:5]  # Get a few neighbors
                subgraph_nodes.update(neighbors)
            G = G.subgraph(list(subgraph_nodes))
            ax.set_title(f"Graph Structure (Sampled)\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        else:
            ax.set_title(f"Graph Structure\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=node_colors[: len(G.nodes())],
            node_size=30,
            with_labels=False,
            edge_color="gray",
            alpha=0.6,
            width=0.5,
        )

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
        log.info(f"  Root Node: {root} -> {self.router.node_id_to_coord(root)}")

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
        for node_id in terminal_nodes:
            coord = self.router.node_id_to_coord(node_id)
            log.info(f"    Node {node_id:4d} -> {coord} (prize: {int(prizes_array[node_id])})")

        log.info(f"\n🔍 Edge Type Breakdown:")
        intra_cell = 0
        inter_cell = 0

        for node_a, node_b in edges:
            r_a, c_a, dir_a = self.router.node_id_to_coord(node_a)
            r_b, c_b, dir_b = self.router.node_id_to_coord(node_b)

            if r_a == r_b and c_a == c_b:
                intra_cell += 1
            else:
                inter_cell += 1

        total = len(edges)
        log.info(f"  Intra-cell (turn) edges: {intra_cell:6d} ({intra_cell / total * 100:5.1f}%)")
        log.info(f"  Inter-cell (move) edges: {inter_cell:6d} ({inter_cell / total * 100:5.1f}%)")

        # Analyze turn costs
        turn_edges = sum(1 for c in costs if c == self.router.cost.turn)
        straight_edges = sum(1 for c in costs if c == 0)
        log.info(f"\n🔄 Turn Analysis:")
        log.info(f"  Zero-cost edges: {straight_edges:6d} ({straight_edges / total * 100:5.1f}%)")
        log.info(f"  Turn-cost edges: {turn_edges:6d} ({turn_edges / total * 100:5.1f}%)")

        log.info(f"\n{'=' * 60}\n")


# Usage example:
"""
# In your router code:
pcst = self.setup_pcst_grid(net_name)
debugger = PCSTGridDebugger(self, pcst)
debugger.print_debug_stats(net_name)
debugger.visualize_grid(net_name, save_path=f'debug_{net_name}.png')
"""

