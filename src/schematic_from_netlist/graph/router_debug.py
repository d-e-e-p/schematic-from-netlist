from __future__ import annotations

import logging as log
import os
import time
from typing import TYPE_CHECKING, Dict, List, Tuple

import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from tabulate import tabulate

from schematic_from_netlist.database.netlist_structures import Module
from schematic_from_netlist.graph.routing_helpers import get_halo_geometries, get_macro_geometries
from schematic_from_netlist.graph.routing_utils import Junction, Pin, Topology

if TYPE_CHECKING:
    from schematic_from_netlist.database.netlist_structures import Module


class RouterDebugger:
    def log_junction_summary(self, junctions: Dict[Module, List[Topology]]):
        """Log detailed summary of inserted junctions."""
        for module, topos in junctions.items():
            log.debug(f"module {module.name=} size {module.draw.geom}")
            macros = get_macro_geometries(module)
            if not macros.is_empty:
                log.debug(f"  Macro blockages at: {macros.wkt}")
            for topo in topos:
                # Log detailed junction info
                for junction in topo.junctions:
                    log.debug(f"Inserting {junction.name=} in {topo.net.name} at {junction.location}")
                    for child in junction.children:
                        if isinstance(child, Junction):
                            log.debug(f"  Connected to junction {child.name} at {child.location}")
                        elif isinstance(child, Pin):
                            log.debug(f"  Connected to pin {child.full_name} at {child.draw.geom}")
                        else:
                            log.debug(f"  Connected to unknown child type {type(child)}")

        summary = []
        for module, topos in junctions.items():
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

    def plot_junction_summary(self, junctions: Dict[Module, List[Topology]], stage: str = "", title: str = ""):
        """
        Generate per-module schematic overview plots showing macros, pins, junctions, and existing net geometries.
        """
        out_dir = "data/images/summary"
        os.makedirs(out_dir, exist_ok=True)

        cmap = plt.get_cmap("tab20")  # color map for nets

        for module, topos in junctions.items():
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Module: {module.name} {title}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            # --- Draw macros ---
            macros = get_macro_geometries(module)
            if not macros.is_empty:
                if isinstance(macros, MultiPolygon):
                    for sub in macros.geoms:
                        x, y = sub.exterior.xy
                        ax.fill(x, y, color="lightgrey", alpha=0.6)
                else:
                    if isinstance(macros, Polygon):
                        x, y = macros.exterior.xy
                        ax.fill(x, y, color="lightgrey", alpha=0.6)

            # --- Draw halos ---
            halos = get_halo_geometries(macros)
            if not halos.is_empty:
                if isinstance(halos, MultiPolygon):
                    for sub in halos.geoms:
                        x, y = sub.exterior.xy
                        ax.plot(x, y, color="blue", ls="--", lw=1)
                else:
                    if isinstance(halos, Polygon):
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
                    if False:
                        for child in junction.children:
                            if isinstance(child, Pin) and hasattr(child.draw, "geom") and child.draw.geom:
                                pgeom = child.draw.geom
                                if pgeom is None:
                                    continue
                                if pgeom is None:
                                    continue
                                if isinstance(pgeom, Point):
                                    px, py = float(pgeom.x), float(pgeom.y)
                                else:
                                    centroid = pgeom.centroid
                                    px, py = float(centroid.x), float(centroid.y)
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

    def plot_cost_calculation(
        self,
        topology: Topology,
        paths_with_metrics,
        crossing_points,
        context: "RoutingContext",
        plot_filename_prefix: str | None = None,
    ):
        # --- Plotting logic ---
        out_dir = "data/images/cost_debug"
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal", adjustable="box")
        title = f"Cost Calculation for Net: {topology.net.name}"
        if context.module:
            title += f" in {context.module.name}"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # --- Draw other nets for context ---
        for geom in context.other_nets_geoms:
            x, y = geom.xy
            ax.plot(x, y, color="lightgray", lw=0.75, zorder=1)

        # --- Draw macros ---
        if not context.macros.is_empty:
            if isinstance(context.macros, MultiPolygon):
                for sub in context.macros.geoms:
                    x, y = sub.exterior.xy
                    ax.fill(x, y, color="lightgrey", alpha=0.6, zorder=2)
            else:
                if isinstance(context.macros, Polygon):
                    x, y = context.macros.exterior.xy
                    ax.fill(x, y, color="lightgrey", alpha=0.6, zorder=2)

        # --- Draw halos ---
        if not context.halos.is_empty:
            if isinstance(context.halos, MultiPolygon):
                for sub in context.halos.geoms:
                    x, y = sub.exterior.xy
                    ax.plot(x, y, color="blue", ls="--", lw=1, zorder=3)
            else:
                if isinstance(context.halos, Polygon):
                    x, y = context.halos.exterior.xy
                    ax.plot(x, y, color="blue", ls="--", lw=1, zorder=3)

        # --- Draw paths and labels ---
        total_cost_for_plot = 0
        for path, metrics in paths_with_metrics:
            x, y = path.xy
            ax.plot(x, y, color="red", lw=2, zorder=4)
            total_cost_for_plot += metrics.total_cost

            # Add labels for cost components
            label_pos = path.interpolate(0.5, normalized=True)
            label_text = (
                f"wl: {metrics.cost_wirelength:.0f}\n"
                f"macro: {metrics.cost_macro:.0f}\n"
                f"halo: {metrics.cost_halo:.0f}\n"
                f"cong: {metrics.cost_congestion:.0f}\n"
                f"cross: {metrics.cost_crossing:.0f}\n"
                f"track: {metrics.cost_track_overlap:.0f}\n"
                f"pen: {metrics.cost_macro_junction_penalty + metrics.cost_halo_junction_penalty:.0f}\n"
                f"TOTAL: {metrics.total_cost:.0f}"
            )
            ax.text(
                label_pos.x,
                label_pos.y,
                label_text,
                fontsize=8,
                color="darkgreen",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2),
                zorder=6,
            )

        # --- Draw crossing points ---
        for p in crossing_points:
            ax.scatter(p.x, p.y, c="red", s=40, marker="x", zorder=5, lw=1)

        # --- Draw junctions and pins ---
        for junction in topology.junctions:
            jx, jy = junction.location.x, junction.location.y
            ax.scatter(jx, jy, c="blue", s=80, marker="x", zorder=5)
            ax.text(jx + 0.5, jy + 0.5, junction.name, fontsize=7, color="blue", zorder=6)

        pins = [p for p in topology.net.pins.values() if hasattr(p.draw, "geom") and p.draw.geom is not None]
        for pin in pins:
            pgeom = pin.draw.geom
            if pgeom is None:
                continue
            if isinstance(pgeom, Point):
                px, py = pgeom.x, pgeom.y
            else:
                px, py = pgeom.centroid.x, pgeom.centroid.y
            ax.scatter(px, py, c="black", s=20, marker="o", zorder=5)
            ax.text(px + 0.5, py + 0.5, pin.full_name, fontsize=6, color="black", zorder=6)

        title = f"Cost Calculation for Net: {topology.net.name}"
        if context.module:
            title += f" in {context.module.name}"
        title += f"\nTotal Cost: {total_cost_for_plot:.2f}"
        ax.set_title(title)
        fig.tight_layout()

        # Save figure
        safe_module_name = context.module.name.replace("/", "_") if context.module else "unknown_module"
        safe_net_name = topology.net.name.replace("/", "_")
        filename = f"{plot_filename_prefix or ''}{safe_module_name}_{safe_net_name}_cost.png"
        full_path = os.path.join(out_dir, filename)
        plt.savefig(full_path, dpi=200)
        plt.close(fig)
        log.debug(f"Saved cost debug plot for net {topology.net.name} -> {full_path}")

    def _plot_junction_move_heatmap(
        self,
        module: Module,
        topology: Topology,
        moved_junction: Junction,
        tried_locations_costs: Dict[Tuple[float, float], float],
        best_location: Point,
        min_cost: float,
        macros: Polygon,
        halos: Polygon,
        filename_prefix: str,
    ):
        """Generate a heatmap of junction move costs."""
        out_dir = "data/images/heatmaps"
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Junction Move Heatmap: {moved_junction.name} in {topology.net.name}")

        # --- Draw macros and halos ---
        if not macros.is_empty:
            if isinstance(macros, MultiPolygon):
                for sub in macros.geoms:
                    x, y = sub.exterior.xy
                    ax.fill(x, y, color="lightgrey", alpha=0.6)
            else:
                x, y = macros.exterior.xy
                ax.fill(x, y, color="lightgrey", alpha=0.6)
        if not halos.is_empty:
            if isinstance(halos, MultiPolygon):
                for sub in halos.geoms:
                    x, y = sub.exterior.xy
                    ax.plot(x, y, color="blue", ls="--", lw=1)
            else:
                x, y = halos.exterior.xy
                ax.plot(x, y, color="blue", ls="--", lw=1)

        # --- Draw other junctions and pins for context ---
        for junction in topology.junctions:
            if junction is not moved_junction:
                jx, jy = junction.location.x, junction.location.y
                ax.scatter(jx, jy, c="grey", s=50, marker="x")
        pins = [p for p in topology.net.pins.values() if hasattr(p.draw, "geom") and p.draw.geom is not None]
        for pin in pins:
            pgeom = pin.draw.geom
            px, py = (pgeom.x, pgeom.y) if isinstance(pgeom, Point) else (pgeom.centroid.x, pgeom.centroid.y)
            ax.scatter(px, py, c="black", s=20, marker="o")

        # --- Prepare heatmap data ---
        locations = []
        costs_perc = []
        if min_cost > 0:
            for loc, cost in tried_locations_costs.items():
                locations.append(loc)
                if cost == float("inf"):
                    # Use a high value for invalid spots, maybe max of others + 50%
                    valid_costs = [c for c in tried_locations_costs.values() if c != float("inf")]
                    max_perc = max([((c - min_cost) / min_cost) * 100 for c in valid_costs]) if valid_costs else 100
                    costs_perc.append(max_perc + 50)
                else:
                    percentage_increase = ((cost - min_cost) / min_cost) * 100
                    costs_perc.append(percentage_increase)

        if not locations:
            log.warning("No locations to plot for heatmap.")
            plt.close(fig)
            return

        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]

        # --- Plot heatmap ---
        sc = ax.scatter(x_coords, y_coords, c=costs_perc, cmap="viridis_r", s=120, alpha=0.9, edgecolors="k", linewidth=0.5)
        cbar = plt.colorbar(sc, ax=ax, label="Cost increase (%) over best")
        cbar.set_alpha(1)

        # Annotate points with cost percentage
        for i, txt in enumerate(costs_perc):
            ax.annotate(
                f"{txt:.0f}%", (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7
            )

        # Mark best location
        ax.scatter(best_location.x, best_location.y, c="red", s=200, marker="*", label="Best Location")
        ax.legend()

        fig.tight_layout()
        filename = f"{module.name}_{topology.net.name}_{filename_prefix}.png"
        full_path = os.path.join(out_dir, filename)
        plt.savefig(full_path, dpi=200)
        plt.close(fig)
        log.info(f"Saved junction move heatmap -> {full_path}")
