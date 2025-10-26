import logging as log
import os

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.ops import unary_union

from schematic_from_netlist.astar_router.ar_cost_estimator import CostEstimator
from schematic_from_netlist.astar_router.ar_occupancy import OccupancyMap
from schematic_from_netlist.database.netlist_structures import Module, Net


def plot_routing_debug_image(
    module: Module, net: Net, route_tree: list, occupancy_map: OccupancyMap, cost_estimator: CostEstimator, output_path: str
):
    """
    Generates and saves a debug image for the routing of a single net in grid units.
    """
    log.debug(f"Plotting debug image for net {net.name} to {output_path}")

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot cost heatmap
    grid = np.full((occupancy_map.nx, occupancy_map.ny), 0.0)
    for i in range(occupancy_map.nx):
        for j in range(occupancy_map.ny):
            grid[i, j] = cost_estimator.occupancy_map.grid[i, j]

    # Transpose grid for correct orientation with imshow
    grid = grid.T

    # Use grid indices for extent
    extent = [0, occupancy_map.nx, 0, occupancy_map.ny]
    im = ax.imshow(grid, origin="lower", extent=extent, cmap="viridis_r", interpolation="nearest", alpha=0.5)
    fig.colorbar(im, ax=ax, label="Base Routing Cost")

    # Plot module instances (macros) in grid units
    for inst in module.instances.values():
        if inst.draw.geom:
            if isinstance(inst.draw.geom, Polygon):
                minx, miny, maxx, maxy = inst.draw.geom.bounds
                gminx, gminy = occupancy_map._world_to_grid(minx, miny)
                gmaxx, gmaxy = occupancy_map._world_to_grid(maxx, maxy)
                rect = plt.Rectangle((gminx, gminy), gmaxx - gminx, gmaxy - gminy, fill=True, color="gray", alpha=0.6)
                ax.add_patch(rect)
            else:
                x, y = inst.draw.geom.exterior.xy
                gx, gy = occupancy_map._world_to_grid(x, y)
                ax.fill(gx, gy, alpha=0.6, fc="gray", ec="none")

    # Plot net pins in grid units
    for pin in net.pins.values():
        if pin.draw.geom:
            gx, gy = occupancy_map._world_to_grid(pin.draw.geom.x, pin.draw.geom.y)
            ax.plot(gx, gy, "ro", markersize=5)
            ax.text(gx, gy, pin.name, fontsize=8)

    # Plot route tree in grid units
    for path in route_tree:
        gx = []
        gy = []
        for x, y in path.coords:
            gx_, gy_ = occupancy_map._world_to_grid(x, y)
            gx.append(gx_)
            gy.append(gy_)
        log.info(f"  Route path: {list(zip(gx, gy))}")
        ax.plot(gx, gy, "b-", linewidth=1.5)

    # Set plot limits and labels in grid units
    combined = unary_union(route_tree)
    minx, miny, maxx, maxy = combined.bounds
    gminx, gminy = occupancy_map._world_to_grid(minx, miny)
    gmaxx, gmaxy = occupancy_map._world_to_grid(maxx, maxy)
    width = gmaxx - gminx
    height = gmaxy - gminy
    ax.set_xlim(gminx - width * 0.5, gmaxx + width * 0.5)
    ax.set_ylim(gminy - height * 0.5, gmaxy + height * 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Routing for Net: {net.name} in Module: {module.name}")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the figure
    try:
        plt.savefig(output_path, dpi=300)
    except Exception as e:
        log.error(f"Failed to save debug image to {output_path}: {e}")
    finally:
        plt.close(fig)


def plot_routing_summary(module: Module, nets: list, occupancy_map: OccupancyMap, output_path: str):
    """
    Generates and saves a summary plot showing all routed nets in different colors in grid units.
    """
    log.info(f"Plotting routing summary for module {module.name} to {output_path}")

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot occupancy grid as background
    # Create a grid showing occupancy levels
    occupancy_grid = occupancy_map.grid.T  # Transpose for correct orientation
    extent = [0, occupancy_map.nx, 0, occupancy_map.ny]
    im = ax.imshow(
        occupancy_grid,
        origin="lower",
        extent=extent,
        cmap="Reds",
        alpha=0.3,
        vmin=0,
        vmax=np.max(occupancy_grid) if np.max(occupancy_grid) > 0 else 1,
    )
    fig.colorbar(im, ax=ax, label="Occupancy Level")

    # Plot module instances (macros) as blockages in grid units
    for inst in module.instances.values():
        if inst.draw.geom:
            if isinstance(inst.draw.geom, Polygon):
                minx, miny, maxx, maxy = inst.draw.geom.bounds
                gminx, gminy = occupancy_map._world_to_grid(minx, miny)
                gmaxx, gmaxy = occupancy_map._world_to_grid(maxx, maxy)
                rect = plt.Rectangle(
                    (gminx, gminy), gmaxx - gminx, gmaxy - gminy, fill=True, color="black", alpha=0.8, label="Blockage"
                )
                ax.add_patch(rect)
            else:
                x, y = inst.draw.geom.exterior.xy
                gx, gy = occupancy_map._world_to_grid(x, y)
                ax.fill(gx, gy, alpha=0.8, fc="black", ec="none", label="Blockage")

    # Generate a color map for nets
    import matplotlib.cm as cm

    colors = cm.rainbow(np.linspace(0, 1, len(nets)))

    # Plot all nets with their pins and routes in grid units
    for i, net in enumerate(module.nets.values()):
        color = colors[i]

        # Plot net pins
        for pin in net.pins.values():
            if pin.draw.geom:
                gx, gy = occupancy_map._world_to_grid(pin.draw.geom.x, pin.draw.geom.y)
                ax.plot(gx, gy, "o", color=color, markersize=8, markeredgecolor="black")
                ax.text(gx, gy, f"{net.name}:{pin.name}", fontsize=8, alpha=0.9)

        # Plot net route if it exists
        draw_route(ax, net, color, occupancy_map)

    # Set plot limits based on grid size
    ax.set_xlim(0, occupancy_map.nx)
    ax.set_ylim(0, occupancy_map.ny)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Routing Summary with Occupancy for Module: {module.name}")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Add legend
    ax.legend(loc="upper right", fontsize=8)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info(f"Successfully saved routing summary to {output_path}")
    except Exception as e:
        log.error(f"Failed to save routing summary to {output_path}: {e}")
    finally:
        plt.close(fig)


def draw_route(ax, net, color, occupancy_map):
    geom = net.draw.geom

    if isinstance(geom, list):
        geom = MultiLineString(geom)

    # Handle simple LineString
    if isinstance(geom, LineString):
        gx = []
        gy = []
        for x, y in geom.coords:
            gx_, gy_ = occupancy_map._world_to_grid(x, y)
            gx.append(gx_)
            gy.append(gy_)
        ax.plot(gx, gy, "-", color=color, linewidth=3, label=net.name)

        # Add arrow at midpoint
        if len(gx) > 1:
            mid = len(gx) // 2
            ax.annotate(
                "",
                xy=(gx[mid], gy[mid]),
                xytext=(gx[mid - 1], gy[mid - 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
            )
        return

    # Handle MultiLineString
    if isinstance(geom, MultiLineString):
        for i, seg in enumerate(geom.geoms):
            gx = []
            gy = []
            for x, y in seg.coords:
                gx_, gy_ = occupancy_map._world_to_grid(x, y)
                gx.append(gx_)
                gy.append(gy_)
            # Only label first segment (avoid legend spam)
            label = net.name if i == 0 else None
            ax.plot(gx, gy, "-", color=color, linewidth=3, label=label)

            # Add arrow to each segment
            if len(gx) > 1:
                mid = len(gx) // 2
                ax.annotate(
                    "",
                    xy=(gx[mid], gy[mid]),
                    xytext=(gx[mid - 1], gy[mid - 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2),
                )
        return


def plot_occupancy_summary(module: Module, occupancy_map: OccupancyMap, output_path: str):
    """
    Generates and saves a detailed occupancy map plot showing blockages and congestion in grid units.
    """
    log.info(f"Plotting occupancy summary for module {module.name} to {output_path}")

    fig, ax = plt.subplots(figsize=(16, 12))

    # Create a custom colormap for occupancy levels
    from matplotlib.colors import ListedColormap

    # Define colors: white (0), green (1), yellow (2), orange (3), red (4+)
    colors = ["white", "green", "yellow", "orange", "red"]
    cmap = ListedColormap(colors)

    # Prepare the occupancy grid for plotting
    occupancy_grid = occupancy_map.grid.T  # Transpose for correct orientation

    # Normalize the occupancy levels
    vmin = 0
    vmax = 4

    # Create a grid where values above vmax are clipped
    plot_grid = np.copy(occupancy_grid)
    # Handle blockages (inf) by setting them to a value outside our colormap range
    # We'll plot them separately
    blockage_mask = np.isinf(plot_grid)
    plot_grid[blockage_mask] = 0  # Set blockages to 0 for now, we'll handle them differently
    plot_grid = np.clip(plot_grid, vmin, vmax)

    # Plot occupancy grid using grid indices
    extent = [0, occupancy_map.nx, 0, occupancy_map.ny]
    im = ax.imshow(plot_grid, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Occupancy Level")

    # Plot module instances (macros) as blockages in grid units
    for inst in module.instances.values():
        if inst.draw.geom:
            if isinstance(inst.draw.geom, Polygon):
                minx, miny, maxx, maxy = inst.draw.geom.bounds
                gminx, gminy = occupancy_map._world_to_grid(minx, miny)
                gmaxx, gmaxy = occupancy_map._world_to_grid(maxx, maxy)
                rect = plt.Rectangle(
                    (gminx, gminy), gmaxx - gminx, gmaxy - gminy, fill=False, edgecolor="blue", linewidth=2, label="Blockage"
                )
                ax.add_patch(rect)
            else:
                x, y = inst.draw.geom.exterior.xy
                gx, gy = occupancy_map._world_to_grid(x, y)
                ax.plot(gx, gy, "b-", linewidth=2, label="Blockage")

    # Add grid lines to show individual grid cells
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, occupancy_map.nx, 1))
    ax.set_yticks(np.arange(0, occupancy_map.ny, 1))

    # Highlight over-occupied cells (occupancy >= 2)
    over_occupied_mask = (occupancy_grid >= 2) & (~blockage_mask)
    over_occupied_indices = np.where(over_occupied_mask)
    for ix, iy in zip(over_occupied_indices[0], over_occupied_indices[1]):
        ax.plot(ix, iy, "rx", markersize=4)

    # Set plot limits
    ax.set_xlim(0, occupancy_map.nx)
    ax.set_ylim(0, occupancy_map.ny)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Occupancy Map for Module: {module.name}\n(Blockages in Blue, Over-occupied marked with red X)")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info(f"Successfully saved occupancy summary to {output_path}")
    except Exception as e:
        log.error(f"Failed to save occupancy summary to {output_path}: {e}")
    finally:
        plt.close(fig)
