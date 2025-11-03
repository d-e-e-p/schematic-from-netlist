import datetime
import logging as log
import os
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString


def clean_hierarchical_name(name: str, delimiter: str = "_") -> str:
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


def plot_result(net, obstacles, existing_nets, prefix=""):
    """
    Plot the routing result including terminals, obstacles, and paths
    Save to data/images/sastar/ instead of showing on screen
    """
    current_net = net
    net_name = net.name
    terminals = [(pin.draw.geom.x, pin.draw.geom.y) for pin in net.pins.values()]
    path = net.draw.geom
    # Create directory if it doesn't exist
    os.makedirs("data/images/sastar", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate viewport bounds with padding
    padding = 20  # 20 units of padding around the net
    all_points = []

    # Add current net's terminals
    all_points.extend(terminals)

    # Add current net's paths
    if isinstance(path, LineString):
        all_points.extend(path.coords)
    elif isinstance(path, MultiLineString):
        for segment in path.geoms:
            all_points.extend(segment.coords)

    # Plot obstacles
    for i, obstacle in enumerate(obstacles):
        x, y = obstacle.exterior.xy
        ax.fill(x, y, alpha=0.5, fc="gray", ec="black", label="Obstacles" if i == 0 else "")

    # Track which nets we've already plotted
    plotted_nets = set()

    # Plot all existing nets with unique colors
    if existing_nets:
        for i, net in enumerate(existing_nets.values()):
            if path:
                color = plt.cm.tab20(i % 20)  # Use a colormap with more distinct colors
                if isinstance(path, LineString):
                    x, y = path.xy
                    # Only add label once per net
                    label = f"{net.name} (cost: {sum(net.draw.step_costs.values()):.1f})" if net.name not in plotted_nets else ""
                    ax.plot(x, y, color=color, linewidth=2, label=label)
                    if label:
                        plotted_nets.add(net.name)
                    # Add points to viewport calculation
                    all_points.extend(path.coords)
                elif isinstance(path, MultiLineString):
                    for segment in path.geoms:
                        x, y = segment.xy
                        # Only add label once per net
                        label = (
                            f"{net.name} (cost: {sum(net.draw.step_costs.values()):.1f})" if net.name not in plotted_nets else ""
                        )
                        ax.plot(x, y, color=color, linewidth=2, label=label)
                        if label:
                            plotted_nets.add(net.name)
                        # Add points to viewport calculation
                        all_points.extend(segment.coords)

    # Find the current net object to get its cost
    current_net_cost = ""
    if net.name and existing_nets:
        net = existing_nets[net.name]
        current_net_cost = f"total cost: {net.draw.total_cost}"

    # Plot current paths in blue
    i = 0
    if isinstance(path, LineString):
        x, y = path.xy
        ax.plot(x, y, "b-", linewidth=2, label=f"Current: {net_name}{current_net_cost}" if i == 0 else "")
    elif isinstance(path, MultiLineString):
        for j, segment in enumerate(path.geoms):
            x, y = segment.xy
            ax.plot(x, y, "b-", linewidth=2, label=f"Current: {net_name}{current_net_cost}" if i == 0 and j == 0 else "")

    # Plot current paths with step costs
    if current_net.draw.step_costs:
        # Plot all stored step costs directly
        for midpoint, cost in current_net.draw.step_costs.items():
            ax.text(
                midpoint[0],
                midpoint[1],
                f"{cost:.0f}",
                fontsize=6,
                color="red",
                ha="center",
                va="center",
            )

    # Plot terminals
    for i, terminal in enumerate(terminals):
        ax.plot(terminal[0], terminal[1], "ro", markersize=10, label="Terminals" if i == 0 else "")
        ax.text(terminal[0], terminal[1], f"T{i}", fontsize=12, ha="center", va="bottom")

    # Set viewport limits if we have points
    if all_points:
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        xmin, xmax = min(xs) - padding, max(xs) + padding
        ymin, ymax = min(ys) - padding, max(ys) + padding

        # Set viewport limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        # Fallback to entire design bounds if no points
        ax.autoscale()

    ax.set_aspect("equal")
    ax.grid(True)

    # Create a title with net name and cost
    if net_name:
        title = f"Routing Result - {net_name}{current_net_cost}"
    else:
        title = "Routing Result"
    ax.set_title(title, fontsize=14)

    # Add legend if we have labels
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")

    # Generate filename with cleaned net name
    filename = f"data/images/sastar/route_{prefix}_{clean_hierarchical_name(net_name)}.png"

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    log.info(f"Saved plot to {filename}")
