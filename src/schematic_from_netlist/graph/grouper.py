import math
import random
import time

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np


class Grouper:
    def __init__(self):
        pass

    def group_endpoints(self, points_to_connect, threshold_multiplier=1.5, cost_function="manhattan"):
        num_points = len(points_to_connect)
        if num_points < 2:
            return [points_to_connect], [0] if points_to_connect else []

        # Create a complete graph in igraph
        g = ig.Graph(num_points)
        weights = []
        edges = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                p1 = points_to_connect[i]
                p2 = points_to_connect[j]
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

                if cost_function == "log_manhattan":
                    weight = math.log(dist + 1)
                else:  # Default to manhattan
                    weight = dist

                edges.append((i, j))
                weights.append(weight)

        g.add_edges(edges)
        g.es["weight"] = weights

        # Compute the Minimum Spanning Tree
        mst = g.spanning_tree(weights=g.es["weight"])

        # Remove long edges to form clusters
        edge_weights = mst.es["weight"]
        if not edge_weights:
            clusters = mst.connected_components(mode="weak")
            groups_indices = [list(cluster) for cluster in clusters]
        else:
            avg_weight = sum(edge_weights) / len(edge_weights)
            threshold = threshold_multiplier * avg_weight

            edges_to_remove_indices = [edge.index for edge in mst.es if edge["weight"] > threshold]
            mst.delete_edges(edges_to_remove_indices)

            # Get the connected components (clusters)
            clusters = mst.connected_components(mode="weak")
            groups_indices = [list(cluster) for cluster in clusters]

        groups = [[points_to_connect[i] for i in group_idx] for group_idx in groups_indices]

        print(f"Formed {len(groups)} groups.")

        if len(groups) <= 1:
            return groups, list(range(len(groups)))

        # Calculate centroids
        centroids = []
        for group in groups:
            x_coords = [p[0] for p in group]
            y_coords = [p[1] for p in group]
            centroids.append((sum(x_coords) / len(group), sum(y_coords) / len(group)))

        # Build a graph of group centroids
        group_graph = ig.Graph(len(centroids))
        group_weights = []
        group_edges = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
                group_edges.append((i, j))
                group_weights.append(dist)
        group_graph.add_edges(group_edges)
        group_graph.es["weight"] = group_weights

        # Find MST of the group graph
        group_mst = group_graph.spanning_tree(weights=group_graph.es["weight"])

        if len(group_mst.vs) > 0:
            degrees = group_mst.degree()
            start_node = np.argmax(degrees)
            ordering = group_mst.dfs(vid=start_node)[0]
        else:
            ordering = []

        # Reorder the groups themselves to match the ordering
        ordered_groups = [groups[i] for i in ordering]

        return ordered_groups, ordering


def visualize_groups(name, groups, points_to_connect, obstacles, ordering):
    print(f"Visualizing endpoint groups for '{name}'.")
    fig, ax = plt.subplots(figsize=(12, 9))

    if obstacles:
        obs_x, obs_y = zip(*obstacles)
        ax.scatter(obs_x, obs_y, c="red", marker="x", s=50, label="Obstacles")

    colors = plt.cm.get_cmap("tab10", len(groups))

    for i, group in enumerate(groups):
        if not group:
            continue
        group_x, group_y = zip(*group)
        original_group_index = ordering[i]
        ax.scatter(group_x, group_y, color=colors(i), marker="o", s=60, label=f"Group {original_group_index + 1}", zorder=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f'Endpoint Grouping for "{name}"')
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")

    filename = f"route_{name}_groups.png"
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    plt.close(fig)


def run_test(name, points_to_connect, obstacles, threshold_multiplier=1.5, cost_function="manhattan"):
    print(f"\n--- Running Test Case: {name} (multiplier: {threshold_multiplier}, cost: {cost_function}) ---")
    grouper = Grouper()
    start_time = time.time()
    groups, ordering = grouper.group_endpoints(points_to_connect, threshold_multiplier=threshold_multiplier, cost_function=cost_function)
    ig_time = time.time() - start_time

    print(f"Optimal group ordering (0-indexed): {ordering}")
    print(f"igraph implementation took {ig_time:.6f} seconds.")

    visualize_groups(name, groups, points_to_connect, obstacles, ordering)


def test_four_clusters():
    points = [(25, 25), (35, 35), (30, 28), (165, 25), (175, 35), (170, 28), (25, 165), (35, 175), (30, 168), (165, 165), (175, 175), (170, 168)]
    run_test("four_clusters", points, [])


def test_four_quadrants_as_one():
    points = [(25, 25), (35, 35), (30, 28), (165, 25), (175, 35), (170, 28), (25, 165), (35, 175), (30, 168), (165, 165), (175, 175), (170, 168)]
    run_test("four_quadrants_as_one", points, [], cost_function="log_manhattan", threshold_multiplier=3.0)


def test_clustered():
    random.seed(42)
    points = []
    # Create a tight cluster of 20 points
    for _ in range(20):
        points.append((random.randint(20, 50), random.randint(20, 50)))
    run_test("clustered", points, [], threshold_multiplier=0.8)


def test_distributed():
    points = []
    # Create a grid of 20 points
    for i in range(4):
        for j in range(5):
            points.append((20 + i * 40, 20 + j * 40))
    run_test("distributed", points, [], threshold_multiplier=0.8)


if __name__ == "__main__":
    test_four_clusters()
    test_four_quadrants_as_one()
    test_clustered()
    test_distributed()

