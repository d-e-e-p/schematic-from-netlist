import heapq
import random
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Router:
    def __init__(self, width, height, cost_settings):
        self.width = width
        self.height = height
        self.layers = 3  # M1 (z=0, horiz), Via (z=1), M2 (z=2, vert)
        self.cost_settings = cost_settings
        self.grid = [[([0] * width) for _ in range(height)] for _ in range(self.layers)]

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def _a_star_search(self, start, end):
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            x, y, z = current
            neighbors = []

            # Metal layer movements
            if z == 0 or z == 2:
                pref_cost = self.cost_settings["pref_dir_cost"]
                wrong_way_cost = self.cost_settings["wrong_way_cost"]
                neighbors.extend(
                    [
                        ((x + 1, y, z), pref_cost if z == 0 else wrong_way_cost),
                        ((x - 1, y, z), pref_cost if z == 0 else wrong_way_cost),
                        ((x, y + 1, z), wrong_way_cost if z == 0 else pref_cost),
                        ((x, y - 1, z), wrong_way_cost if z == 0 else pref_cost),
                    ]
                )

            # Via movements
            via_cost = self.cost_settings["via_cost"]
            if z == 0:
                neighbors.append(((x, y, z + 2), via_cost))
            elif z == 2:
                neighbors.append(((x, y, z - 2), via_cost))

            for neighbor_pos, cost in neighbors:
                nx, ny, nz = neighbor_pos

                if not (0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.layers) or self.grid[nz][ny][nx] == 1:
                    continue

                if abs(z - nz) == 2 and self.grid[1][ny][nx] == 1:
                    continue

                tentative_g_score = g_score[current] + cost
                if neighbor_pos not in g_score or tentative_g_score < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current
                    g_score[neighbor_pos] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor_pos, end)
                    heapq.heappush(open_set, (f_score, neighbor_pos))

        return None

    def route(self, points_to_connect, obstacles):
        # Mark obstacles
        for x, y in obstacles:
            if 0 <= x < self.width and 0 <= y < self.height:
                for z in range(self.layers):
                    self.grid[z][y][x] = 1

        # Steiner Tree Approximation using MST
        G = nx.Graph()
        for i, p1 in enumerate(points_to_connect):
            for j, p2 in enumerate(points_to_connect):
                if i < j:
                    dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                    G.add_edge(p1, p2, weight=dist)

        mst = nx.minimum_spanning_tree(G)
        steiner_length = sum(edge[2]["weight"] for edge in mst.edges(data=True))
        print(f"Estimated Steiner Tree Length: {steiner_length}")

        # Route MST edges
        all_paths = []
        for p1, p2 in mst.edges():
            start_node = (p1[0], p1[1], 0)
            end_node = (p2[0], p2[1], 0)

            path = self._a_star_search(start_node, end_node)
            if path:
                all_paths.append(path)
            else:
                print(f"No path found between {p1} and {p2}")

        return all_paths

    def visualize_routing(self, name, all_paths, points_to_connect, obstacles):
        print("Routing complete.")
        fig, ax = plt.subplots(figsize=(12, 9))

        if obstacles:
            obs_x, obs_y = zip(*obstacles)
            ax.scatter(obs_x, obs_y, c="red", marker="x", s=50, label="Obstacles")

        term_x, term_y = zip(*points_to_connect)
        ax.scatter(term_x, term_y, c="black", marker="s", s=60, label="Terminals", zorder=5)

        for path in all_paths:
            vias, segments_m1, segments_m2 = [], [], []
            current_segment = []
            for point in path:
                if not current_segment or point[2] == current_segment[-1][2]:
                    current_segment.append(point)
                else:
                    (segments_m1 if current_segment[-1][2] == 0 else segments_m2).append(current_segment)
                    vias.append(current_segment[-1])
                    current_segment = [current_segment[-1], point]
            (segments_m1 if current_segment[-1][2] == 0 else segments_m2).append(current_segment)

            for seg in segments_m1:
                seg_arr = np.array(seg)
                ax.plot(seg_arr[:, 0], seg_arr[:, 1], color="blue", linewidth=2, label="Metal1" if "Metal1" not in ax.get_legend_handles_labels()[1] else "")
            for seg in segments_m2:
                seg_arr = np.array(seg)
                ax.plot(seg_arr[:, 0], seg_arr[:, 1], color="green", linewidth=2, label="Metal2" if "Metal2" not in ax.get_legend_handles_labels()[1] else "")
            if vias:
                via_arr = np.array(vias)
                ax.scatter(via_arr[:, 0], via_arr[:, 1], c="purple", marker="o", s=40, label="Vias" if "Vias" not in ax.get_legend_handles_labels()[1] else "", zorder=10)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f'2D Routing Visualization for "{name}" (Steiner MST)')
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")

        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        filename = f"data/images/route_{safe}.png"
        plt.savefig(filename)
        print(f"Saved visualization to {filename}")
        plt.close(fig)


def test_routing():
    default_costs = {"pref_dir_cost": 1, "wrong_way_cost": 2, "via_cost": 5}

    # tiny
    name = "tiny"
    width, height = 10, 10
    points_to_connect = [(0, 0), (9, 9)]
    obstacles = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
    router = Router(width, height, default_costs)
    paths = router.route(points_to_connect, obstacles)
    router.visualize_routing(name, paths, points_to_connect, obstacles)

    # medium
    name = "med"
    width, height = 1500, 1000
    points_to_connect = [(173, 754), (149, 285), (448, 106), (914, 520), (448, 935), (838, 840), (106, 520), (1043, 343), (1043, 697), (838, 201), (512, 524), (1191, 527)]
    obstacles = []
    router = Router(width, height, default_costs)
    paths = router.route(points_to_connect, obstacles)
    router.visualize_routing(name, paths, points_to_connect, obstacles)

    # large_fanout
    name = "large_fanout"
    width, height = 200, 200
    random.seed(42)
    points_to_connect = [(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(20)]
    obstacles = []
    # Add some rectangular blockages
    for _ in range(3):
        x1, y1 = random.randint(0, width - 50), random.randint(0, height - 50)
        x2, y2 = x1 + random.randint(10, 40), y1 + random.randint(10, 40)
        for x in range(x1, x2):
            for y in range(y1, y2):
                obstacles.append((x, y))
    router = Router(width, height, default_costs)
    paths = router.route(points_to_connect, obstacles)
    router.visualize_routing(name, paths, points_to_connect, obstacles)


if __name__ == "__main__":
    test_routing()
