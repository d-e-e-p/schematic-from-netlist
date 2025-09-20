import os

import matplotlib.pyplot as plt
import networkx as nx


class Pathfinder:
    def __init__(self, db, schematic_db):
        self.db = db
        self.schematic_db = schematic_db
        self.G = nx.grid_2d_graph(0, 0)

    def modify_line_blockages(self, create_blockage_questionmark, pt_start, pt_end):
        # Scale to grid coordinates
        x0, y0 = (pt_start[0], pt_start[1])
        x1, y1 = (pt_end[0], pt_end[1])

        def bresenham(x0, y0, x1, y1):
            """Yield integer grid points along a line from (x0, y0) to (x1, y1)."""
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                yield x0, y0
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        # Get obstacle points along the line and remove them from the graph
        blockages = list(bresenham(x0, y0, x1, y1))
        if create_blockage_questionmark:
            self.G.add_nodes_from(blockages)
        else:
            self.G.remove_nodes_from(blockages)
            pass

    def modify_wire_blockages(self, create_blockage_questionmark, wire_shape):
        pt_start = wire_shape.points[0]
        for pt in wire_shape.points[1:]:
            pt_end = pt
            self.modify_line_blockages(create_blockage_questionmark, pt_start, pt_end)
            pt_start = pt_end

    def create_inst_blockages(self, inst_shape):
        """don't route over blocks"""

        ll_x, ll_y, ur_x, ur_y = inst_shape.rect
        obstacles = [(x, y) for x in range(ll_x, ur_x + 1) for y in range(ll_y, ur_y + 1)]
        self.G.remove_nodes_from(obstacles)

        pin_access_points = []
        for shape in inst_shape.port_shapes:
            x = shape.point[0]
            y = shape.point[1]
            pin_access_points.append((x, y))
        self.G.add_nodes_from(pin_access_points)

    def find_pins_of_net(self, net):
        endpoints = []
        for pin in net.pins:
            for inst_shape in self.schematic_db.inst_shapes:
                for port_shape in inst_shape.port_shapes:
                    if pin.name == port_shape.pin.name:
                        x = port_shape.point[0]
                        y = port_shape.point[1]
                        endpoints.append((x, y))
        # self.G.remove_nodes_from(endpoints)
        return endpoints

    def reroute_net(self, net, points_to_connect):
        # ---  Compute Pairwise Shortest Paths ---
        # Create a complete graph H where nodes are the points_to_connect
        # and edge weights are the shortest path distances in the grid graph G.
        # compute obstructions instead of cache
        width, height = self.schematic_db.sheet_size
        original_nodes = set(nx.grid_2d_graph(width, height).nodes())
        current_nodes = set(self.G.nodes())
        obstacles = original_nodes - current_nodes
        print(f"{points_to_connect=}")
        # print(f"{obstacles=}")

        self.G.add_nodes_from(points_to_connect)
        H = nx.Graph()
        for i in range(len(points_to_connect)):
            for j in range(i + 1, len(points_to_connect)):
                start_node = points_to_connect[i]
                end_node = points_to_connect[j]
                try:
                    # Use Dijkstra's algorithm for shortest path length
                    distance = nx.shortest_path_length(self.G, start_node, end_node, weight="weight")
                    H.add_edge(start_node, end_node, weight=distance)
                except nx.NetworkXNoPath:
                    print(f"Warning: No path found between {start_node} and {end_node}. The points might not be connectable.")
                    # Add edge with infinite weight if no path exists
                    H.add_edge(start_node, end_node, weight=float("inf"))

        # --- 4. Find the Minimum Spanning Tree (MST) ---
        # The MST on H gives the cheapest set of connections between our points.
        mst = nx.minimum_spanning_tree(H)

        # --- 5. Reconstruct the Full Path ---
        # Create a new graph to hold the final combined path.
        full_path_graph = nx.Graph()
        for u, v in mst.edges():
            # For each edge in the MST, find the actual shortest path in the grid graph G
            try:
                path = nx.shortest_path(self.G, u, v, weight="weight")
                # Add this path to our final graph
                nx.add_path(full_path_graph, path)
            except nx.NetworkXNoPath:
                print(f"Warning: fullpath: No path found between {u} and {v}.")

        # --- 6. Visualize the Grid and Route ---
        plt.figure(figsize=(12, 12))

        G_drawing = nx.grid_2d_graph(width, height)
        pos = {node: node for node in G_drawing.nodes()}
        # Draw all grid points from the drawing graph
        nx.draw_networkx_nodes(G_drawing, pos, node_size=50, node_color="lightgray")
        # Draw the obstacles
        nx.draw_networkx_nodes(G_drawing, pos, nodelist=obstacles, node_color="red", node_size=200, node_shape="s")
        # Draw the points to connect
        nx.draw_networkx_nodes(G_drawing, pos, nodelist=points_to_connect, node_color="green", node_size=300)

        # Draw the final route
        nx.draw(full_path_graph, pos, with_labels=False, node_size=100, node_color="cyan", width=2.5, edge_color="blue")

        # Add labels for points to connect for clarity
        labels = {node: f"({node[0]},{node[1]})" for node in points_to_connect}
        nx.draw_networkx_labels(G_drawing, pos, labels=labels, font_size=8, verticalalignment="bottom")

        plt.title(f"Pathfinding for {net.name} ")
        plt.axis("on")
        plt.grid(True)
        plt.savefig(f"data/images/reoute_{net.name}.png")
        plt.close()
        print("just stop here for now")
        exit()

    def cleanup_routes(self):
        width, height = self.schematic_db.sheet_size
        print(f"Grid {width=} X {height=} ")
        self.G = nx.grid_2d_graph(width, height)
        print(f"Grid {width=} X {height=} = {self.G.size()=}")

        for inst_shape in self.schematic_db.inst_shapes:
            self.create_inst_blockages(inst_shape)

        # Process wires
        netname2shape = {}
        for wire_shape in self.schematic_db.net_shapes:
            netname2shape[wire_shape.name] = wire_shape
            # self.modify_wire_blockages(create_blockage_questionmark=True, wire_shape=wire_shape)

        sorted_nets = sorted(self.db.nets_by_name.values(), key=lambda net: net.num_conn, reverse=True)
        for net in sorted_nets:
            if net.name in netname2shape:
                wire_shape = netname2shape[net.name]
                self.modify_wire_blockages(create_blockage_questionmark=False, wire_shape=wire_shape)
                endpoints = self.find_pins_of_net(net)
                self.reroute_net(net, endpoints)
