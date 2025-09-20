import os

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx


class Pathfinder:
    def __init__(self, db, schematic_db):
        self.db = db
        self.schematic_db = schematic_db
        self.G = nx.grid_2d_graph(0, 0)
        self.obstacles = set()

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
            self.obstacles.update(blockages)
        else:
            self.obstacles.difference_update(blockages)

    def modify_wire_blockages(self, create_blockage_questionmark, wire_shape):
        pt_start = wire_shape.points[0]
        for pt in wire_shape.points[1:]:
            pt_end = pt
            self.modify_line_blockages(create_blockage_questionmark, pt_start, pt_end)
            pt_start = pt_end

    def create_inst_blockages(self, inst_shape):
        """don't route over blocks"""

        ll_x, ll_y, ur_x, ur_y = inst_shape.rect
        blockages = [(x, y) for x in range(ll_x, ur_x + 1) for y in range(ll_y, ur_y + 1)]
        self.obstacles.update(blockages)

        # but make sure pins can be accessed
        pin_access_points = []
        for shape in inst_shape.port_shapes:
            x = shape.point[0]
            y = shape.point[1]
            pin_access_points.append((x, y))

        self.obstacles.difference_update(pin_access_points)

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
        # ---  Compute Pairwise Shortest Paths using igraph ---
        width, height = self.schematic_db.sheet_size
        original_nodes = set(nx.grid_2d_graph(width, height).nodes())

        # Add points to connect to the graph
        self.G.add_nodes_from(points_to_connect)

        # Ensure all points to connect are in the graph
        for p in points_to_connect:
            if p not in self.G:
                print(f"Warning: Point {p} for net {net.name} is not in the graph. Skipping this net.")
                return

        # Filter out points that are not in the graph (i.e., they are obstacles)
        points_to_connect = [p for p in points_to_connect if p in self.G]

        if len(points_to_connect) < 2:
            print(f"Warning: Not enough connectable points for net {net.name}")
            return

        current_nodes = set(self.G.nodes())
        obstacles = original_nodes - current_nodes
        print(f"{points_to_connect=}")
        # print(f"Nodes in graph: {list(self.G.nodes())}")

        # Convert networkx graph to igraph graph for performance
        nodes = list(self.G.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        reverse_node_map = {i: node for i, node in enumerate(nodes)}

        ig_edges = [(node_map[u], node_map[v]) for u, v in self.G.edges()]
        g = ig.Graph(ig_edges, directed=False)
        # g.vs["name"] = nodes

        # Find the largest connected component
        components = g.connected_components()
        if not components:
            print(f"Warning: No connected components found for net {net.name}")
            return
        largest_component = max(components, key=len)
        # largest_component_nodes = {list(g.vs)[i]["name"] for i in largest_component}

        # Filter points to connect to only include those in the largest component
        points_to_connect_ids = [node_map[p] for p in points_to_connect if node_map[p] in largest_component_nodes]

        if len(points_to_connect_ids) < 2:
            print(f"Warning: Not enough connectable points in the largest component for net {net.name}")
            return

        # Create a complete graph H where nodes are the points_to_connect
        # and edge weights are the shortest path distances in the grid graph g.
        weights = g.shortest_paths(source=points_to_connect_ids, target=points_to_connect_ids)

        # Create the complete graph H using igraph
        num_points = len(points_to_connect_ids)
        H = ig.Graph(n=num_points, directed=False)
        h_edges = []
        h_weights = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                h_edges.append((i, j))
                h_weights.append(weights[i][j])

        H.add_edges(h_edges)
        H.es["weight"] = h_weights

        # --- Find the Minimum Spanning Tree (MST) ---
        mst = H.spanning_tree(weights=H.es["weight"])

        # --- Reconstruct the Full Path ---
        full_path_graph = nx.Graph()

        # Map indices in H back to vertex IDs in g
        h_idx_to_g_id = {i: points_to_connect_ids[i] for i in range(num_points)}

        for edge in mst.es:
            h_source_idx, h_target_idx = edge.tuple

            grid_source_id = h_idx_to_g_id[h_source_idx]
            grid_target_id = h_idx_to_g_id[h_target_idx]

            try:
                path_ids = g.get_shortest_path(grid_source_id, grid_target_id)
                path_coords = [reverse_node_map[pid] for pid in path_ids]
                nx.add_path(full_path_graph, path_coords)
            except Exception as e:  # igraph raises generic exception for no path
                # Find original nodes for error message
                u = reverse_node_map[grid_source_id]
                v = reverse_node_map[grid_target_id]
                print(f"Warning: fullpath: No path found between {u} and {v}. Error: {e}")

        # --- 6. Visualize the Grid and Route ---
        plt.figure(figsize=(12, 12))

        G_drawing = nx.grid_2d_graph(width, height)
        pos = {node: node for node in G_drawing.nodes()}
        # Draw all grid points from the drawing graph
        nx.draw_networkx_nodes(G_drawing, pos, node_size=50, node_color="lightgray")
        # Draw the obstacles
        nx.draw_networkx_nodes(G_drawing, pos, nodelist=list(obstacles), node_color="red", node_size=200, node_shape="s")
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
