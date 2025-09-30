import heapq
import logging as log
import math
import os

import networkx as nx
import shapely
from matplotlib import pyplot as plt
from rtree import index
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import linemerge, unary_union

log.getLogger("matplotlib").setLevel(log.WARNING)


class LayoutOptimizer:
    def __init__(self, db):
        self.db = db

    def _calculate_best_orientation(self, old_pins, macro_pins_local, centroid):
        orientations = {"R0": 0, "R90": 90, "R180": 180, "R270": 270}
        min_dist = float("inf")
        best_orient = None
        best_pin_map = None
        best_rotated_pins_local = None

        for orient_name, angle in orientations.items():
            rotated_new_pins_local = [rotate(p, angle, origin="center") for p in macro_pins_local]
            new_pins_global_centered = [Point(centroid.x + p.x, centroid.y + p.y) for p in rotated_new_pins_local]

            dist1 = old_pins[0].distance(new_pins_global_centered[0]) + old_pins[1].distance(new_pins_global_centered[1])
            dist2 = old_pins[0].distance(new_pins_global_centered[1]) + old_pins[1].distance(new_pins_global_centered[0])

            current_dist, current_pin_map = (dist1, {0: 0, 1: 1}) if dist1 < dist2 else (dist2, {0: 1, 1: 0})

            if current_dist < min_dist:
                min_dist = current_dist
                best_orient = orient_name
                best_pin_map = current_pin_map
                best_rotated_pins_local = rotated_new_pins_local

        return best_orient, best_pin_map, best_rotated_pins_local

    def adjust_location(self, inst):
        """
        Place and orient a new macro shape to minimize distance to existing pins.
        """
        log.info(f"Optimizing instance {inst.name}")
        log.info(f"  Initial geom: {inst.geom}, orient: {inst.orient}")
        for name, pin in inst.pins.items():
            log.info(f"  Pin {name}: {pin.geom}")

        old_geom = inst.geom
        old_pins = [p.geom for p in inst.pins.values()]
        pin_names = list(inst.pins.keys())
        centroid = old_geom.centroid

        macro_box = box(-1, -3, 1, 3)
        macro_pins_local = [Point(0, 3), Point(0, -3)]

        best_orient, best_pin_map, best_rotated_pins_local = self._calculate_best_orientation(old_pins, macro_pins_local, centroid)

        p1_geom, p2_geom = old_pins[0], old_pins[1]
        np_local1 = best_rotated_pins_local[best_pin_map[0]]
        np_local2 = best_rotated_pins_local[best_pin_map[1]]

        npgc1 = Point(centroid.x + np_local1.x, centroid.y + np_local1.y)
        npgc2 = Point(centroid.x + np_local2.x, centroid.y + np_local2.y)

        dx, dy = 0, 0
        if best_orient in ["R0", "R180"]:
            dy = (p1_geom.y + p2_geom.y - (npgc1.y + npgc2.y)) / 2
            dy = max(-1.5, min(1.5, dy))
        else:  # R90, R270
            dx = (p1_geom.x + p2_geom.x - (npgc1.x + npgc2.x)) / 2
            dx = max(-1.5, min(1.5, dx))

        final_translation_x = round(centroid.x + dx)
        final_translation_y = round(centroid.y + dy)

        orientations = {"R0": 0, "R90": 90, "R180": 180, "R270": 270}
        angle = orientations[best_orient]
        new_geom = translate(rotate(macro_box, angle, origin="center"), xoff=final_translation_x, yoff=final_translation_y)

        inst.geom = new_geom
        inst.orient = best_orient

        new_pin_geoms = [
            Point(
                final_translation_x + best_rotated_pins_local[best_pin_map[0]].x,
                final_translation_y + best_rotated_pins_local[best_pin_map[0]].y,
            ),
            Point(
                final_translation_x + best_rotated_pins_local[best_pin_map[1]].x,
                final_translation_y + best_rotated_pins_local[best_pin_map[1]].y,
            ),
        ]

        for i, pin_name in enumerate(pin_names):
            pin = inst.pins[pin_name]
            old_pin_geom = old_pins[i]
            new_pin_geom = new_pin_geoms[i]

            if pin.net:
                log.info(
                    f"  Net {pin.net.name} updated: {pin.net.geom} with new_segment={LineString([old_pin_geom, new_pin_geom])}"
                )
                new_segment = LineString([old_pin_geom, new_pin_geom])
                if pin.net.geom and hasattr(pin.net.geom, "geoms"):
                    existing_lines = list(pin.net.geom.geoms)
                else:
                    existing_lines = []
                pin.net.geom = MultiLineString(existing_lines + [new_segment])

            pin.geom = new_pin_geom

        log.info(f"  Final geom: {inst.geom}, orient: {inst.orient}")
        for name, pin in inst.pins.items():
            log.info(f"  Pin {name}: {pin.geom}")

    def _plot_net_geometry(self, net, geom, stage, old_geom=None, all_macros=None):
        fig, ax = plt.subplots()
        ax.set_title(f"Net: {net.name} - Stage: {stage}")
        log.info(f"Plotting net {net.name} - Stage: {stage}")

        if all_macros:
            for macro in all_macros:
                if macro and not macro.is_empty:
                    x, y = macro.exterior.xy
                    ax.fill(x, y, alpha=0.3, fc="gray", ec="black")

        if old_geom and not old_geom.is_empty:
            if hasattr(old_geom, "geoms"):
                for line in old_geom.geoms:
                    x, y = line.xy
                    ax.plot(x, y, "k-", alpha=0.3)
            elif isinstance(old_geom, LineString):
                x, y = old_geom.xy
                ax.plot(x, y, "k-", alpha=0.3)

        if geom and not geom.is_empty:
            if hasattr(geom, "geoms"):
                for line in geom.geoms:
                    x, y = line.xy
                    ax.plot(x, y, "b-")
            elif isinstance(geom, LineString):
                x, y = geom.xy
                ax.plot(x, y, "b-")

        pin_coords = [tuple(p.geom.coords[0]) for p in net.pins]
        if pin_coords:
            x, y = zip(*pin_coords)
            ax.plot(x, y, "ro", markersize=5)

        ax.set_aspect("equal", "datalim")
        output_dir = "data/images/beautify_plots"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"net_{net.name.replace('/', '_')}_{stage}.png")
        plt.savefig(filename)
        plt.close(fig)
        log.info(f"  Saved plot: {filename}")
        if geom and hasattr(geom, "geoms"):
            log.info(f"  Net {net.name} geom: {[line for line in geom.geoms]}")

    def _snap_to_grid(self, geom):
        if geom is None or geom.is_empty:
            return geom

        def snap_coords(coords):
            return [(round(x), round(y)) for x, y in coords]

        if isinstance(geom, Point):
            return Point(snap_coords(geom.coords)[0])
        if isinstance(geom, LineString):
            return LineString(snap_coords(geom.coords))
        if isinstance(geom, MultiLineString):
            return MultiLineString([LineString(snap_coords(line.coords)) for line in geom.geoms])
        return geom

    def beautify_routes(self):
        log.info("Starting route beautification...")
        all_macros = {inst.name: inst.geom for inst in self.db.top_module.get_all_instances().values()}
        clearance = 2.0
        routed_net_geoms = []
        epsilon = -1e-9  # A small negative buffer to shrink macros slightly

        # Sort nets by name for deterministic routing order
        nets_to_route = sorted(self.db.top_module.nets.values(), key=lambda n: n.num_conn)
        # nets_to_route = [self.db.top_module.nets["Net__U1_PA10_A2_D2_"]]

        for net in nets_to_route:
            if not net.pins or len(net.pins) < 2:
                continue

            log.info(f"Beautifying net {net.name}")
            self._plot_net_geometry(net, net.geom, "initial", all_macros=all_macros.values())

            for pin in net.pins:
                pin.geom = self._snap_to_grid(pin.geom)

            # --- Obstacle Calculation ---
            # Find the macros connected to the current net's pins
            current_pin_macros = {self._get_pin_macro(p) for p in net.pins if self._get_pin_macro(p)}

            obstacles = []
            # Add all other macros, buffered by clearance, as obstacles
            for macro in all_macros.values():
                if macro and macro not in current_pin_macros:
                    obstacles.append(macro.buffer(clearance))

            # Add the current net's own macros, but shrunken slightly.
            # This allows the router to connect to pins on the boundary but not route through the macro.
            for macro in current_pin_macros:
                obstacles.append(macro.buffer(epsilon))

            # Add all previously routed nets, buffered by clearance, as obstacles
            if routed_net_geoms:
                obstacles.append(unary_union(routed_net_geoms).buffer(clearance))

            obstacle_union = unary_union(obstacles) if obstacles else None
            # --- End Obstacle Calculation ---

            log.info(f"astar b {net.name}")
            new_geom = self.reroute_net_with_astar(net, obstacle_union)
            log.info(f"astar c {net.name}")

            if new_geom and not new_geom.is_empty:
                merged_geom = self.merge_collinear_segments(new_geom)
                if isinstance(merged_geom, LineString):
                    merged_geom = MultiLineString([merged_geom])

                self._plot_net_geometry(net, merged_geom, "after_astar", old_geom=net.geom, all_macros=all_macros.values())
                self._check_connectivity(net, merged_geom)
                net.geom = merged_geom
                routed_net_geoms.append(merged_geom)  # Add to obstacles for the next net
            else:
                log.warning(f"  Routing failed for net {net.name}, keeping original geometry.")
                self._plot_net_geometry(net, net.geom, "final_failed", all_macros=all_macros.values())

    def reroute_net_with_astar(self, net, obstacle_union):
        pins = list(net.pins)
        pin_geoms = [p.geom for p in pins]
        if len(pin_geoms) < 2:
            return None

        # Build a Minimum Spanning Tree (MST) to define the primary connection topology
        pin_graph = nx.Graph()
        for i in range(len(pin_geoms)):
            for j in range(i + 1, len(pin_geoms)):
                p1 = pin_geoms[i]
                p2 = pin_geoms[j]
                dist = abs(p1.x - p2.x) + abs(p1.y - p2.y)
                pin_graph.add_edge(i, j, weight=dist)
        mst = nx.minimum_spanning_tree(pin_graph)

        # --- Pass 1: Route all MST edges independently ---
        routed_segments = []
        for u, v in mst.edges():
            start_pin = pin_geoms[u]
            end_pin = pin_geoms[v]
            path = self._astar_path(start_pin, end_pin, obstacle_union)
            if len(path) > 1:
                routed_segments.append(LineString(path))

        if not routed_segments:
            log.warning(f"    Initial A* routing failed for all segments in net {net.name}")
            return None

        # --- Pass 2: Connect any disconnected sub-trees ---
        final_route_geom = unary_union(routed_segments)

        g = nx.Graph()
        lines = [final_route_geom] if isinstance(final_route_geom, LineString) else list(final_route_geom.geoms)
        for line in lines:
            for i in range(len(line.coords) - 1):
                g.add_edge(line.coords[i], line.coords[i + 1])

        components = list(nx.connected_components(g))

        if len(components) > 1:
            log.info(f"    Net {net.name} requires stitching. Found {len(components)} disconnected components.")
            main_component_geom = unary_union([LineString(list(c)) for c in components[:-1]])

            stitch_lines = []
            last_component_geom = LineString(list(components[-1]))

            path = self._astar_path_to_geometry(Point(last_component_geom.coords[0]), main_component_geom, obstacle_union)
            if path:
                stitch_lines.append(LineString(path))

            final_route_geom = unary_union([final_route_geom] + stitch_lines)

        return final_route_geom

    def _astar_path(self, start_point, end_point, obstacle_union):
        start = (round(start_point.x), round(start_point.y))
        end = (round(end_point.x), round(end_point.y))
        log.info(f"  begin A* from {start} to {end}")

        pq = [(0, start, [start], (0, 0))]
        # Visited needs to store the path to prevent cycles that are longer
        # but arrive at the same point. Using g_cost for this.
        visited = {}

        def is_obstacle(p_tuple):
            if p_tuple == start or p_tuple == end:
                return False
            if obstacle_union is None or obstacle_union.is_empty:
                return False
            return obstacle_union.contains(Point(p_tuple))

        while pq:
            f_cost, current, path, direction = heapq.heappop(pq)
            log.info(f"    Visiting {current} with path {path} and direction {direction}")

            g_cost = len(path) - 1

            # If we have found a better or equal-length path to this node before, skip.
            if visited.get(current, float("inf")) <= g_cost:
                continue
            visited[current] = g_cost

            if current == end:
                log.info(f"  A* path found from {start} to {end}")
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_direction = (dx, dy)
                neighbor = (current[0] + dx, current[1] + dy)

                if is_obstacle(neighbor):
                    continue

                new_g_cost = g_cost + 1

                # Check if the neighbor has been visited with a shorter or equal path
                if visited.get(neighbor, float("inf")) <= new_g_cost:
                    continue

                turn_penalty = 5 if direction != (0, 0) and new_direction != direction else 0
                h_cost = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                new_f_cost = new_g_cost + h_cost + turn_penalty

                new_path = path + [neighbor]
                heapq.heappush(pq, (new_f_cost, neighbor, new_path, new_direction))

        log.warning(f"  A* failed to find a path from {start} to {end}. Falling back to diagonal.")
        return [start_point.coords[0], end_point.coords[0]]

    def _astar_path_to_geometry(self, start_point, target_geom, obstacle_union):
        start = (round(start_point.x), round(start_point.y))

        # Create a set of all points on the target geometry for a quick lookup
        target_points = set()
        lines = (
            [target_geom]
            if isinstance(target_geom, LineString)
            else (list(target_geom.geoms) if hasattr(target_geom, "geoms") else [])
        )
        for line in lines:
            for i in range(len(line.coords) - 1):
                p1 = line.coords[i]
                p2 = line.coords[i + 1]
                # Bresenham's line algorithm to get all integer points on the segment
                x1, y1 = round(p1[0]), round(p1[1])
                x2, y2 = round(p2[0]), round(p2[1])
                dx = abs(x2 - x1)
                sx = 1 if x1 < x2 else -1
                dy = -abs(y2 - y1)
                sy = 1 if y1 < y2 else -1
                err = dx + dy
                while True:
                    target_points.add((x1, y1))
                    if x1 == x2 and y1 == y2:
                        break
                    e2 = 2 * err
                    if e2 >= dy:
                        err += dy
                        x1 += sx
                    if e2 <= dx:
                        err += dx
                        y1 += sy

        if not target_points:  # Handle case where target_geom is a Point
            if isinstance(target_geom, Point):
                target_points.add((round(target_geom.x), round(target_geom.y)))
            else:
                log.warning("A* target geometry is empty or invalid.")
                return None

        pq = [(0, start, [start], (0, 0))]
        visited = {}

        def is_obstacle(p_tuple):
            if p_tuple == start:
                return False
            if obstacle_union is None or obstacle_union.is_empty:
                return False
            return obstacle_union.contains(Point(p_tuple))

        while pq:
            f_cost, current, path, direction = heapq.heappop(pq)

            if current in target_points:
                log.info(f"  A* path found from {start} to target geometry")
                return path

            g_cost = len(path) - 1

            if visited.get((current, direction), float("inf")) <= g_cost:
                continue
            visited[(current, direction)] = g_cost

            # Find the closest point in target_points to estimate heuristic
            closest_target = min(target_points, key=lambda p: abs(current[0] - p[0]) + abs(current[1] - p[1]))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_direction = (dx, dy)
                neighbor = (current[0] + dx, current[1] + dy)

                if is_obstacle(neighbor):
                    continue

                new_g_cost = g_cost + 1
                turn_penalty = 5 if direction != (0, 0) and new_direction != direction else 0
                h_cost = abs(neighbor[0] - closest_target[0]) + abs(neighbor[1] - closest_target[1])
                new_f_cost = new_g_cost + h_cost + turn_penalty

                new_path = path + [neighbor]
                heapq.heappush(pq, (new_f_cost, neighbor, new_path, new_direction))

        log.warning(f"  A* failed to find a path from {start} to the target geometry.")
        return None

    def _decompose_into_segments(self, geom):
        if geom is None or geom.is_empty:
            return MultiLineString()

        segments = []
        lines_to_process = []
        if isinstance(geom, LineString):
            lines_to_process = [geom]
        elif hasattr(geom, "geoms"):
            lines_to_process = list(geom.geoms)

        for line in lines_to_process:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                segments.append(LineString([coords[i], coords[i + 1]]))

        return MultiLineString(segments)

    def _astar_path(self, start_point, end_point, obstacle_union):
        start = (round(start_point.x), round(start_point.y))
        end = (round(end_point.x), round(end_point.y))

        pq = [(0, start, [start], (0, 0))]
        visited = {}

        def is_obstacle(p_tuple):
            if p_tuple == start or p_tuple == end:
                return False
            if obstacle_union is None or obstacle_union.is_empty:
                return False
            return obstacle_union.contains(Point(p_tuple))

        MAX_EXPANSIONS = 10000
        expansions = 0
        while pq:
            expansions += 1
            if expansions > MAX_EXPANSIONS:
                log.warning(f"A* search aborted after {expansions} expansions — likely no path.")
                break
            f_cost, current, path, direction = heapq.heappop(pq)
            log.info(f"    Visiting {current} with path {path} and direction {direction}")

            if current == end:
                log.info(f"  A* path found from {start} to {end}")
                return path

            g_cost = len(path) - 1

            if visited.get((current, direction), float("inf")) <= g_cost:
                continue
            visited[(current, direction)] = g_cost

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_direction = (dx, dy)
                neighbor = (current[0] + dx, current[1] + dy)

                if is_obstacle(neighbor):
                    continue

                new_g_cost = g_cost + 1
                turn_penalty = 0
                if direction != (0, 0) and new_direction != direction:
                    turn_penalty = 5

                h_cost = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                new_f_cost = new_g_cost + h_cost + turn_penalty

                new_path = path + [neighbor]
                heapq.heappush(pq, (new_f_cost, neighbor, new_path, new_direction))

        log.warning(f"  A* failed to find a path from {start} to {end}. Falling back to diagonal.")
        return [start_point.coords[0], end_point.coords[0]]

    def _check_connectivity(self, net, geom):
        if not net.pins:
            return True
        g = nx.Graph()
        pin_nodes = {tuple(p.geom.coords[0]) for p in net.pins}
        g.add_nodes_from(pin_nodes)
        log.info(f"  Connectivity check PASS for net {net.name} with pins {pin_nodes}")

        lines = []
        if geom and hasattr(geom, "geoms"):
            lines = geom.geoms
        elif isinstance(geom, LineString):
            lines = [geom]

        for line in lines:
            for i in range(len(line.coords) - 1):
                p1, p2 = tuple(line.coords[i]), tuple(line.coords[i + 1])
                g.add_edge(p1, p2)

        if not nx.is_connected(g):
            log.warning(f"  Connectivity check FAIL for net {net.name}: Not connected.")
            return False
        log.info(f"  Connectivity check PASS for net {net.name}")
        return True

    from shapely.geometry import LineString, MultiLineString

    def ensure_multilinestring(self, geom):
        """
        Convert a LineString or MultiLineString to a MultiLineString.
        Handles GeometryCollections by extracting all LineStrings.
        """
        if geom.is_empty:
            return MultiLineString([])

        if geom.geom_type == "LineString":
            return MultiLineString([geom])

        elif geom.geom_type == "MultiLineString":
            return geom

        elif geom.geom_type == "GeometryCollection":
            # Extract only the line components
            lines = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
            merged = []
            for g in lines:
                if g.geom_type == "LineString":
                    merged.append(g)
                else:  # MultiLineString
                    merged.extend(g.geoms)
            return MultiLineString(merged)

        else:
            raise TypeError(f"Unsupported geometry type for merge: {geom.geom_type}")

    def merge_collinear_segments(self, geom):
        if geom is None or geom.is_empty:
            return geom
        # First, merge connected lines into the fewest possible objects
        mls = self.ensure_multilinestring(geom)
        merged = linemerge(mls)
        # Then, simplify the geometry to remove redundant collinear points
        # A tolerance of 0.0 ensures only perfectly collinear points are removed.
        if hasattr(merged, "simplify"):
            return merged.simplify(0.0)
        return merged

    def _get_pin_macro(self, pin):
        for inst in self.db.top_module.get_all_instances().values():
            if pin in inst.pins.values():
                return inst.geom
        return None

    def _build_spatial_index(self, geoms):
        idx = index.Index()
        for i, geom in enumerate(geoms):
            if geom and not geom.is_empty:
                idx.insert(i, geom.bounds)
        return idx

    def optimize_layout(self):
        for inst in self.db.top_module.get_all_instances().values():
            if len(inst.pins) == 2:
                self.adjust_location(inst)
        breakpoint()
        self.beautify_routes()

        # Decompose the final, beautiful routes into simple segments for the writer
        for net in self.db.top_module.nets.values():
            net.geom = self._decompose_into_segments(net.geom)

        self.db.geom2shape()
