import heapq
import logging as log
import math
import os

import networkx as nx
import shapely
from matplotlib import pyplot as plt
from rtree import index
from scipy.optimize import linear_sum_assignment
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import linemerge, unary_union

from schematic_from_netlist.interfaces.symbol_library import SymbolLibrary

log.getLogger("matplotlib").setLevel(log.WARNING)


class LayoutOptimizer:
    def __init__(self, db):
        self.db = db
        self.symbol_outlines = SymbolLibrary().get_symbol_outlines()

    def optimize_component_placement(self, inst):
        self.adjust_location(inst)

    def _calculate_best_orientation(self, old_pins, new_local_pins, centroid):
        """
        Choose rotation that minimizes total pin movement using Hungarian algorithm.
        """
        orientations = {"R0": 0, "R90": 90, "R180": 180, "R270": 270}
        min_total = float("inf")
        best_orient = None
        best_pin_map = None
        best_rotated_local = None

        for orient_name, angle in orientations.items():
            # Rotate local pins around (0,0)
            rotated_local = [rotate(p, angle, origin=(0, 0)) for p in new_local_pins]

            # Convert to global using centroid (alignment anchor)
            rotated_global = [Point(centroid.x + p.x, centroid.y + p.y) for p in rotated_local]

            # Build cost matrix: distance old <-> new
            cost_matrix = [[op.distance(rp) for rp in rotated_global] for op in old_pins]

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_dist = sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind))

            pin_map = dict(zip(row_ind, col_ind))

            log.debug(f"{orient_name}: angle={angle}, dist={total_dist:.2f}, map={pin_map}")

            if total_dist < min_total:
                min_total = total_dist
                best_orient = orient_name
                best_pin_map = pin_map
                best_rotated_local = rotated_local

        log.info(f"Best orientation: {best_orient} dist={min_total:.2f}")

        return best_orient, best_pin_map, best_rotated_local

    def adjust_location(self, inst):
        """
        Replace the instance geometry with the symbol definition and
        orient & position it to minimize wire displacement.
        """

        sym = self.symbol_outlines[inst.module.name]
        width, height = sym.width, sym.height
        port_defs = sym.ports  # dict name -> Port

        old_geom = inst.draw.geom
        old_centroid = old_geom.centroid

        # Existing global pin positions
        pin_names = list(inst.pins.keys())
        old_pins = [p.draw.geom for p in inst.pins.values()]

        # New symbol base shape centered at origin
        macro_box_local = box(-width / 2, -height / 2, width / 2, height / 2)

        # New symbol pins (local coordinates)
        macro_pins_local = [port_defs[name].fig for name in pin_names]

        # Best rotation for pin matching
        best_orient, best_pin_map, rotated_local = self._calculate_best_orientation(
            old_pins=old_pins,
            new_local_pins=macro_pins_local,
            centroid=old_centroid,  # rotate around centroid
        )

        angle = {"R0": 0, "R90": 90, "R180": 180, "R270": 270}[best_orient]

        # ===== Apply rotation around centroid =====
        rotated_geom = rotate(macro_box_local, angle, origin=(0, 0))
        rotated_geom = translate(rotated_geom, xoff=old_centroid.x, yoff=old_centroid.y)

        # Align first matched pin to minimize movement
        i0 = 0
        target_pos = old_pins[i0]
        local_rot = rotated_local[best_pin_map[i0]]
        rotated_pin_global = Point(old_centroid.x + local_rot.x, old_centroid.y + local_rot.y)

        dx = target_pos.x - rotated_pin_global.x
        dy = target_pos.y - rotated_pin_global.y

        # Apply final translation
        final_geom = translate(rotated_geom, xoff=dx, yoff=dy)
        inst.draw.geom = final_geom
        inst.draw.orient = best_orient

        # Update pins & routing segments
        for i, pin_name in enumerate(pin_names):
            pin = inst.pins[pin_name]
            old_pin = old_pins[i]
            new_local = rotated_local[best_pin_map[i]]
            new_pin = Point(new_local.x + old_centroid.x + dx, new_local.y + old_centroid.y + dy)

            # Extend net routing if exists
            if pin.net:
                new_segment = LineString([old_pin, new_pin])
                if pin.net.draw.geom and hasattr(pin.net.draw.geom, "geoms"):
                    existing = list(pin.net.draw.geom.geoms)
                else:
                    existing = []
                pin.net.draw.geom = MultiLineString(existing + [new_segment])

            pin.draw.geom = new_pin

        log.info(f"→ {inst.name} placed: orient={inst.draw.orient}, geom={inst.draw.geom}")

    def _get_local_pin_positions(self, inst):
        """Convert each global pin position into macro-local coordinates relative to macro center."""
        macro_center = inst.draw.geom.centroid
        locals = []
        for pin in inst.pins.values():
            gp = pin.draw.geom
            locals.append(Point(gp.x - macro_center.x, gp.y - macro_center.y))
        return locals

    def _apply_orientation(self, geom, angle, origin):
        """Rotate geom around macro center."""
        return rotate(geom, angle, origin=origin)

    def _translate_macro(self, geom, dx, dy):
        """Apply a rigid translation to entire geometry."""
        return translate(geom, xoff=dx, yoff=dy)

    def _orient_to_angle(self, orient):
        """Convert R90 etc. into a numeric rotation angle."""
        return {"R0": 0, "R90": 90, "R180": 180, "R270": 270}[orient]

    def _plot_net_geom(self, net, geom, stage, old_geom=None, all_macros=None):
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
        all_macros = {inst.name: inst.draw.geom for inst in self.db.top_module.get_all_instances().values()}
        clearance = 2.0
        routed_net.draw.geoms = []
        epsilon = -1e-9  # A small negative buffer to shrink macros slightly

        # Sort nets by name for deterministic routing order
        nets_to_route = sorted(self.db.top_module.nets.values(), key=lambda n: n.num_conn)
        # nets_to_route = [self.db.top_module.nets["Net__U1_PA10_A2_D2_"]]

        for net in nets_to_route:
            if not net.pins or len(net.pins) < 2:
                continue

            log.info(f"Beautifying net {net.name}")
            self._plot_net.draw.geometry(net, net.draw.geom, "initial", all_macros=all_macros.values())

            for pin in net.pins:
                pin.draw.geom = self._snap_to_grid(pin.draw.geom)

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
            if routed_net.draw.geoms:
                obstacles.append(unary_union(routed_net.draw.geoms).buffer(clearance))

            obstacle_union = unary_union(obstacles) if obstacles else None
            # --- End Obstacle Calculation ---

            log.info(f"astar b {net.name}")
            new_geom = self.reroute_net_with_astar(net, obstacle_union)
            log.info(f"astar c {net.name}")

            if new_geom and not new_geom.is_empty:
                merged_geom = self.merge_collinear_segments(new_geom)
                if isinstance(merged_geom, LineString):
                    merged_geom = MultiLineString([merged_geom])

                self._plot_net.draw.geometry(
                    net, merged_geom, "after_astar", old_geom=net.draw.geom, all_macros=all_macros.values()
                )
                self._check_connectivity(net, merged_geom)
                net.draw.geom = merged_geom
                routed_net.draw.geoms.append(merged_geom)  # Add to obstacles for the next net
            else:
                log.warning(f"  Routing failed for net {net.name}, keeping original geometry.")
                self._plot_net.draw.geometry(net, net.draw.geom, "final_failed", all_macros=all_macros.values())

    def reroute_net_with_astar(self, net, obstacle_union):
        pins = list(net.pins)
        pin.draw.geoms = [p.geom for p in pins]
        if len(pin.draw.geoms) < 2:
            return None

        # Build a Minimum Spanning Tree (MST) to define the primary connection topology
        pin_graph = nx.Graph()
        for i in range(len(pin.draw.geoms)):
            for j in range(i + 1, len(pin.draw.geoms)):
                p1 = pin.draw.geoms[i]
                p2 = pin.draw.geoms[j]
                dist = abs(p1.x - p2.x) + abs(p1.y - p2.y)
                pin_graph.add_edge(i, j, weight=dist)
        mst = nx.minimum_spanning_tree(pin_graph)

        # --- Pass 1: Route all MST edges independently ---
        routed_segments = []
        for u, v in mst.edges():
            start_pin = pin.draw.geoms[u]
            end_pin = pin.draw.geoms[v]
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
                return inst.draw.geom
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
        self.beautify_routes()

        # Decompose the final, beautiful routes into simple segments for the writer
        for net in self.db.top_module.nets.values():
            net.draw.geom = self._decompose_into_segments(net.draw.geom)

        self.db.geom2shape()
