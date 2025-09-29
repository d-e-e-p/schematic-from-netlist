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


def round_point(pt, grid=1.0):
    """Round shapely Point to nearest grid multiple."""
    x = round(pt.x / grid) * grid
    y = round(pt.y / grid) * grid
    return Point(x, y)


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
            pin.geom = new_pin_geom

            if pin.net and pin.net.geom:
                new_segment = LineString([old_pin_geom, new_pin_geom])
                if hasattr(pin.net.geom, "geoms"):
                    existing_lines = list(pin.net.geom.geoms)
                else:
                    existing_lines = []
                pin.net.geom = MultiLineString(existing_lines + [new_segment])
                log.info(f"  Net {pin.net.name} updated: {pin.net.geom} with {new_segment=}")  # [line for line in net.geom.geoms]

        log.info(f"  Final geom: {inst.geom}, orient: {inst.orient}")
        for name, pin in inst.pins.items():
            log.info(f"  Pin {name}: {pin.geom}")

    def _plot_net_geometry(self, net, geom, stage):
        """
        Plots the geometry of a net at a given stage for debugging.
        """
        fig, ax = plt.subplots()
        ax.set_title(f"Net: {net.name} - Stage: {stage}")

        # Plot wires
        if geom and not geom.is_empty:
            if hasattr(geom, "geoms"):
                for line in geom.geoms:
                    x, y = line.xy
                    ax.plot(x, y, "b-")
            elif isinstance(geom, LineString):
                x, y = geom.xy
                ax.plot(x, y, "b-")
            breakpoint()

        # Plot pins
        pin_coords = [tuple(p.geom.coords[0]) for p in net.pins]
        if pin_coords:
            x, y = zip(*pin_coords)
            ax.plot(x, y, "ro", markersize=5)

        ax.set_aspect("equal", "datalim")

        # Create directory if it doesn't exist
        output_dir = "data/images/beautify_plots"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"net_{net.name}_{stage}.png")
        plt.savefig(filename)
        plt.close(fig)
        log.info(f"  Saved plot: {filename}")

    def _snap_to_grid(self, geom):
        """
        Snaps all coordinates of a geometry to the nearest integer.
        """
        if geom is None or geom.is_empty:
            return geom

        def snap_coords(coords):
            return [(round(x), round(y)) for x, y in coords]

        if isinstance(geom, LineString):
            return LineString(snap_coords(geom.coords))
        elif isinstance(geom, MultiLineString):
            return MultiLineString([LineString(snap_coords(line.coords)) for line in geom.geoms])
        elif isinstance(geom, Point):
            return Point(snap_coords([geom.coords[0]])[0])
        else:
            return geom

    def beautify_routes(self):
        """
        Main pipeline to beautify all routed nets.
        """
        log.info("Starting route beautification...")
        all_macros = [inst.geom for inst in self.db.top_module.get_all_instances().values()]
        macros_idx = self._build_spatial_index(all_macros)
        clearance = 1

        for net in self.db.top_module.nets.values():
            if not net.geom or net.geom.is_empty:
                continue
            log.info(f"Beautifying net {net.name}")
            self._plot_net_geometry(net, net.geom, "initial")

            connected_geom = self._force_connectivity(net, macros_idx, clearance)
            self._plot_net_geometry(net, connected_geom, "after_connectivity")

            snapped_geom = self._snap_to_grid(connected_geom)
            log.info(f"  Geom after snapping to grid: {snapped_geom}")
            self._plot_net_geometry(net, snapped_geom, "after_snapping")

            # Skip stub removal for now
            cleaned_geom = snapped_geom

            merged_geom = self.merge_collinear_segments(cleaned_geom)
            log.info(f"  Geom after merging: {merged_geom}")
            self._plot_net_geometry(net, merged_geom, "after_merge")

            jog_straightened_geom = self.straighten_jogs(merged_geom, macros_idx, clearance)
            orthogonal_geom = self.convert_diagonals(jog_straightened_geom, all_macros, macros_idx, clearance)
            final_geom = self.merge_duplicate_or_parallel_wires(orthogonal_geom)
            aligned_geom = self.align_rows_and_columns(final_geom)

            if isinstance(aligned_geom, LineString):
                aligned_geom = MultiLineString([aligned_geom])

            # Snap pins to the grid before the final connectivity check
            for pin in net.pins:
                pin.geom = self._snap_to_grid(pin.geom)

            self._check_connectivity(net, aligned_geom)
            self._plot_net_geometry(net, aligned_geom, "final")
            net.geom = aligned_geom

        self.resolve_crossings_between_nets(clearance)

    def _force_connectivity(self, net, macros_idx, clearance):
        """
        Ensures all pins in a net are part of the same connected component.
        """
        if not net.pins or len(net.pins) <= 1:
            return net.geom

        g = nx.Graph()
        pin_nodes = {tuple(p.geom.coords[0]): p for p in net.pins}
        g.add_nodes_from(pin_nodes.keys())

        lines = []
        if net.geom and hasattr(net.geom, "geoms"):
            lines = list(net.geom.geoms)

        for line in lines:
            for i in range(len(line.coords) - 1):
                p1 = tuple(line.coords[i])
                p2 = tuple(line.coords[i + 1])
                g.add_edge(p1, p2)

        components = list(nx.connected_components(g))
        if len(components) <= 1:
            return net.geom  # Already connected

        log.warning(f"  Net {net.name} has {len(components)} disconnected components. Forcing connectivity.")

        # Find the largest component (by number of pins)
        components.sort(key=lambda c: len(c.intersection(pin_nodes.keys())), reverse=True)
        main_component = components[0]

        new_lines = list(net.geom.geoms)

        for component in components[1:]:
            # Find a pin in this outlier component
            outlier_pin_node = None
            for node in component:
                if node in pin_nodes:
                    outlier_pin_node = node
                    break

            if not outlier_pin_node:
                # This component has no pins, just routing geometry. It can be discarded.
                continue

            # Find the nearest node in the main component to this outlier pin
            min_dist = float("inf")
            closest_main_node = None
            for main_node in main_component:
                dist = Point(outlier_pin_node).distance(Point(main_node))
                if dist < min_dist:
                    min_dist = dist
                    closest_main_node = main_node

            # Create a simple L-shaped route
            p1 = Point(outlier_pin_node)
            p2 = Point(closest_main_node)
            new_lines.append(LineString([p1, Point(p1.x, p2.y)]))
            new_lines.append(LineString([Point(p1.x, p2.y), p2]))

        return MultiLineString(new_lines)

    def _check_connectivity(self, net, geom):
        """
        Checks if all pins of a net are connected in the given geometry.
        """
        if not net.pins:
            return True

        g = nx.Graph()

        lines = []
        if hasattr(geom, "geoms"):
            lines = geom.geoms
        elif isinstance(geom, LineString):
            lines = [geom]

        for line in lines:
            for i in range(len(line.coords) - 1):
                p1 = tuple(line.coords[i])
                p2 = tuple(line.coords[i + 1])
                g.add_edge(p1, p2)

        if g.number_of_nodes() == 0:
            if len(net.pins) > 1:
                log.warning(f"  Connectivity check FAIL for net {net.name}: No geometry left, but {len(net.pins)} pins exist.")
                return False
            return True

        pin_nodes = {tuple(p.geom.coords[0]) for p in net.pins}

        # Check if all pin nodes are in the graph
        missing_pins = pin_nodes - set(g.nodes())
        if missing_pins:
            log.warning(f"  Connectivity check FAIL for net {net.name}: Pins {missing_pins} not in geometry graph.")

        # Find the component containing the first pin
        first_pin = next(iter(pin_nodes))
        if first_pin not in g:
            # Add pins to graph to ensure they are considered
            g.add_nodes_from(pin_nodes)
            if first_pin not in g:
                log.warning(f"  Connectivity check FAIL for net {net.name}: First pin {first_pin} not in geometry graph.")
                return False

        components = list(nx.connected_components(g))

        pin_component = None
        for component in components:
            if first_pin in component:
                pin_component = component
                break

        if pin_component is None:
            log.warning(f"  Connectivity check FAIL for net {net.name}: Could not find component for first pin.")
            return False

        # Check if all other pins are in the same component
        unconnected_pins = pin_nodes - pin_component
        if unconnected_pins:
            log.warning(
                f"  Connectivity check FAIL for net {net.name}: Pins {unconnected_pins} are not connected to the rest of the net."
            )
            return False

        log.info(f"  Connectivity check PASS for net {net.name}")
        return True

    def remove_loops_and_stubs(self, net, geom):
        """
        Remove redundant loops and dangling stubs within a single net.
        """
        pins = {tuple(p.geom.coords[0]) for p in net.pins}
        G = nx.Graph()
        G.add_nodes_from(pins)

        lines = []
        if hasattr(geom, "geoms"):
            lines = geom.geoms
        elif isinstance(geom, LineString):
            lines = [geom]

        for line in lines:
            for i in range(len(line.coords) - 1):
                p1 = tuple(line.coords[i])
                p2 = tuple(line.coords[i + 1])
                G.add_edge(p1, p2)

        # Remove stubs
        stubs_removed = True
        while stubs_removed:
            stubs_removed = False
            nodes_to_remove = []
            for node in G.nodes():
                if G.degree(node) == 1 and node not in pins:
                    nodes_to_remove.append(node)
            if nodes_to_remove:
                G.remove_nodes_from(nodes_to_remove)
                stubs_removed = True

        # Remove loops by finding MST for each connected component
        new_edges = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            mst = nx.minimum_spanning_tree(subgraph)
            new_edges.extend(mst.edges())

        new_lines = [LineString([u, v]) for u, v in new_edges]
        return MultiLineString(new_lines) if new_lines else MultiLineString()

    def merge_collinear_segments(self, geom):
        """
        Merge consecutive or overlapping collinear wire segments.
        """
        if not hasattr(geom, "geoms"):
            return geom
        return linemerge(geom)

    def straighten_jogs(self, geom, macros_idx, clearance):
        """
        Simplify small 90° jogs or zig-zags if they can be replaced by a straight line.
        """
        # This is a complex operation, so we'll start with a simple placeholder
        # and build it out in the next steps.
        return geom

    def convert_diagonals(self, geom, all_macros, macros_idx, clearance):
        """
        Replace diagonal segments with equivalent orthogonal L-shaped connections.
        """
        if not hasattr(geom, "geoms"):
            return geom

        new_lines = []
        for line in geom.geoms:
            p1 = Point(line.coords[0])
            p2 = Point(line.coords[1])

            if p1.x == p2.x or p1.y == p2.y:
                new_lines.append(line)
                continue

            # Diagonal line
            l_path1 = LineString([p1, Point(p1.x, p2.y), p2])
            l_path2 = LineString([p1, Point(p2.x, p1.y), p2])

            is_safe1 = True
            for i in macros_idx.intersection(l_path1.buffer(clearance).bounds):
                if all_macros[i].intersects(l_path1.buffer(clearance)):
                    is_safe1 = False
                    break

            if is_safe1:
                new_lines.extend([LineString([p1, Point(p1.x, p2.y)]), LineString([Point(p1.x, p2.y), p2])])
            else:
                is_safe2 = True
                for i in macros_idx.intersection(l_path2.buffer(clearance).bounds):
                    if all_macros[i].intersects(l_path2.buffer(clearance)):
                        is_safe2 = False
                        break
                if is_safe2:
                    new_lines.extend([LineString([p1, Point(p2.x, p1.y)]), LineString([Point(p2.x, p1.y), p2])])
                else:
                    # Fallback to original diagonal
                    new_lines.append(line)

        return MultiLineString(new_lines) if new_lines else MultiLineString()

    def fix_t_junctions(self, geom):
        """
        Normalize and align all T-junctions.
        """
        # Placeholder
        return geom

    def merge_duplicate_or_parallel_wires(self, geom):
        """
        Remove duplicate, overlapping, or near-parallel wire segments within the same net.
        """
        # Placeholder
        return geom

    def align_rows_and_columns(self, geom):
        """
        Snap endpoints of wires and pins to shared grid rows/columns for visual alignment.
        """
        # Placeholder
        return geom

    def resolve_crossings_between_nets(self, clearance):
        """
        Detect and resolve visual wire crossings between different nets.
        """
        # Placeholder
        pass

    def _build_spatial_index(self, geoms):
        """
        Build an R-tree index for fast intersection/overlap queries.
        """
        idx = index.Index()
        for i, geom in enumerate(geoms):
            if geom and not geom.is_empty:
                idx.insert(i, geom.bounds)
        return idx

    def optimize_layout(self):
        for inst in self.db.top_module.get_all_instances().values():
            if len(inst.pins) == 2:
                self.adjust_location(inst)
        # self.db.geom2shape()
        # self.beautify_routes()
        self.db.geom2shape()
        breakpoint()
