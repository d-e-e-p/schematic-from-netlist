import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

from schematic_from_netlist.interfaces.netlist_database import Pin


@dataclass
class NetShape:
    name: str
    points: List[Tuple[int, int]] | None = None
    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] | None = None

    def __hash__(self):
        return hash(self.name)


@dataclass
class PortShape:
    name: str
    pin: Pin
    point: Tuple[int, int]

    def __hash__(self):
        return hash(self.name)


@dataclass
class InstShape:
    name: str
    module_ref: str
    rect: Tuple[int, int, int, int]
    port_shapes: List[PortShape]

    def __hash__(self):
        return hash(self.name)


@dataclass
class SchmaticData:
    sheet_size: Tuple[int, int] = (1000, 1000)
    port_shapes: List[PortShape] = field(default_factory=list)
    inst_shapes: List[InstShape] = field(default_factory=list)
    net_shapes: List[NetShape] = field(default_factory=list)
    # lookup tables (lazy-built via _build_lookup_tables)
    netshape_by_name: dict[str, NetShape] = field(init=False, default_factory=dict)
    instshape_by_name: dict[str, InstShape] = field(init=False, default_factory=dict)
    portshapes_by_instname: dict[str, list[PortShape]] = field(init=False, default_factory=dict)
    portshape_by_name: dict[str, PortShape] = field(init=False, default_factory=dict)

    def _build_lookup_tables(self):
        """Build fast lookup dictionaries for nets, instances, and ports."""

        # Nets
        self.netshape_by_name = {net.name: net for net in self.net_shapes}

        # Instances
        self.instshape_by_name = {inst.name: inst for inst in self.inst_shapes}

        # Ports by instance name
        ports_by_inst = defaultdict(list)
        for inst in self.inst_shapes:
            for port in inst.port_shapes:
                ports_by_inst[inst.name].append(port)
        self.portshapes_by_instname = dict(ports_by_inst)

        # Ports by name (unique lookup)
        self.portshape_by_name = {}
        # TODO: First add top-level sheet ports

        # Then add instance ports
        for inst in self.inst_shapes:
            # print(f"{inst.name=}")
            for port in inst.port_shapes:
                # print(f"{port.name=} {port.point=}")
                if port.name in self.portshape_by_name:
                    pass
                    # TODO fix remove buffer multi causing this
                    # raise ValueError(f"Duplicate port name across instances: {port.name}")
                self.portshape_by_name[port.name] = port


class GenSchematicData(SchmaticData):
    def __init__(self, geom_db, netlist_db):
        super().__init__()
        self.geom_db = geom_db
        self.db = netlist_db
        self.graph_to_sch_scale = 0.25  # from graphviz to LTspice, needs to be adjusted to avoid limits
        self.schematic_grid_size = 16  # needs to be a multiple of 50 mils on kicad import for wires to connect to pin
        # TODO: many make it equal area to the smallest other macro?
        self.min_block_size_in_gridlines = 10
        self.block_moat_size_in_gridlines = 1

    def find_net_between_inst(self, list_of_inst):
        """
        Find nets that are common to all instances in list_of_inst.
        Each inst must have .get_connected_nets() -> List[Net].
        """
        if not list_of_inst:
            return []

        # Get nets of the first inst.. common_bets is a dict of name2net
        common_nets = {net.name: net for net in list_of_inst[0].get_connected_nets()}

        # Intersect with the rest
        for inst in list_of_inst[1:]:
            inst_nets = {net.name: net for net in inst.get_connected_nets()}
            # keep only common nets
            common_nets = {n: net for n, net in common_nets.items() if n in inst_nets}

        # Return list of Net objects
        return list(common_nets.values())

    def associate_ports_to_blocks(self):
        # look over all port pairs and compare them to

        for port in self.geom_db.ports:
            inst1 = self.db.inst_by_name[port.name]
            inst2 = self.db.inst_by_name[port.conn]

            # find a net connecting these 2 inst
            match_nets = self.find_net_between_inst([inst1, inst2])
            # print(f"{[net.name for net in match_nets]=} {inst1.name=} {inst2.name=}")

            # if multiple net connections between the same 2 inst, we have to assign one port pair
            # to each connection
            for net in match_nets:
                for pin in net.pins:
                    if pin.instance.name == inst1.name:
                        x, y = port.point
                        scaled_point = (int(x * self.graph_to_sch_scale), int(y * self.graph_to_sch_scale))
                        ps = PortShape(pin.full_name, pin, scaled_point)
                        self.port_shapes.append(ps)

    def find_block_bounding_boxes(self):
        instname2shapes = {}

        # Group port_shapes by instance name
        for shape in self.port_shapes:
            pin = getattr(shape, "pin", None)
            if pin is None or pin.instance is None:
                continue  # skip dangling ports
            instname = pin.instance.name
            instname2shapes.setdefault(instname, []).append(shape)

        # Aggregate rects for each instance
        for instname, shapes in instname2shapes.items():
            pt_list = [s.point for s in shapes if hasattr(s, "point")]

            if not pt_list:
                continue

            # TODO: need to grow nicely -- if only 1 port
            # if len(pt_list) == 1:

            x_min = min(p[0] for p in pt_list)
            y_min = min(p[1] for p in pt_list)
            x_max = max(p[0] for p in pt_list)
            y_max = max(p[1] for p in pt_list)

            # too small -- need to grow
            size = self.min_block_size_in_gridlines
            if x_max - x_min < size:
                x_min -= size // 2
                x_max += size // 2
            if y_max - y_min < size:
                y_min -= size // 2
                y_max += size // 2

            # block_moat_size a bit
            size = self.block_moat_size_in_gridlines
            x_min -= size
            x_max += size
            y_min -= size
            y_max += size

            rect = (x_min, y_min, x_max, y_max)

            inst = self.db.inst_by_name.get(instname)
            if inst is None:
                # fallback or skip if inst not found
                print(f"Warning: Inst {instname} not found")
                continue

            module_ref = getattr(inst, "module_ref", "")
            # print(f"{instname=} {module_ref=} {rect=}")
            # for shape in shapes:
            #    print(f"  {shape.pin.name=} {shape.point=}")
            inst_shape = InstShape(
                name=instname,
                module_ref=module_ref,
                rect=rect,
                port_shapes=shapes,
            )
            self.inst_shapes.append(inst_shape)

    def scale_and_uniq_points(self, points):
        """
        Remove adjacent identical points from a list.
        """
        if not points:
            return []

        scaled_points = [(int(x * self.graph_to_sch_scale), int(y * self.graph_to_sch_scale)) for x, y in points]
        unique_points = [scaled_points[0]]  # Always keep the first point

        for pt in scaled_points[1:]:
            # Only add if different from the previous point
            if pt != unique_points[-1]:
                unique_points.append(pt)

        return unique_points

    def find_net_shapes(self):
        for net_geom in self.geom_db.nets:
            list_of_inst = []
            for instname in net_geom.conn:
                list_of_inst.append(self.db.inst_by_name[instname])

            nets = self.find_net_between_inst(list_of_inst)
            if nets:
                netname = nets[0].name
                points = self.scale_and_uniq_points(net_geom.points)
                # print(f"{netname}: {points}")
                net_shape = NetShape(name=netname, points=points)
                self.net_shapes.append(net_shape)

    def center_points(self, rect):
        # center of first rectangle
        center_x = round((rect[0] + rect[2]) / 2)
        center_y = round((rect[1] + rect[3]) / 2)
        point = (center_x, center_y)
        return point

    def snap_buffer_wire_to_center(self, pt_center, pt_port):
        for shape in self.net_shapes:
            points = shape.points
            new_pts = []
            modified = False
            for pt_wire in points:
                if pt_wire == pt_port:
                    new_pts.append(pt_center)
                else:
                    new_pts.append(pt_wire)
                    modified = True
            if modified:
                shape.points = new_pts
                shape.name = re.sub(r"_fanout_buffer_\d+", "", shape.name)

    def route_to_center(self, pt_center, pt_port):
        pass

    def patch_and_remove_buffers(self):
        """
        Finds buffer instances, replaces them with a wire in the geometric data,
        and then removes them from the logical netlist.
        """
        buffers_to_remove = []

        # Find buffer instance shapes
        for inst_shape in self.inst_shapes:
            if inst_shape.name.startswith(self.db.inserted_buf_prefix):
                buffers_to_remove.append(inst_shape)

                # Get center points of the buffer inst
                pt_center = self.center_points(inst_shape.rect)
                for port_shape in inst_shape.port_shapes:
                    pt_port = port_shape.point
                    self.route_to_center(pt_center, pt_port)

        # ok now all wires of buffering nets
        for net_shape in self.net_shapes:
            if net_shape.name.find(self.db.inserted_net_suffix) > -1:
                orig_netname = re.sub(rf"{re.escape(self.db.inserted_net_suffix)}\d+$", "", net_shape.name)
                net_shape.name = orig_netname

        # ok now remove logical buffers and restore logical connection
        self.db.remove_multi_fanout_buffers()
        self._build_lookup_tables()

    def shift_all_geom_by(self, dx, dy):
        for inst_shape in self.inst_shapes:
            x1, y1, x2, y2 = inst_shape.rect
            inst_shape.rect = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            for port_shape in inst_shape.port_shapes:
                port_shape.point = (port_shape.point[0] + dx, port_shape.point[1] + dy)

        for net_shape in self.net_shapes:
            net_shape.points = [(x + dx, y + dy) for x, y in net_shape.points]

    def calc_sheet_size(self):
        """too concise perhaps?"""
        coords = []

        # Rectangles: extract x1, y1, x2, y2
        for shape in self.inst_shapes:
            coords.extend([shape.rect[0], shape.rect[1], shape.rect[2], shape.rect[3]])

        # Points: extract x, y from each point
        for net_shape in self.net_shapes:
            for x, y in net_shape.points:
                coords.extend([x, y])

        padding = 100  # units is in grid by this point
        if coords:
            xs = coords[::2]  # Even indices
            ys = coords[1::2]  # Odd indices
            if min(xs) < padding or min(ys) < padding:
                self.shift_all_geom_by(padding - min(xs), padding - min(ys))

            xsize = int(max(xs) - min(xs) + 2 * padding)
            ysize = int(max(ys) - min(ys) + 2 * padding)

            self.sheet_size = (xsize, ysize)
        else:
            self.sheet_size = (padding, padding)

    def flip_y_axis(self):
        """
        In schematic drawing programs historically the origin is upper-left.
        This flips all shapes vertically relative to the sheet height.
        """
        _, sheet_height = self.sheet_size

        # Flip instance rectangles
        for shape in self.inst_shapes:
            x1, y1, x2, y2 = shape.rect
            # Reflect across horizontal axis at sheet_height
            y1_flipped = sheet_height - y1
            y2_flipped = sheet_height - y2
            # Normalize (y1 should be min, y2 max)
            y1n, y2n = min(y1_flipped, y2_flipped), max(y1_flipped, y2_flipped)
            shape.rect = (x1, y1n, x2, y2n)

        for shape in self.inst_shapes:
            for shape in shape.port_shapes:
                x, y = shape.point
                # Reflect across horizontal axis at sheet_height
                y_flipped = sheet_height - y
                shape.point = (x, y_flipped)

        # Flip net points
        for shape in self.net_shapes:
            flipped_points = []
            for x, y in shape.points:
                flipped_points.append((x, sheet_height - y))
            shape.points = flipped_points

    def snap_all_points_to_grid(self):
        """
        Ensures coordinates are multiples of self.grid_size.
        """

        def snap(v):
            # Round to nearest multiple of grid
            return round(v * self.graph_to_sch_scale)

        # Snap instance rectangles
        for inst_shape in self.inst_shapes:
            x1, y1, x2, y2 = inst_shape.rect
            inst_shape.rect = (snap(x1), snap(y1), snap(x2), snap(y2))

            # Snap instance ports
            for port_shape in inst_shape.port_shapes:
                x, y = port_shape.point
                port_shape.point = (snap(x), snap(y))

        # Snap net shapes
        for net_shape in self.net_shapes:
            snapped_points = []
            for x, y in net_shape.points:
                snapped_points.append((snap(x), snap(y)))
            net_shape.points = snapped_points

    def generate_schematic_info(self):
        self.associate_ports_to_blocks()
        self.find_block_bounding_boxes()
        self.find_net_shapes()
        self.patch_and_remove_buffers()
        self.snap_all_points_to_grid()
        self.calc_sheet_size()
        # self.flip_y_axis()
        return self
