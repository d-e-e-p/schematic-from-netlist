from dataclasses import dataclass, field
from typing import List, Tuple

from schematic_from_netlist.interfaces.netlist_database import Pin


@dataclass
class NetShape:
    name: str
    points: List[Tuple[int, int]]


@dataclass
class PortShape:
    pin: Pin
    point: Tuple[int, int]


@dataclass
class InstShape:
    name: str
    module_ref: str
    rect: Tuple[int, int, int, int]
    port_shapes: List[PortShape]


@dataclass
class SchmaticData:
    sheet_size: Tuple[int, int] = (1000, 1000)
    port_shapes: List[PortShape] = field(default_factory=list)
    inst_shapes: List[InstShape] = field(default_factory=list)
    net_shapes: List[NetShape] = field(default_factory=list)


class GenSchematicData(SchmaticData):
    def __init__(self, geom_db, netlist_db):
        super().__init__()
        self.geom_db = geom_db
        self.netlist_db = netlist_db
        self.scale = 16  # from graphviz to LTspice
        self.grid_size = 16  # needs to be a multiple of 50 mils on kicad import for wires to connect to pin
        # TODO: many make it equal area to the smallest other macro?
        self.min_block_size = 10 * self.scale
        self.block_moat_size = int(1 * self.scale)

    def find_net_between_inst(self, list_of_inst):
        """
        Find nets that are common to all instances in list_of_inst.
        Each inst must have .get_connected_nets() -> List[Net].
        """
        if not list_of_inst:
            return []

        # Get nets of the first inst
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
            inst1 = self.netlist_db.inst_by_name[port.name]
            inst2 = self.netlist_db.inst_by_name[port.conn]

            # find a net connecting these 2 inst
            match_nets = self.find_net_between_inst([inst1, inst2])

            # if multiple net connections between the same 2 inst, we have to assign one port pair
            # to each connection
            for net in match_nets:
                for pin in net.pins:
                    if pin.instance.name == inst1.name:
                        x, y = port.point
                        scaled_point = (int(x * self.scale), int(y * self.scale))
                        ps = PortShape(pin, scaled_point)
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
            bs = self.min_block_size
            if x_max - x_min < bs:
                x_min -= bs // 2
                x_max += bs // 2
            if y_max - y_min < bs:
                y_min -= bs // 2
                y_max += bs // 2

            # block_moat_size a bit
            bf = self.block_moat_size
            x_min -= bf
            x_max += bf
            y_min -= bf
            y_max += bf

            rect = (x_min, y_min, x_max, y_max)
            # Ensure hierarchical instname is well-formed
            full_instname = instname
            if not full_instname.startswith(self.netlist_db.top_module.name + "/"):
                full_instname = f"{self.netlist_db.top_module.name}/{instname}"

            inst = self.netlist_db.inst_by_name.get(full_instname)
            if inst is None:
                # fallback or skip if inst not found
                continue

            module_ref = getattr(inst, "module_ref", "")
            print(f"{instname=} {module_ref=} {rect=}")
            for shape in shapes:
                print(f"  {shape.pin.name=} {shape.point=}")
            inst_shape = InstShape(
                name=full_instname,
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

        scaled_points = [(int(x * self.scale), int(y * self.scale)) for x, y in points]
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
                list_of_inst.append(self.netlist_db.inst_by_name[instname])

            nets = self.find_net_between_inst(list_of_inst)
            if nets:
                netname = nets[0].name
                points = self.scale_and_uniq_points(net_geom.points)
                print(f"{netname}: {points}")
                net_shape = NetShape(name=netname, points=points)
                self.net_shapes.append(net_shape)

    def center_points(self, rect0, rect1):
        # center of first rectangle
        center0_x = (rect0[0] + rect0[2]) / 2
        center0_y = (rect0[1] + rect0[3]) / 2

        # center of second rectangle
        center1_x = (rect1[0] + rect1[2]) / 2
        center1_y = (rect1[1] + rect1[3]) / 2

        point0 = (int(center0_x), int(center0_y))
        point1 = (int(center1_x), int(center1_y))

        return [point0, point1]

    def patch_and_remove_buffers(self):
        """
        Finds buffer instances, replaces them with a wire in the geometric data,
        and then removes them from the logical netlist.
        """
        buffers_to_remove = []
        new_wires = []

        # Find buffer instance shapes
        for inst_shape in self.inst_shapes:
            if inst_shape.name.find("/buf_") > 0:
                buffers_to_remove.append(inst_shape)

                input_port = None
                output_port = None
                for port_shape in inst_shape.port_shapes:
                    if port_shape.pin.name == "I":
                        input_port = port_shape
                    elif port_shape.pin.name == "O":
                        output_port = port_shape

                if input_port and output_port:
                    # Create a new wire to "patch" the connection
                    in_rect = input_port.rect
                    out_rect = output_port.rect

                    # Get center points of the port rectangles
                    scaled_points = self.center_points(in_rect, out_rect)

                    # The net can be anything as it's just for drawing, just should NOT be labeled
                    # TODO: replace orig name
                    new_wires.append(NetShape(name="", points=scaled_points))

        # Remove buffer shapes from the list
        self.inst_shapes = [s for s in self.inst_shapes if s not in buffers_to_remove]

        # Add the new patch wires
        self.net_shapes.extend(new_wires)

        # Remove buffers from the logical netlist
        self.netlist_db.remove_buffers()
        print(f"Patched and removed {len(buffers_to_remove)} buffer instances.")

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

        if coords:
            xs = coords[::2]  # Even indices
            ys = coords[1::2]  # Odd indices

            padding = 100
            xsize = int(max(xs) - min(xs) + 2 * padding)
            ysize = int(max(ys) - min(ys) + 2 * padding)

            self.sheet_size = (int(max(xs) - min(xs) + 2 * padding), int(max(ys) - min(ys) + 2 * padding))
        else:
            self.sheet_size = (1000, 1000)

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
        Snap all schematic geometry (instances, ports, nets) to the nearest grid point.
        Ensures coordinates are multiples of self.grid_size.
        """
        grid = self.grid_size

        def snap(v):
            # Round to nearest multiple of grid
            return round(v / grid) * grid

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
        self.snap_all_points_to_grid()
        self.calc_sheet_size()
        self.flip_y_axis()
        return self
