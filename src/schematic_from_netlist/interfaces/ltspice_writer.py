import os


class LTSpiceWriter:
    def __init__(self, db, schematic_db):
        self.db = db
        self.schematic_db = schematic_db
        self.module_names = {}
        self.add_comments = False  # kicad can't seem to parse spice with comments

    def upscale_rect(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Scale a rectangle (x1,y1,x2,y2) from grid units to real units."""
        g = self.schematic_db.schematic_grid_size
        return (rect[0] * g, rect[1] * g, rect[2] * g, rect[3] * g)

    def upscale_segments(self, segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Scale a list of line segments [((x1,y1),(x2,y2)), ...]."""
        g = self.schematic_db.schematic_grid_size
        return [((x1 * g, y1 * g), (x2 * g, y2 * g)) for (x1, y1), (x2, y2) in segments]

    def upscale_points(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Scale a list of points [(x,y), ...]."""
        g = self.schematic_db.schematic_grid_size
        return [(x * g, y * g) for (x, y) in points]

    def upscale_point(self, point: tuple[int, int]) -> tuple[int, int]:
        """Scale a list of points [(x,y), ...]."""
        g = self.schematic_db.schematic_grid_size
        x, y = point
        return (x * g, y * g)

    def _asc_place_inst(self, inst_shape):
        """Formats a single symbol line for the .asc file."""
        out = ""
        name = inst_shape.name
        module_ref = inst_shape.module_ref
        x1, y1, x2, y2 = self.upscale_rect(inst_shape.rect)
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        out += f"SYMBOL {module_ref} {x} {y} R0\n"
        out += f"SYMATTR InstName {name}\n"
        out += f"SYMATTR Value {module_ref}\n"
        return out

    def _format_asc_wire(self, wire_shape):
        """Formats a single WIRE line for the .asc file."""
        # out = "* WIRE\n"
        out = ""
        comment = f" $   {wire_shape.name}" if self.add_comments else ""
        if wire_shape.segments is not None:
            segments = self.upscale_segments(wire_shape.segments)
            print(f"{wire_shape.name} {segments=}")
            for seg in segments:
                pt_start, pt_end = seg
                out += f"WIRE {pt_start[0]} {pt_start[1]} {pt_end[0]} {pt_end[1]}{comment}\n"

        if wire_shape.points is not None:
            points = self.upscale_points(wire_shape.points)
            pt_start = points[0]
            for pt in points[1:]:
                pt_end = pt
                out += f"WIRE {pt_start[0]} {pt_start[1]} {pt_end[0]} {pt_end[1]}{comment}\n"
                pt_start = pt_end
        return out

    def _uniquify_module_name(self, inst_name, module_name):
        name = module_name.replace("/", "_")
        if name in self.module_names:
            self.module_names[name] += 1
            return f"{name}_{self.module_names[name]}"
        else:
            self.module_names[name] = 0
            return name

    def _get_pin_side(self, pt, rect):
        """Expecting NONE, BOTTOM, TOP, LEFT, RIGHT, VBOTTOM, VTOP, VCENTER, VLEFT or VRIGHT"""
        x, y = pt
        x1, y1, x2, y2 = rect

        # Calculate distances to each side
        dist_left = abs(x - x1)
        dist_right = abs(x - x2)
        dist_top = abs(y - y1)
        dist_bottom = abs(y - y2)

        # Return closest side
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_dist == dist_left:
            return "RIGHT"
        elif min_dist == dist_right:
            return "LEFT"
        elif min_dist == dist_top:
            return "BOTTOM"
        elif min_dist == dist_bottom:
            return "TOP"

        return "TOP"

    def generate_symbol_window_commands(self, hw, hh):
        """Generate symbol with properly positioned WINDOW fields."""

        # Symbol shape
        asy = f"RECTANGLE NORMAL {-hw} {-hh} {hw} {hh}\n"

        # Calculate text positions relative to symbol bounds
        # Rectangle bounds: left=-hw, top=-hh, right=hw, bottom=hh

        # Option 1: Outside the rectangle
        ref_x = hw + 8  # Reference name to the right
        ref_y = -hh  # Aligned with top

        val_x = hw + 8  # Value to the right
        val_y = hh  # Aligned with bottom

        # Option 2: Inside the rectangle (if large enough)
        if hw > 30 and hh > 20:
            ref_x = 0  # Centered horizontally
            ref_y = -hh + 8  # Near top, inside
            val_x = 0  # Centered horizontally
            val_y = hh - 8  # Near bottom, inside

        simd_x, simd_y = (0, -10)
        simp_x, simp_y = (0, 10)

        # Add WINDOW definitions
        font_size = round(10 * self.schematic_db.graph_to_sch_scale)
        asy = ""
        asy += f"WINDOW 0 {ref_x} {ref_y} Left {font_size}\n"  # InstName (Reference)
        asy += f"WINDOW 3 {val_x} {val_y} Left {font_size}\n"  # Value
        asy += f"WINDOW 38 {simd_x} {simd_x} Left {font_size}\n"  # Sim.Device
        asy += f"WINDOW 39 {simp_y} {simp_y} Left {font_size}\n"  # Sim.Params

        return asy

    def _generate_symbol_asy(self, inst_shape, output_dir="data/ltspice"):
        """Generates an .asy file for a given module."""

        rect = self.upscale_rect(inst_shape.rect)
        hw = (rect[2] - rect[0]) // 2  # half width
        hh = (rect[3] - rect[1]) // 2  # half height

        inst_offset_x = (rect[2] + rect[0]) // 2
        inst_offset_y = (rect[3] + rect[1]) // 2

        cell_rect = [-hw, -hh, hw, hh]

        asy = "Version 4\n"
        asy += "SymbolType BLOCK\n"
        asy += f"RECTANGLE NORMAL {-hw} {-hh} {hw} {hh}\n"
        asy += f"SYMATTR SpiceModel {inst_shape.module_ref}\n"  # This sets Sim.Device
        asy += f"SYMATTR SpiceLine -\n"  # This sets Sim.Device
        asy += self.generate_symbol_window_commands(hw, hh)
        for shape in inst_shape.port_shapes:
            port_center_x, port_center_y = self.upscale_point(shape.point)
            x = port_center_x - inst_offset_x
            y = port_center_y - inst_offset_y
            pt = (x, y)
            side = self._get_pin_side(pt, cell_rect)
            asy += f"PIN {x} {y} {side} 0\n"
            asy += f"PINATTR PinName {shape.pin.name}\n"

        asy_path = os.path.join(output_dir, f"{inst_shape.module_ref}.asy")
        with open(asy_path, "w") as f:
            f.write(asy)
        print(f"Generated symbol file: {asy_path}")

    def uniquify_module_names(self):
        """
        Make every module different so it can have ports in different locations
        """
        for inst_shape in self.schematic_db.inst_shapes:
            module_name = self._uniquify_module_name(inst_shape.name, inst_shape.module_ref)
            inst_shape.module_ref = module_name

    def produce_schematic(self, output_dir="data/ltspice"):
        """Generates the .asc schematic and .asy symbol files."""
        os.makedirs(output_dir, exist_ok=True)

        # first create symbols
        self.uniquify_module_names()
        for inst_shape in self.schematic_db.inst_shapes:
            self._generate_symbol_asy(inst_shape, output_dir)

        top_module_name = self.db.top_module.name
        asc_path = os.path.join(output_dir, f"{top_module_name}.asc")

        width, height = self.schematic_db.sheet_size
        width *= self.schematic_db.schematic_grid_size
        height *= self.schematic_db.schematic_grid_size
        asc_content = ["Version 4", f"Sheet 1 {width} {height}"]

        for inst_shape in self.schematic_db.inst_shapes:
            asc_content.append(self._asc_place_inst(inst_shape))

        # Process wires
        for wire_shape in self.schematic_db.net_shapes:
            asc_content.append(self._format_asc_wire(wire_shape))

        with open(asc_path, "w") as f:
            f.write("\n".join(asc_content))
        print(f"Generated schematic file: {asc_path}")
