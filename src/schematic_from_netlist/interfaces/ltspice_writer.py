import os


class LTSpiceWriter:
    def __init__(self, db, schematic_db):
        self.db = db
        self.schematic_db = schematic_db
        self.module_names = {}

    def _asc_place_inst(self, inst_shape):
        """Formats a single symbol line for the .asc file."""
        out = ""
        name = inst_shape.name
        module_ref = inst_shape.module_ref
        x = (inst_shape.rect[0] + inst_shape.rect[2]) // 2
        y = (inst_shape.rect[1] + inst_shape.rect[3]) // 2
        out += f"SYMBOL {module_ref} {x} {y} R0\n"
        out += f"SYMATTR InstName {name}\n"
        out += f"SYMATTR Value {module_ref}\n"
        return out

    def _format_asc_wire(self, wire_shape):
        """Formats a single WIRE line for the .asc file."""
        # out = "* WIRE\n"
        out = ""
        pt_start = wire_shape.points[0]
        for pt in wire_shape.points[1:]:
            pt_end = pt
            out += f"WIRE {pt_start[0]} {pt_start[1]} {pt_end[0]} {pt_end[1]}\n"
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
            return "LEFT"
        elif min_dist == dist_right:
            return "RIGHT"
        elif min_dist == dist_top:
            return "TOP"
        elif min_dist == dist_bottom:
            return "BOTTOM"

        return "TOP"

    def _generate_symbol_asy(self, inst_shape, output_dir="data/ltspice"):
        """Generates an .asy file for a given module."""

        rect = inst_shape.rect
        hw = (rect[2] - rect[0]) // 2  # half width
        hh = (rect[3] - rect[1]) // 2  # half height

        inst_offset_x = (rect[2] + rect[0]) // 2
        inst_offset_y = (rect[3] + rect[1]) // 2

        cell_rect = [-hw, -hh, hw, hh]

        asy = "Version 4\n"
        asy += "SymbolType BLOCK\n"
        asy += f"RECTANGLE NORMAL {-hw} {-hh} {hw} {hh}\n"

        for shape in inst_shape.port_shapes:
            rect = shape.rect
            port_center_x = (rect[2] + rect[0]) // 2
            port_center_y = (rect[3] + rect[1]) // 2
            x = port_center_x - inst_offset_x
            y = port_center_y - inst_offset_y
            pt = [x, y]
            side = self._get_pin_side(pt, cell_rect)
            # breakpoint()
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

        sheet_size = self.schematic_db.sheet_size
        asc_content = ["Version 4", f"Sheet 1 {sheet_size[0]} {sheet_size[1]}"]

        for inst_shape in self.schematic_db.inst_shapes:
            asc_content.append(self._asc_place_inst(inst_shape))

        # Process wires
        for wire_shape in self.schematic_db.net_shapes:
            asc_content.append(self._format_asc_wire(wire_shape))

        with open(asc_path, "w") as f:
            f.write("\n".join(asc_content))
        print(f"Generated schematic file: {asc_path}")
