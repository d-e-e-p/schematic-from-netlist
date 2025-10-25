import logging as log
import os
from enum import Enum

from .symbol_library import SymbolLibrary


class LineType(Enum):
    NORMAL = "Normal"
    WIDE = "WIDE"
    THIN = "Thin"
    THICK = "Thick"
    DASHED = "Dashed"
    DOT = "Dot"
    DOT_DASHED = "DotDashed"


# kicad only supports 0 to 4
class LineColor(Enum):
    WHITE = 0
    BLACK = 1
    RED = 2
    ORANGE = 3
    YELLOW = 4
    GREEN = 5
    CYAN = 6
    BLUE = 7


class LTSpiceWriter:
    def __init__(self, db):
        self.db = db
        self.schematic_db = db.schematic_db
        self.module_names = {}
        self.add_comments = False  # kicad can't seem to parse spice with comments
        self.output_dir = "output/ltspice"
        self.symlib = SymbolLibrary()

    def upscale_rect(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Scale a rectangle (x1,y1,x2,y2) from grid units to real units."""
        g = self.schematic_db.schematic_grid_size
        return (rect[0] * g, rect[1] * g, rect[2] * g, rect[3] * g)

    def upscale_segments(
        self, segments: list[tuple[tuple[int, int], tuple[int, int]]]
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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

    def asc_place_inst(self, inst, xoffset=0, yoffset=0):
        """Formats a single symbol line for the .asc file."""
        out = ""

        if not inst.draw.shape:
            return out

        name = inst.name
        module_ref = inst.module_ref_uniq
        x1, y1, x2, y2 = self.upscale_rect(inst.draw.shape)
        x = (x1 + x2) // 2 + xoffset
        y = (y1 + y2) // 2 + yoffset
        out += f"SYMBOL {module_ref} {x} {y} {inst.draw.orient}\n"
        out += f"SYMATTR InstName {name}\n"
        out += f"SYMATTR Value {module_ref}\n"
        return out

    def asc_place_module(self, inst, xoffset=0, yoffset=0):
        """Formats a module with other instances in .asc file."""
        out = ""

        if not inst.draw.shape:
            return out

        x1, y1, x2, y2 = self.upscale_rect(inst.draw.shape)
        (x1, x2) = (x1 + xoffset, x2 + xoffset)
        (y1, y2) = (y1 + yoffset, y2 + yoffset)

        out += f"TEXT {x2} {y2} Center 2 ;{inst.name} ({inst.module_ref})\n"
        out += f"RECTANGLE NORMAL {x1} {y1} {x2} {y2}\n"

        for inst_child in inst.module.instances.values():
            if inst_child.module.is_leaf:
                out += self.asc_place_inst(inst_child, x1, y1)
            else:
                out += self.asc_place_module(inst_child, x1, y1)
        return out

    def format_asc_wire(self, net):
        """Formats WIRE and FLAG lines for the .asc file."""

        if not net.draw.shape:
            return ""

        def segments_to_wire(out, segments, netname, comment, add_label):
            for pt_start, pt_end in segments:
                out += f"WIRE {pt_start[0]} {pt_start[1]} {pt_end[0]} {pt_end[1]}{comment}\n"
                if add_label:
                    pt_mid = (
                        (pt_start[0] + pt_end[0]) // 2,
                        (pt_start[1] + pt_end[1]) // 2,
                    )
                    out += f"FLAG {pt_mid[0]} {pt_mid[1]} {netname}\n"
            return out

        comment = f" $   {net.name}" if self.add_comments else ""
        out = ""

        # Main net segments (with label)
        out = segments_to_wire(out, self.upscale_segments(net.draw.shape), net.name, comment, add_label=True)
        # Patch points (without label)
        out = segments_to_wire(out, self.upscale_segments(net.draw.buffer_patch_points), net.name, comment, add_label=False)

        return out

    def format_asc_cluster_outline(self, cluster, line_type: LineType = LineType.WIDE, line_color: LineColor = LineColor.RED):
        """Formats a RECTANGLE line for the cluster outline."""
        if not cluster.shape:
            return ""

        x1, y1, x2, y2 = self.upscale_rect(cluster.shape)

        return f"RECTANGLE {line_type.value} {x1} {y1} {x2} {y2} {line_color.value}\n"

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
        font_size = 1
        asy = ""
        asy += f"WINDOW 0 {ref_x} {ref_y} Left {font_size}\n"  # InstName (Reference)
        asy += f"WINDOW 3 {val_x} {val_y} Left {font_size}\n"  # Value
        asy += f"WINDOW 38 {simd_x} {simd_y} Left {font_size}\n"  # Sim.Device
        asy += f"WINDOW 39 {simp_x} {simp_y} Left {font_size}\n"  # Sim.Params

        return asy

    def generate_symbol_asy(self, inst):
        """Generates an .asy file for a given module."""

        # eg a cap connected between vdd and gnd
        if not inst.draw.shape:
            return
        rect = self.upscale_rect(inst.draw.shape)
        hw = (rect[2] - rect[0]) // 2  # half width
        hh = (rect[3] - rect[1]) // 2  # half height

        inst_offset_x = (rect[2] + rect[0]) // 2
        inst_offset_y = (rect[3] + rect[1]) // 2

        cell_rect = [-hw, -hh, hw, hh]

        asy = "Version 4\n"
        asy += "SymbolType BLOCK\n"
        asy += f"RECTANGLE NORMAL {-hw} {-hh} {hw} {hh}\n"
        asy += f"SYMATTR SpiceModel {inst.module_ref_uniq}\n"  # This sets Sim.Device
        asy += f"SYMATTR SpiceLine -\n"  # This sets Sim.Device
        asy += self.generate_symbol_window_commands(hw, hh)
        for pinname, pin in inst.pins.items():
            if not pin.draw.shape:
                continue
            port_center_x, port_center_y = self.upscale_point(pin.draw.shape)
            from_router = True
            if from_router:
                x = port_center_x - inst_offset_x
                y = port_center_y - inst_offset_y
            else:
                x = port_center_x - hw
                y = port_center_y - hh
            pt = (x, y)
            side = self._get_pin_side(pt, cell_rect)
            asy += f"PIN {x} {y} {side} 0\n"
            asy += f"PINATTR PinName {pinname}\n"

        asy_path = os.path.join(self.output_dir, f"{inst.module_ref_uniq}.asy")
        with open(asy_path, "w") as f:
            f.write(asy)
        log.debug(f"Generated symbol file: {asy_path}")

    def produce_schematic(self, output_dir="data/ltspice"):
        """Generates the .asc schematic and .asy symbol files."""
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # first create symbols
        self.db.uniquify_module_names()
        for module in self.db.design.modules.values():
            for inst in module.instances.values():
                if inst.module.is_leaf:
                    if inst.module.name in self.symlib.symbols:
                        self.symlib.generate_symbol_asy(inst.module.name, output_dir)
                        inst.module_ref_uniq = inst.module_ref
                    else:
                        self.generate_symbol_asy(inst)

        top_module_name = self.db.design.top_module.name
        asc_path = os.path.join(output_dir, f"{top_module_name}.asc")

        width, height = self.schematic_db.sheet_size
        width *= self.schematic_db.schematic_grid_size
        height *= self.schematic_db.schematic_grid_size
        asc_content = ["Version 4", f"Sheet 1 {width} {height}"]

        for inst in self.db.design.top_module.instances.values():
            if inst.module.is_leaf:
                asc_content.append(self.asc_place_inst(inst))
            else:
                asc_content.append(self.asc_place_module(inst))

        # Process wires
        for net in self.db.design.top_module.nets.values():
            if not net.name:
                breakpoint()
            asc_content.append(self.format_asc_wire(net))

        with open(asc_path, "w") as f:
            f.write("\n".join(asc_content))
        log.info(f"Generated schematic file: {asc_path}")
