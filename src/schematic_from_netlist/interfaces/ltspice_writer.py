import os
from schematic_from_netlist.interfaces.netlist_database import Module

class LTSpiceWriter:
    def __init__(self, db, schematic_db):
        self.db = db
        self.schematic_db = schematic_db
        self.scale = 80  # Scale factor from graphviz points to LTspice coordinates

    def _format_asc_symbol(self, inst_shape):
        """Formats a single symbol line for the .asc file."""
        inst = inst_shape.inst
        # Use the center of the instance's bounding box for placement
        x = int(((inst_shape.rect[0] + inst_shape.rect[2]) / 2) * self.scale)
        y = int(((inst_shape.rect[1] + inst_shape.rect[3]) / 2) * self.scale)
        return f"SYMBOL {inst.module_ref} {x} {y} R0"

    def _format_asc_instname(self, inst_shape):
        """Formats the SYMATTR InstName line."""
        inst = inst_shape.inst
        x = int(inst_shape.rect[0] * self.scale)
        y = int(inst_shape.rect[3] * self.scale) + 8 # Place above the symbol
        return f"SYMATTR InstName {inst.name}"

    def _format_asc_wire(self, net_shape):
        """Formats a single WIRE line for the .asc file."""
        # For simplicity, we draw a line between the first and last points of the net
        x1 = int(net_shape.points[0][0])
        y1 = int(net_shape.points[0][1])
        x2 = int(net_shape.points[-1][0])
        y2 = int(net_shape.points[-1][1])
        return f"WIRE {x1} {y1} {x2} {y2}"

    def _generate_symbol_asy(self, inst_shape, asy_path: str):
        """Generates an .asy file for a given module based on an instance shape."""
        if os.path.exists(asy_path):
            return  # Symbol already generated

        asy_content = ["Version 4", "SymbolType BLOCK"]
        
        inst = inst_shape.inst
        module = self.db.modules.get(inst.module_ref)
        if not module:
            return

        # Calculate symbol origin (center of the instance bounding box)
        origin_x = (inst_shape.rect[0] + inst_shape.rect[2]) / 2
        origin_y = (inst_shape.rect[1] + inst_shape.rect[3]) / 2

        # Define symbol body relative to the origin
        width = (inst_shape.rect[2] - inst_shape.rect[0]) * self.scale
        height = (inst_shape.rect[3] - inst_shape.rect[1]) * self.scale
        asy_content.append(f"RECTANGLE NORMAL {-width//2} {-height//2} {width//2} {height//2}")

        # Add pins with relative positions
        for i, port_shape in enumerate(inst_shape.port_shapes):
            port_rect = port_shape.rect
            port_center_x = (port_rect[0] + port_rect[2]) / 2
            port_center_y = (port_rect[1] + port_rect[3]) / 2
            
            rel_x = int((port_center_x - origin_x) * self.scale)
            rel_y = int((port_center_y - origin_y) * self.scale)

            # Determine pin justification based on relative position
            if abs(rel_x) > abs(rel_y):
                direction = "LEFT" if rel_x < 0 else "RIGHT"
            else:
                direction = "BOTTOM" if rel_y < 0 else "TOP"

            asy_content.append(f"PIN {rel_x} {rel_y} {direction} 8")
            asy_content.append(f"PINATTR PinName {port_shape.pin.name}")
            asy_content.append(f"PINATTR SpiceOrder {i + 1}")

        asy_content.append(f"SYMATTR InstName U?")
        asy_content.append(f"SYMATTR Prefix X")
        asy_content.append(f"SYMATTR Value {module.name}")

        with open(asy_path, "w") as f:
            f.write("\n".join(asy_content))
        print(f"Generated symbol file: {asy_path}")

    def produce_schematic(self, output_dir="data/ltspice"):
        """Generates the .asc schematic and .asy symbol files."""
        os.makedirs(output_dir, exist_ok=True)
        
        top_module_name = self.db.top_module.name
        asc_path = os.path.join(output_dir, f"{top_module_name}.asc")
        
        asc_content = ["Version 4", f"Sheet 1 1200 800"]

        # Group instance shapes by module type to find a template for each symbol
        module_ref_to_inst_shape = {}
        for inst_shape in self.schematic_db.inst_shapes:
            if inst_shape.inst.module_ref not in module_ref_to_inst_shape:
                module_ref_to_inst_shape[inst_shape.inst.module_ref] = inst_shape

        # Generate symbol for each unique module
        for module_ref, inst_shape in module_ref_to_inst_shape.items():
            asy_path = os.path.join(output_dir, f"{module_ref}.asy")
            self._generate_symbol_asy(inst_shape, asy_path)

        # Process instances for the schematic
        for inst_shape in self.schematic_db.inst_shapes:
            asc_content.append(self._format_asc_symbol(inst_shape))
            asc_content.append(self._format_asc_instname(inst_shape))

        # Process wires
        for net_shape in self.schematic_db.net_shapes:
            asc_content.append(self._format_asc_wire(net_shape))

        with open(asc_path, "w") as f:
            f.write("\n".join(asc_content))
        print(f"Generated schematic file: {asc_path}")
        self.scale = 80  # Scale factor from graphviz points to LTspice coordinates

    def _format_asc_symbol(self, inst_shape):
        """Formats a single symbol line for the .asc file."""
        inst = inst_shape.inst
        x = int(inst_shape.rect[0] * self.scale)
        y = int(inst_shape.rect[1] * self.scale)
        return f"SYMBOL {inst.module_ref} {x} {y} R0"

    def _format_asc_instname(self, inst_shape):
        """Formats the SYMATTR InstName line."""
        inst = inst_shape.inst
        x = int(inst_shape.rect[0] * self.scale)
        y = int(inst_shape.rect[1] * self.scale) + 32 # Offset for visibility
        return f"SYMATTR InstName {inst.name}"

    def _format_asc_wire(self, wire_shape):
        """Formats a single WIRE line for the .asc file."""
        x1 = int(wire_shape.points[0][0] * self.scale)
        y1 = int(wire_shape.points[0][1] * self.scale)
        x2 = int(wire_shape.points[1][0] * self.scale)
        y2 = int(wire_shape.points[1][1] * self.scale)
        return f"WIRE {x1} {y1} {x2} {y2}"

    def _generate_symbol_asy(self, module: Module, asy_path: str):
        """Generates an .asy file for a given module."""
        if os.path.exists(asy_path):
            return  # Symbol already generated

        asy_content = ["Version 4", "SymbolType BLOCK"]

        # Define a simple rectangular symbol body
        pin_count = len(module.ports)
        height = max(80, pin_count * 16)
        width = 128
        asy_content.append(f"RECTANGLE NORMAL {-width//2} {-height//2} {width//2} {height//2}")

        # Add pins
        input_pins = sorted([p for p in module.ports.values() if p.direction.name == 'INPUT'], key=lambda p: p.name)
        output_pins = sorted([p for p in module.ports.values() if p.direction.name == 'OUTPUT'], key=lambda p: p.name)
        inout_pins = sorted([p for p in module.ports.values() if p.direction.name == 'INOUT'], key=lambda p: p.name)

        pin_spacing = 32
        current_y = (len(input_pins) -1) * pin_spacing // 2
        for pin in input_pins:
            asy_content.append(f"PIN {-width//2} {current_y} LEFT 8")
            asy_content.append(f"PINATTR PinName {pin.name}")
            asy_content.append(f"PINATTR SpiceOrder {len(asy_content) // 3}") # Simple SpiceOrder
            current_y -= pin_spacing
        
        current_y = (len(output_pins) -1) * pin_spacing // 2
        for pin in output_pins:
            asy_content.append(f"PIN {width//2} {current_y} RIGHT 8")
            asy_content.append(f"PINATTR PinName {pin.name}")
            asy_content.append(f"PINATTR SpiceOrder {len(asy_content) // 3}")
            current_y -= pin_spacing

        # Add attributes
        asy_content.append(f"SYMATTR InstName U?")
        asy_content.append(f"SYMATTR Prefix X")
        asy_content.append(f"SYMATTR Value {module.name}")

        with open(asy_path, "w") as f:
            f.write("\n".join(asy_content))
        print(f"Generated symbol file: {asy_path}")

    def produce_schematic(self, output_dir="data/ltspice"):
        """Generates the .asc schematic and .asy symbol files."""
        os.makedirs(output_dir, exist_ok=True)
        
        top_module_name = self.db.top_module.name
        asc_path = os.path.join(output_dir, f"{top_module_name}.asc")
        
        asc_content = ["Version 4", f"Sheet 1 880 680"]

        # Process instances and generate symbols
        generated_symbols = set()
        for inst_shape in self.schematic_db.inst_shapes:
            asc_content.append(self._format_asc_symbol(inst_shape))
            asc_content.append(self._format_asc_instname(inst_shape))
            
            module_ref = inst_shape.inst.module_ref
            if module_ref not in generated_symbols:
                module = self.db.modules.get(module_ref)
                if module:
                    asy_path = os.path.join(output_dir, f"{module_ref}.asy")
                    self._generate_symbol_asy(module, asy_path)
                    generated_symbols.add(module_ref)

        # Process wires
        for wire_shape in self.schematic_db.net_shapes:
            asc_content.append(self._format_asc_wire(wire_shape))

        with open(asc_path, "w") as f:
            f.write("\n".join(asc_content))
        print(f"Generated schematic file: {asc_path}")
