import logging
import os
import sys
from dataclasses import dataclass, field
from optparse import OptionParser
from typing import Any, Dict, List, Optional, Set, Tuple

import pyverilog.vparser.ast as vast
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.parser import parse

from schematic_from_netlist.interfaces.netlist_database import NetlistDatabase
from schematic_from_netlist.interfaces.netlist_structures import (
    Bus,
    Instance,
    Module,
    Net,
    NetType,
    Pin,
    PinDirection,
    Port,
)
from schematic_from_netlist.interfaces.verilog_ast_modifier import (
    VerilogModifier,
    portArgInfo,
)

log = logging.getLogger(__name__)



class VerilogParser:
    def __init__(self):
        self.db = NetlistDatabase()
        self.top_module_name = None
        self.modifier = VerilogModifier("")

    # --------------------------------------------------------------------------
    # Main Public API
    # --------------------------------------------------------------------------

    def parse_and_store(self, filename):
        ast, _ = parse([filename])
        self.modifier = VerilogModifier(ast)
        self.walker(ast)
        self.create_ast_stubs(ast)
        if self.db.top_module:
            output_filename = f"data/verilog/processed_{self.db.top_module.name}.v"
            self.write_ast(ast, output_filename)
        return self.db

    # --------------------------------------------------------------------------
    # Core Processing Methods
    # --------------------------------------------------------------------------

    def walker(self, ast):
        # First pass: Collect module definitions
        for node in ast.description.definitions:
            if isinstance(node, vast.ModuleDef):
                self.walk_add_module(node)

        # Second pass: Populate modules with instances etc
        for node in ast.description.definitions:
            if isinstance(node, vast.ModuleDef):
                module_name = self._clean_name(node.name)
                module = self.db.modules[module_name]
                self.walk_populate_module(module, node)

    def create_ast_stubs(self, ast):
        stubs = [module for module in self.db.modules.values() if module.is_stub]
        for module in stubs:
            portlist = []
            for port in module.ports.values():
                if not port.bus:
                    vport = self.modifier.create_port(port.name, port.direction.value)
                else:
                    vport = self.modifier.create_port(
                        port.name, port.direction.value, width=(port.bus.msb, port.bus.lsb)
                    )
                portlist.append(vport)
            # Assuming parameters are stored somehow, not fully implemented in stub creation yet
            vmodule = self.modifier.create_module(module_name=module.name, ports=portlist, parameters=[])
            current_defs = list(ast.description.definitions)
            current_defs.append(vmodule)
            ast.description = vast.Description(tuple(current_defs))
        return ast

    def write_ast(self, ast, filename):
        codegen = ASTCodeGenerator()
        rslt = codegen.visit(ast)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(rslt)
        log.info(f"Reordered Verilog written to {filename}")

    # --------------------------------------------------------------------------
    # Walker & Population Methods
    # --------------------------------------------------------------------------

    def walk_add_module(self, node):
        module_name = self._clean_name(node.name)
        module = self.db.modules.get(module_name)
        if not module:
            module = Module(name=module_name)
            self.db.modules[module_name] = module

        module.is_stub = False

        if not self.db.top_module:
            self.db.top_module = module

        module.ports.clear()
        module.nets.clear()
        module.busses.clear()

        if node.portlist:
            for port in node.portlist.ports:
                info = self.modifier.extract_port_info(port)
                name, direction_str, msb, lsb = info["name"], info["direction"], info["msb"], info["lsb"]
                direction = PinDirection(direction_str)
                port_obj = module.add_port(name, direction)

                if msb is not None:
                    msb, lsb = int(msb), int(lsb)
                    bus = Bus(name=name, msb=msb, lsb=lsb)
                    module.busses[name] = bus
                    port_obj.bus = bus
                    for i in range(lsb, msb + 1):
                        net_name = f"{name}[{i}]"
                        net = module.add_net(net_name)
                        net.bus = bus
                else:
                    module.add_net(name)

    def walk_populate_module(self, module, module_node):
        for item in module_node.items:
            if isinstance(item, vast.Decl):
                for decl in item.list:
                    if isinstance(decl, vast.Wire):
                        self.pop_add_net(decl, module)
        for item in module_node.items:
            if isinstance(item, vast.InstanceList):
                for inst in item.instances:
                    self.pop_add_instance(inst, module, module_node)
        for item in module_node.items:
            if isinstance(item, vast.GenerateStatement):
                for statement in item.items:
                    self.pop_add_generate(statement, module, module_node)
        for item in module_node.items:
            if not isinstance(
                item, (vast.Decl, vast.InstanceList, vast.GenerateStatement, vast.Assign)
            ):
                log.warning(f"Parser skipping AST node: {type(item)}")

    def pop_add_instance(self, vinst, module, module_node):
        module_ref_name = self._clean_name(vinst.module)
        module_ref = self.db.modules.get(module_ref_name)
        if not module_ref:
            module_ref = self.pop_create_stub_module(module_ref_name, vinst, module_node)

        inst = module.add_instance(vinst.name, module=module_ref, module_ref=module_ref.name)

        if vinst.parameterlist:
            for param_arg in vinst.parameterlist:
                param_name = self._clean_name(param_arg.paramname)
                inst.parameters[param_name] = "PARAM_VALUE"  # Placeholder

        for i, port_arg in enumerate(vinst.portlist):
            info = self.modifier.extract_portarg_with_width(port_arg, module_node)
            port_info = None
            port_name = info.port_name

            if port_name:
                port_info = module_ref.ports.get(port_name)
            else:
                if i < len(module_ref.ports):
                    port_info = list(module_ref.ports.values())[i]
                    info.port_name = port_info.name
                else:
                    log.error(f"Positional connection index {i} out of bounds for '{module_ref.name}'")
                    continue

            if not port_info:
                log.error(f"Port '{port_name}' not found in module '{module_ref.name}'")
                continue

            mapping = self._get_bit_mapping(info, port_info)
            for port_pin, signal_pin in mapping:
                net_name = signal_pin
                # Map constants to VDD/GND
                if "1'b0" in signal_pin:
                    net_name = "GND"
                elif "1'b1" in signal_pin:
                    net_name = "VDD"

                net = module.nets.get(net_name)
                if not net:
                    net = module.add_net(net_name)
                    if net_name == "GND":
                        net.net_type = NetType.SUPPLY0
                    elif net_name == "VDD":
                        net.net_type = NetType.SUPPLY1

                pin = inst.pins.get(port_pin)
                if pin:
                    net.add_pin(pin)
                else:
                    log.error(f"Pin '{port_pin}' not found on instance '{inst.name}'")

    def pop_add_net(self, decl, module: Module):
        name = self._clean_name(decl.name)
        if decl.width:
            msb = int(decl.width.msb.value)
            lsb = int(decl.width.lsb.value)
            bus = Bus(name=name, msb=msb, lsb=lsb)
            module.busses[name] = bus
            for i in range(lsb, msb + 1):
                net_name = f"{name}[{i}]"
                net = module.add_net(net_name)
                net.bus = bus
        else:
            module.add_net(name)

    def pop_add_generate(self, statement, module, node):
        log.info(f"Found generate statement: {statement.show()}")

    def pop_create_stub_module(self, module_ref_name, vinst, module_node):
        module_ref = Module(name=module_ref_name)
        self.db.modules[module_ref_name] = module_ref
        module_ref.is_stub = True

        if vinst.parameterlist:
            for param_arg in vinst.parameterlist:
                param_name = self._clean_name(param_arg.paramname)
                # Parameters are not fully handled for stubs yet
        for i, port_arg in enumerate(vinst.portlist):
            info = self.modifier.extract_portarg_with_width(port_arg, module_node)
            port_name = info.port_name or f"p{i}"
            port = module_ref.add_port(port_name, PinDirection.INOUT)
            if info.total_connection_width > 1:
                msb, lsb = info.total_connection_width - 1, 0
                bus = Bus(name=port_name, msb=msb, lsb=lsb)
                module_ref.busses[port_name] = bus
                port.bus = bus
        return module_ref

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _clean_name(self, name):
        if isinstance(name, str):
            return name.replace("\\", "")
        return name

    def get_signal_name(self, signal):
        if isinstance(signal, vast.Identifier):
            return self._clean_name(signal.name)
        elif isinstance(signal, vast.Pointer):
            var_name = self.get_signal_name(signal.var)
            ptr_str = self.get_signal_name(signal.ptr)
            return f"{var_name}[{ptr_str}]"
        elif isinstance(signal, vast.Partselect):
            var_name = self.get_signal_name(signal.var)
            msb = self.get_signal_name(signal.msb) if signal.msb else ""
            lsb = self.get_signal_name(signal.lsb) if signal.lsb else ""
            return f"{var_name}[{msb}:{lsb}]"
        elif isinstance(signal, vast.Concat):
            parts = [self.get_signal_name(part) for part in signal.list]
            return "{" + ",".join(parts) + "}"
        elif isinstance(signal, vast.IntConst):
            return str(signal.value)
        else:
            return str(signal)

    def _get_bit_mapping(self, info: "portArgInfo", port_info: "Port") -> List[Tuple[str, str]]:
        port_name = info.port_name
        port_width = abs(port_info.bus.msb - port_info.bus.lsb) + 1 if port_info.bus else 1

        signal_bits = []
        for conn in info.connections:
            msb = conn.select_msb if conn.select_msb is not None else conn.signal_msb
            lsb = conn.select_lsb if conn.select_lsb is not None else conn.signal_lsb

            if msb is None and lsb is None:
                signal_bits.append(conn.signal_name)
                continue
            try:
                msb, lsb = int(msb), int(lsb)
                if msb >= lsb:
                    for i in range(msb, lsb - 1, -1):
                        signal_bits.append(f"{conn.signal_name}[{i}]")
                else:
                    for i in range(lsb, msb + 1):
                        signal_bits.append(f"{conn.signal_name}[{i}]")
            except (ValueError, TypeError):
                signal_bits.append(f"{conn.signal_name}[{msb}:{lsb}]")

        if len(signal_bits) != port_width:
            log.warning(f"Width mismatch for port '{port_name}': port is {port_width}, signal is {len(signal_bits)}")
            return [(port_name, "{" + ",".join(signal_bits) + "}")]

        port_pins = []
        if port_width == 1:
            port_pins.append(port_name)
        else:
            if port_info.bus.msb >= port_info.bus.lsb:
                for i in range(port_info.bus.msb, port_info.bus.lsb - 1, -1):
                    port_pins.append(f"{port_name}[{i}]")
            else:
                for i in range(port_info.bus.lsb, port_info.bus.msb + 1):
                    port_pins.append(f"{port_name}[{i}]")

        return list(zip(port_pins, signal_bits))




# Usage Example
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    # Create test file with top-first order and a missing module
    with open("design.sv", "w") as f:
        f.write("""
module foo(input clk, output [7:0] out);
    wire [7:0] cpu_out;
    wire [7:0] mem_out;

    cpu u_cpu(
        .clk(clk),
        .data_in(mem_out),
        .data_out(cpu_out)
    );

    memory u_mem(
        .clk(clk),
        .addr(cpu_out),
        .data_out(mem_out)
    );

    assign out = cpu_out;
endmodule

module cpu(input clk, input [7:0] data_in, output [7:0] data_out);
    wire [7:0] alu_result;
    wire [7:0] reg_result;

    alu u_alu(
        .a(data_in),
        .b(reg_result),
        .out(alu_result)
    );

    reg_file u_regs(
        .clk(clk),
        .d_in(alu_result),
        .d_out(reg_result)
    );

    assign data_out = alu_result;
endmodule

module alu(input [7:0] a, input [7:0] b, output [7:0] out);
    adder u_adder(.a(a), .b(b), .sum(out));
endmodule

module memory(input clk, input [7:0] addr, output [7:0] data_out);
    // This module will be a stub
    ram_block u_ram(
        .clk(clk),
        .addr(addr),
        .data(data_out)
    );
endmodule

module adder(input [7:0] a, input [7:0] b, output [7:0] sum);
    assign sum = a + b;
endmodule

""")

    print("=== Using AST-Based Reordering ===\n")

    parser = VerilogParser()
    try:
        output_filename = parser.parse_and_create_stubs("design.sv")
        print("\n Success! Output written to:", output_filename)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
