import logging
import os
import re
import sys
from dataclasses import dataclass, field
from optparse import OptionParser
from typing import Any, Dict, List, Optional, Set, Tuple

import pyverilog.vparser.ast as vast
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.parser import parse

from schematic_from_netlist.database.netlist_database import NetlistDatabase
from schematic_from_netlist.database.netlist_structures import Bus, Instance, Module, Net, NetType, Pin, PinDirection, Port
from schematic_from_netlist.interfaces.verilog_ast_modifier import VerilogModifier, portArgInfo

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
        # First pass: Create all module objects and populate their ports and nets.
        for node in ast.description.definitions:
            if isinstance(node, vast.ModuleDef):
                module_name = self._clean_name(node.name)
                if module_name not in self.db.modules:
                    self.db.modules[module_name] = Module(name=module_name)
                if not self.db.top_module:
                    self.db.top_module = self.db.modules[module_name]

                module = self.db.modules[module_name]
                self.walk_populate_ports_and_nets(module, node)

        # Second pass: Populate the modules with instances in reverse order.
        for node in reversed(ast.description.definitions):
            if isinstance(node, vast.ModuleDef):
                module_name = self._clean_name(node.name)
                module = self.db.modules[module_name]
                self.walk_populate_instances(module, node)

    def walk_populate_ports_and_nets(self, module, node):
        """Populates an existing Module object with its ports and nets from the AST."""
        module.is_stub = False
        module.ports.clear()
        module.nets.clear()
        module.busses.clear()

        if node.portlist:
            for port in node.portlist.ports:
                info = self.modifier.extract_port_info(port)
                name, direction_str, msb, lsb = info["name"], info["direction"], info["msb"], info["lsb"]
                direction = PinDirection(direction_str) if direction_str else PinDirection.INOUT
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

        for item in node.items:
            if isinstance(item, vast.Decl):
                for decl in item.list:
                    if isinstance(decl, vast.Wire):
                        self.pop_add_net(decl, module)

    def walk_populate_instances(self, module, module_node):
        """Populates an existing Module object with its instances from the AST."""
        for item in module_node.items:
            if isinstance(item, vast.InstanceList):
                for inst in item.instances:
                    self.pop_add_instance(inst, module, module_node)
        for item in module_node.items:
            if isinstance(item, vast.GenerateStatement):
                for statement in item.items:
                    self.pop_add_generate(statement, module, module_node)
        for item in module_node.items:
            if not isinstance(item, (vast.Decl, vast.InstanceList, vast.GenerateStatement, vast.Assign)):
                log.warning(f"Parser skipping AST node: {type(item)}")

    def write_ast(self, ast, filename):
        codegen = ASTCodeGenerator()
        rslt = codegen.visit(ast)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(rslt)
        log.info(f"Reordered Verilog written to {filename}")

    def create_ast_stubs(self, ast):
        stubs = [module for module in self.db.modules.values() if module.is_stub]
        for module in stubs:
            portlist = []
            for port in module.ports.values():
                port_name = self._escape_name(port.name)
                if not port.bus:
                    vport = self.modifier.create_port(port_name, port.direction.value)
                else:
                    vport = self.modifier.create_port(port_name, port.direction.value, width=(port.bus.msb, port.bus.lsb))
                portlist.append(vport)

            vmodule = self.modifier.create_module(module_name=module.name, ports=portlist, parameters=module.parameters)
            current_defs = list(ast.description.definitions)
            current_defs.append(vmodule)
            ast.description = vast.Description(tuple(current_defs))
        return ast

    def pop_add_instance(self, vinst, module, module_node):
        module_ref_name = self._clean_name(vinst.module)
        module_ref = self.db.modules.get(module_ref_name)

        if not module_ref:
            # This case should ideally not happen if all modules are pre-created
            module_ref = self.pop_create_stub_module(module_ref_name, vinst, module_node)

        # If the module we are instantiating has not been populated yet, do it now.
        if module_ref.is_stub:
            # Find the AST node for the module_ref and populate it
            ref_node = next(
                (
                    n
                    for n in self.modifier.ast.description.definitions
                    if isinstance(n, vast.ModuleDef) and self._clean_name(n.name) == module_ref_name
                ),
                None,
            )
            if ref_node:
                self.walk_populate_ports_and_nets(module_ref, ref_node)
            else:
                # It's a true stub (missing definition), so we just use the stub object
                pass

        inst = module.add_instance(vinst.name, module=module_ref, module_ref=module_ref.name)

        if vinst.parameterlist:
            for param_arg in vinst.parameterlist:
                param_name = self._clean_name(param_arg.paramname)
                param_value = self._get_param_value(param_arg.argname)
                inst.parameters[param_name] = param_value

        for i, port_arg in enumerate(vinst.portlist):
            info = self.modifier.extract_portarg_with_width(port_arg, module_node)
            port_info = None
            port_name = self._clean_name(info.port_name)

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
                pin_name = self._clean_name(port_pin)
                net_name = signal_pin
                if "1'b0" in signal_pin:
                    net_name = "GND"
                elif "1'b1" in signal_pin:
                    net_name = "VDD"

                net = module.nets.get(net_name)
                if not net:
                    if net_name is None:
                        log.error(f"Attempted to create a net with a None name for port '{pin_name}' on instance '{inst.name}'")
                        continue
                    net = module.add_net(net_name)
                    if net_name == "GND":
                        net.net_type = NetType.SUPPLY0
                    elif net_name == "VDD":
                        net.net_type = NetType.SUPPLY1

                pin = inst.pins.get(pin_name)
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

    def pop_add_generate(self, statement, module, module_node):
        """Handles Verilog generate blocks, focusing on for-loops."""
        if isinstance(statement, vast.ForStatement):
            try:
                # Correctly parse for-loop syntax: for (pre; cond; post)
                loop_var = statement.pre.left.var.name
                start = int(statement.pre.right.var.value)
                # Assuming condition is simple 'less than'
                end = int(statement.cond.right.value)
                # Assuming step is 'i = i + 1' by checking the post statement
                step = 1

                log.info(f"Unrolling generate for-loop: var={loop_var}, from {start} to {end - 1}")

                for i in range(start, end, step):
                    substitutions = {loop_var: i}
                    block = statement.statement
                    block_name = block.scope

                    for block_item in block.statements:
                        concrete_item = self.modifier.substitute_genvar(block_item, substitutions)
                        if isinstance(concrete_item, vast.InstanceList):
                            for inst in concrete_item.instances:
                                # Prepend the generate block name to the instance name
                                original_name = inst.name
                                inst.name = f"{block_name}[{i}]/{original_name}"
                                self.pop_add_instance(inst, module, module_node)

            except Exception as e:
                log.error(f"Failed to parse generate for-loop: {e}")
                log.warning(f"Skipping complex generate block: {statement.show()}")
        else:
            log.warning(f"Unsupported generate construct of type {type(statement)} found. Skipping.")

    def pop_create_stub_module(self, module_ref_name, vinst, module_node):
        module_ref = Module(name=module_ref_name)
        self.db.modules[module_ref_name] = module_ref
        module_ref.is_stub = True

        if vinst.parameterlist:
            for param_arg in vinst.parameterlist:
                param_name = self._clean_name(param_arg.paramname)
                param_value = self._get_param_value(param_arg.argname)
                module_ref.parameters[param_name] = param_value

        for i, port_arg in enumerate(vinst.portlist):
            info = self.modifier.extract_portarg_with_width(port_arg, module_node)
            port_name = info.port_name or f"p{i}"
            port_name = self._clean_name(port_name)
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

    def _escape_name(self, name):
        if isinstance(name, str):
            # if name has special chars, add a backslash to front
            if re.search(r"[^a-zA-Z0-9_]", name):
                name = f"\\{name}"
        return name

    def _get_param_value(self, argname):
        if isinstance(argname, vast.StringConst):
            return argname.value.strip('"')  # Remove the quotes
        elif isinstance(argname, vast.IntConst):
            return argname.value
        elif isinstance(argname, vast.Identifier):
            return argname.name
        else:
            return self.get_signal_name(argname)

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
