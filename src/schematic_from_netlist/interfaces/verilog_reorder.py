import logging
import os
import sys
from dataclasses import dataclass, field
from optparse import OptionParser
from typing import Any, Dict, List, Optional, Set, Tuple

import pyverilog.vparser.ast as vast
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.parser import parse

from schematic_from_netlist.interfaces.verilog_ast_modifier import VerilogModifier

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Port:
    name: str
    direction: str
    msb: int = 0
    lsb: int = 0


@dataclass(frozen=True)
class Pin:
    name: str
    msb: int = 0
    lsb: int = 0


@dataclass(frozen=True)
class Instance:
    name: str
    module_ref: str
    pins: Set["Pin"] = field(default_factory=set, compare=False, hash=False)

    def add_pin(self, name, msb=0, lsb=0):
        pin = Pin(name, msb, lsb)
        self.pins.add(pin)


@dataclass
class Module:
    name: str = ""
    instances: Set[Instance] = field(default_factory=set)
    ports: Set[Port] = field(default_factory=set)
    level: int = 0
    is_stub: bool = False

    def add_port(self, name, direction, msb=0, lsb=0):
        port = Port(name, direction, msb, lsb)
        self.ports.add(port)

    def add_instance(self, name, module_ref):
        instance = Instance(name, module_ref)
        self.instances.add(instance)
        return instance


class VerilogReorder:
    def __init__(self):
        self.modules = {}
        self.top_module_name = None
        self.modifier = VerilogModifier("")

    def _clean_name(self, name):
        """Strip backslashes from names."""
        if isinstance(name, str):
            return name.replace("\\", "")
        return name

    def parse_and_create_stubs(self, filename):
        ast, _ = parse([filename])
        self.modifier = VerilogModifier(ast)
        self.walker(ast)
        self.create_stubs(ast)
        output_filename = f"data/verilog/processed_{self.top_module_name}.v"
        self.write_ast(ast, output_filename)
        return output_filename

    def create_stubs(self, ast):
        stubs = [module for module in self.modules.values() if module.is_stub]
        for module in stubs:
            portlist = []
            for port in module.ports:
                if port.msb == 0 and port.lsb == 0:
                    vport = self.modifier.create_port(port.name, port.direction)
                else:
                    vport = self.modifier.create_port(port.name, port.direction, width=(port.msb, port.lsb))
                portlist.append(vport)
            vmodule = self.modifier.create_module(module_name=module.name, ports=portlist)
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

    def get_signal_name(self, signal):
        """Extract signal name from various AST node types"""
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

    def walker(self, ast):
        # First pass: Collect module definitions
        for node in ast.description.definitions:
            if isinstance(node, vast.ModuleDef):
                module_name = self._clean_name(node.name)
                module = Module(name=module_name)
                self.modules[module_name] = module
                if not self.top_module_name:
                    self.top_module_name = module_name

                # Collect ports
                if node.portlist:
                    for port in node.portlist.ports:
                        info = self.modifier.extract_port_info(port)
                        module.add_port(info["name"], info["direction"], info["msb"], info["lsb"])

        # Second pass: Populate modules with instances, nets, and connections
        for node in ast.description.definitions:
            if isinstance(node, vast.ModuleDef):
                module_name = self._clean_name(node.name)
                module = self.modules[module_name]
                self.populate_module(module, node)

        # Third pass: Create stub modules for undefined components
        for module in list(self.modules.values()):
            for inst in module.instances:
                if inst.module_ref not in self.modules:
                    logging.warning(f"Creating stub module for undefined component: {inst.module_ref}")
                    stub_module = Module(name=inst.module_ref)
                    stub_module.is_stub = True
                    # Infer ports from the instance's pins
                    for i, pin in enumerate(inst.pins):
                        # Create a generic port, direction might be unknown (default to INOUT)

                        stub_module.add_port(f"{pin.name}", "inout", pin.msb, pin.lsb)
                    self.modules[inst.module_ref] = stub_module

    def _find_decl_for_port(self, module_node, port_name):
        for item in module_node.items:
            if isinstance(item, vast.Decl):
                for decl in item.list:
                    if hasattr(decl, "name") and self._clean_name(decl.name) == port_name:
                        return decl
        return None

    def populate_module(self, module, module_node):
        # Process instances
        for item in module_node.items:
            if isinstance(item, vast.InstanceList):
                for inst in item.instances:
                    module_ref_name = self._clean_name(inst.module)
                    module_ref = self.modules.get(module_ref_name)
                    inst_name = self._clean_name(inst.name)
                    module_ref_name = self._clean_name(inst.module)
                    instance = module.add_instance(inst_name, module_ref_name)
                    for port_arg in inst.portlist:
                        info = self.modifier.extract_portarg_with_width(port_arg, module_node)
                        w = info["connection_width"]
                        if w == 1:
                            instance.add_pin(
                                info["port_name"],
                            )
                        else:
                            instance.add_pin(info["port_name"], (w - 1), 0)


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

    parser = VerilogReorder()
    try:
        output_filename = parser.parse_and_create_stubs("design.sv")
        print("\n Success! Output written to:", output_filename)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
