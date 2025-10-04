import logging
import os
import sys
from dataclasses import dataclass, field
from optparse import OptionParser
from typing import Any, Dict, List, Optional, Set, Tuple

import pyverilog.vparser.ast as vast
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.parser import parse

from schematic_from_netlist.interfaces.netlist_database import (
    Bus,
    Instance,
    Module,
    Net,
    NetlistDatabase,
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


@dataclass
class mBus:
    name: str = ""
    msb: Optional[int] = None
    lsb: Optional[int] = None


@dataclass(frozen=True)
class mNet:
    name: str
    bus: Optional[mBus] = field(default=None, compare=False, hash=False)


@dataclass(frozen=True)
class mPort:
    name: str
    direction: str
    msb: int = 0
    lsb: int = 0


@dataclass(frozen=True)
class mPin:
    name: str


@dataclass(frozen=True)
class mInst:
    name: str
    module: "mModule" = field(compare=False, hash=False)
    pins: Set[mPin] = field(default_factory=set, compare=False, hash=False)
    parameters: Set[str] = field(default_factory=set, compare=False, hash=False)

    def __post_init__(self):
        pins = set()
        for p in self.module.ports:
            if not p.msb:
                pin = mPin(p.name)
                pins.add(pin)
            else:
                for i in range(p.lsb, p.msb + 1):
                    name = f"{p.name}[{i}]"
                    pin = mPin(name)
                    pins.add(pin)
        return self

    def add_parameter(self, param):
        self.parameters.add(param)


@dataclass
class mModule:
    name: str = ""
    insts: Set[mInst] = field(default_factory=set)
    ports: List[mPort] = field(default_factory=list)
    nets: Set[mNet] = field(default_factory=set)
    level: int = 0
    is_stub: bool = False
    parameters: Set[str] = field(default_factory=set)

    def add_port(self, name, direction, msb=0, lsb=0):
        port = mPort(name, direction, msb, lsb)
        self.ports.append(port)

    def add_net(self, name, bus=None):
        net = mNet(name, bus)
        self.nets.add(net)

    def add_inst(self, name, module_ref):
        inst = mInst(name, module_ref)
        self.insts.add(inst)
        return inst

    def add_bus(self, name, msb, lsb):
        bus = mBus(name, msb, lsb)
        for i in range(lsb, msb + 1):
            self.add_net(name, bus=bus)
        return bus


class VerilogParser:
    def __init__(self):
        self.modules = {}
        self.top_module_name = None
        self.modifier = VerilogModifier("")
        self.db = NetlistDatabase()

    def _clean_name(self, name):
        """Strip backslashes from names."""
        if isinstance(name, str):
            return name.replace("\\", "")
        return name

    def parse_and_store(self, filename):
        ast, _ = parse([filename])
        self.modifier = VerilogModifier(ast)
        self.walker(ast)
        self.create_ast_stubs(ast)
        output_filename = f"data/verilog/processed_{self.top_module_name}.v"
        self.write_ast(ast, output_filename)
        return self.db

    def create_ast_stubs(self, ast):
        stubs = [module for module in self.modules.values() if module.is_stub]
        for module in stubs:
            portlist = []
            for port in module.ports:
                if port.msb == 0 and port.lsb == 0:
                    vport = self.modifier.create_port(port.name, port.direction)
                else:
                    vport = self.modifier.create_port(port.name, port.direction, width=(port.msb, port.lsb))
                portlist.append(vport)
            vmodule = self.modifier.create_module(module_name=module.name, ports=portlist, parameters=module.parameters)
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

    def walk_add_module(self, node):
        module_name = self._clean_name(node.name)
        
        # If a stub for this module already exists, update it.
        # Otherwise, create a new module object.
        module = self.modules.get(module_name, mModule(name=module_name))
        module.is_stub = False # It's a real module definition now
        
        if module_name not in self.modules:
            self.modules[module_name] = module

        if not self.top_module_name:
            self.top_module_name = module_name

        # Clear any placeholder ports from the stub and add the real ones
        module.ports.clear()
        
        # Collect ports
        if node.portlist:
            for port in node.portlist.ports:
                info = self.modifier.extract_port_info(port)
                (name, direction, msb, lsb) = info["name"], info["direction"], info["msb"], info["lsb"]
                if msb is None:
                    module.add_port(name, direction)
                    module.add_net(name)
                else:
                    module.add_port(name, direction, msb=int(msb), lsb=int(lsb))
                    module.add_bus(name, msb=int(msb), lsb=int(lsb))

    def get_port_name_from_module_port_position(self, module, position, select_msb, select_lsb):
        return "foo"

    def _get_bit_mapping(self, info: "portArgInfo", port_info: "mPort") -> List[Tuple[str, str]]:
        """
        Generates a list of bit-level connections between port pins and signal pins.

        Args:
            info: A portArgInfo object describing the connection.
            port_info: An mPort object describing the port on the module.

        Returns:
            A list of tuples, where each tuple is (port_pin_str, signal_pin_str).
        """
        port_name = info.port_name
        port_width = abs(port_info.msb - port_info.lsb) + 1 if port_info.msb is not None else 1

        # Create a flat list of all signal bits being connected, from MSB to LSB
        signal_bits = []
        for conn in info.connections:
            msb = conn.select_msb if conn.select_msb is not None else conn.signal_msb
            lsb = conn.select_lsb if conn.select_lsb is not None else conn.signal_lsb

            if msb is None and lsb is None:  # Single bit signal
                signal_bits.append(conn.signal_name)
                continue

            try:
                msb, lsb = int(msb), int(lsb)
            except (ValueError, TypeError):
                log.warning(f"Cannot expand non-integer bus slice for signal '{conn.signal_name}' on port '{port_name}'.")
                unexpanded_signal = f"{conn.signal_name}[{msb}:{lsb}]" if msb is not None else conn.signal_name
                signal_bits.append(unexpanded_signal)
                continue

            if msb >= lsb:  # Descending order e.g., [7:0]
                for i in range(msb, lsb - 1, -1):
                    signal_bits.append(f"{conn.signal_name}[{i}]")
            else:  # Ascending order e.g., [0:7]
                for i in range(lsb, msb + 1):
                    signal_bits.append(f"{conn.signal_name}[{i}]")

        # Check for width mismatch
        if len(signal_bits) != port_width:
            log.warning(
                f"Connection width mismatch for port '{port_name}'. "
                f"Port width is {port_width}, but signal connection width is {len(signal_bits)}."
            )
            # Return a single entry representing the unexpanded connection
            return [(port_name, "{" + ",".join(signal_bits) + "}")]

        # Create a flat list of all port pins, from MSB to LSB
        port_pins = []
        if port_width == 1:
            port_pins.append(port_name)
        else:
            if port_info.msb >= port_info.lsb:  # Descending order
                for i in range(port_info.msb, port_info.lsb - 1, -1):
                    port_pins.append(f"{port_name}[{i}]")
            else:  # Ascending order
                for i in range(port_info.msb, port_info.lsb + 1):
                    port_pins.append(f"{port_name}[{i}]")

        # The first signal bit (from the leftmost Verilog element) maps to the first port pin (MSB)
        return list(zip(port_pins, signal_bits))

    def pop_add_instance(self, vinst, module, module_node):
        module_ref_name = self._clean_name(vinst.module)
        if module_ref_name not in self.modules:
            self.pop_create_stub_module(module_ref_name, vinst, module_node)

        module_ref = self.modules[module_ref_name]
        inst = module.add_inst(vinst.name, module_ref)

        if vinst.parameterlist:
            for param_arg in vinst.parameterlist:
                param_name = self._clean_name(param_arg.paramname)
                inst.add_parameter(param_name)

        for i, port_arg in enumerate(vinst.portlist):
            info = self.modifier.extract_portarg_with_width(port_arg, module_node)

            port_info = None
            port_name = info.port_name

            if port_name:
                # Named port connection
                port_info = next((p for p in module_ref.ports if p.name == port_name), None)
            else:
                # Positional port connection
                if i < len(module_ref.ports):
                    port_info = module_ref.ports[i]
                    info.port_name = port_info.name
                else:
                    log.error(
                        f"Positional connection index {i} is out of bounds for module '{module_ref.name}' "
                        f"(has {len(module_ref.ports)} ports). Skipping connection for instance '{inst.name}'."
                    )
                    continue

            if not port_info:
                log.error(f"Port '{port_name}' not found in module '{module_ref.name}' for instance '{inst.name}'")
                continue

            # Get the bit-level mapping and print the connections
            mapping = self._get_bit_mapping(info, port_info)
            for port_pin, signal_pin in mapping:
                print(f"Adding connection {inst.name}/{port_pin} -> {signal_pin}")


    def pop_create_stub_module(self, module_ref_name, vinst, module_node):
        # module_ref = self.modules.get(module_ref_name)
        inst_name = self._clean_name(vinst.name)
        module_ref_name = self._clean_name(vinst.module)
        module_ref = mModule(name=module_ref_name)
        module_ref.is_stub = True
        self.modules[module_ref_name] = module_ref
        if vinst.parameterlist:
            # 'parameterlist' is a vast.ParamArgList, which contains vast.ParamArg objects
            for param_arg in vinst.parameterlist:
                param_name = self._clean_name(param_arg.paramname)
                module_ref.add_parameter(param_name)
        for i, port_arg in enumerate(vinst.portlist):
            info = self.modifier.extract_portarg_with_width(port_arg, module_node)

            port_name = info.port_name
            if not port_name:
                # For positional stubs, we must create ports in order.
                # The real names will be resolved when the actual module is parsed.
                if i < len(module_ref.ports):
                    port_name = module_ref.ports[i].name
                else:
                    # Create a placeholder name
                    port_name = f"p{i}"
                info.port_name = port_name

            if info.total_connection_width <= 1:
                module_ref.add_port(port_name, "inout")
            else:
                (msb, lsb) = (info.total_connection_width - 1, 0)
                module_ref.add_port(port_name, direction="inout", msb=msb, lsb=lsb)
                module_ref.add_bus(port_name, msb=msb, lsb=lsb)

    def pop_add_net(self, decl, module):
        name = self._clean_name(decl.name)
        if decl.width:
            msb = int(decl.width.msb.value)
            lsb = int(decl.width.lsb.value)
            module.add_bus(name, lsb=lsb, msb=msb)
        else:
            module.add_net(name)

    def pop_add_generate(self, statement, module, node):
        log.info(f"Found generate statement: {statement.show()}")

    def walker(self, ast):
        # First pass: Collect module definitions
        for node in ast.description.definitions:
            if isinstance(node, vast.ModuleDef):
                self.walk_add_module(node)

        # Second pass: Populate modules with instances etc
        for node in ast.description.definitions:
            if isinstance(node, vast.ModuleDef):
                module_name = self._clean_name(node.name)
                module = self.modules[module_name]
                self.walk_populate_module(module, node)

    def walk_populate_module(self, module, module_node):
        # Process nets, instances and pins in that order.
        # Nets come first because we need to make sure all busses have been elaborated before adding pins
        for item in module_node.items:
            if isinstance(item, vast.Decl):
                for decl in item.list:
                    if isinstance(decl, vast.Wire):
                        self.pop_add_net(decl, module)
        # ok, all bus/nets are in place...
        for item in module_node.items:
            if isinstance(item, vast.InstanceList):
                for inst in item.instances:
                    self.pop_add_instance(inst, module, module_node)
        for item in module_node.items:
            if isinstance(item, vast.GenerateStatement):
                for statement in item.items:
                    self.pop_add_generate(statement, module, module_node)

        for item in module_node.items:
            if isinstance(item, vast.Decl) or isinstance(item, vast.InstanceList) or isinstance(item, vast.GenerateStatement):
                pass
            else:
                print(f"parser skipping: {type(item)} : {item.show()}")


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
