#!/usr/bin/env python3
"""
Pyverilog: Adding new modules and instances to AST
Demonstrates how to programmatically modify Verilog AST
"""

import logging as log
from typing import List

from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
from pyverilog.vparser.parser import parse


class VerilogModifier:
    """Helper class to modify Verilog AST"""

    def __init__(self, ast):
        self.ast = ast
        self.codegen = ASTCodeGenerator()

    def extract_port_info(self, port):
        """
        Extract port information from an Ioport node

        Args:
            port: Ioport node

        Returns:
            dict with keys: name, direction, msb, lsb, signed
        """
        port_info = {"name": None, "direction": None, "msb": None, "lsb": None, "signed": False}

        # Get the first child (Input, Output, or Inout)
        if hasattr(port, "first") and port.first:
            port_decl = port.first

            # Extract name
            port_info["name"] = port_decl.name

            # Extract direction
            port_type = type(port_decl).__name__
            port_info["direction"] = port_type.lower()  # 'input', 'output', 'inout'

            # Extract signed
            if hasattr(port_decl, "signed"):
                port_info["signed"] = port_decl.signed

            # Extract width (msb, lsb)
            if hasattr(port_decl, "width") and port_decl.width:
                width = port_decl.width
                if hasattr(width, "msb") and width.msb:
                    # MSB is typically an IntConst
                    if isinstance(width.msb, IntConst):
                        port_info["msb"] = int(width.msb.value)
                    else:
                        port_info["msb"] = str(width.msb)

                if hasattr(width, "lsb") and width.lsb:
                    # LSB is typically an IntConst
                    if isinstance(width.lsb, IntConst):
                        port_info["lsb"] = int(width.lsb.value)
                    else:
                        port_info["lsb"] = str(width.lsb)
            else:
                # Single bit (no width specified)
                port_info["msb"] = None
                port_info["lsb"] = None

        return port_info

    def extract_portarg_info(self, port_arg):
        """
        Extract information from a PortArg node (port connection in an instance)

        Args:
            port_arg: PortArg node from an instance's portlist

        Returns:
            tuple: (port_name, signal_name, msb, lsb) where msb/lsb are for bit selects
                   in the connection itself (e.g., signal[7:0]), NOT the signal's declared width
        """
        port_name = None
        signal_name = None
        msb = None
        lsb = None

        # Get port name
        if hasattr(port_arg, "portname") and port_arg.portname:
            port_name = port_arg.portname

        # Get connected signal/expression
        if hasattr(port_arg, "argname") and port_arg.argname:
            argname = port_arg.argname

            # Simple identifier (most common case)
            if isinstance(argname, Identifier):
                signal_name = argname.name

            # Pointer (bit or part select)
            elif isinstance(argname, Pointer):
                # Pointer has .var (the signal) and .ptr (the index/range)
                if hasattr(argname, "var") and isinstance(argname.var, Identifier):
                    signal_name = argname.var.name

                # Check if it's a bit select or range select
                if hasattr(argname, "ptr"):
                    ptr = argname.ptr

                    # Single bit select: signal[5]
                    if isinstance(ptr, IntConst):
                        msb = int(ptr.value)
                        lsb = int(ptr.value)
                    elif isinstance(ptr, Identifier):
                        msb = ptr.name
                        lsb = ptr.name
                    # Range select: signal[7:0] (Partselect)
                    elif hasattr(ptr, "__class__") and "Partselect" in ptr.__class__.__name__:
                        if hasattr(ptr, "msb"):
                            if isinstance(ptr.msb, IntConst):
                                msb = int(ptr.msb.value)
                            else:
                                msb = str(ptr.msb)
                        if hasattr(ptr, "lsb"):
                            if isinstance(ptr.lsb, IntConst):
                                lsb = int(ptr.lsb.value)
                            else:
                                lsb = str(ptr.lsb)
                    else:
                        log.warning(f"Unknown element in verilog: {ptr.show()}")

            # Concatenation: {signal1, signal2}
            elif isinstance(argname, Concat):
                # Build concatenation string
                concat_signals = []
                if hasattr(argname, "list"):
                    for item in argname.list:
                        if isinstance(item, Identifier):
                            concat_signals.append(item.name)
                        else:
                            concat_signals.append(str(item))
                signal_name = "{" + ", ".join(concat_signals) + "}"

            # Constant value
            elif isinstance(argname, IntConst):
                signal_name = argname.value
            elif isinstance(argname, Partselect):
                if hasattr(argname, "var"):
                    signal_name = argname.var.name
                if hasattr(argname, "msb"):
                    if isinstance(argname.msb, IntConst):
                        msb = int(argname.msb.value)
                if hasattr(argname, "lsb"):
                    if isinstance(argname.lsb, IntConst):
                        lsb = int(argname.lsb.value)

            # Other expressions (unary, binary operations, etc.)
            else:
                log.warning(f"ignoring element in verilog: {argname.show()}")
                signal_name = str(argname)
        return (port_name, signal_name, msb, lsb)

    def get_signal_width_from_module(self, module_def, signal_name):
        """
        Get the declared width of a signal (wire/reg/port) in a module

        Args:
            module_def: ModuleDef node
            signal_name: Name of the signal to look up

        Returns:
            tuple: (msb, lsb, signal_type) where signal_type is 'wire', 'reg', 'input', 'output', 'inout'
                   Returns (None, None, None) if signal not found
        """
        # Check ports
        if module_def.portlist and module_def.portlist.ports:
            for port in module_def.portlist.ports:
                port_info = self.extract_port_info(port)
                if port_info["name"] == signal_name:
                    return (port_info["msb"], port_info["lsb"], port_info["direction"])

        # Check items (wires, regs)
        if module_def.items:
            for item in module_def.items:
                if isinstance(item, Decl):
                    for decl_item in item.list:
                        if hasattr(decl_item, "name") and decl_item.name == signal_name:
                            msb, lsb = None, None
                            signal_type = type(decl_item).__name__.lower()

                            if hasattr(decl_item, "width") and decl_item.width:
                                if hasattr(decl_item.width, "msb") and decl_item.width.msb:
                                    if isinstance(decl_item.width.msb, IntConst):
                                        msb = int(decl_item.width.msb.value)
                                if hasattr(decl_item.width, "lsb") and decl_item.width.lsb:
                                    if isinstance(decl_item.width.lsb, IntConst):
                                        lsb = int(decl_item.width.lsb.value)

                            return (msb, lsb, signal_type)

        return (None, None, None)

    def extract_portarg_with_width(self, port_arg, module_def):
        """
        Extract PortArg info AND the declared width of the connected signal

        Args:
            port_arg: PortArg node
            module_def: ModuleDef node (parent module containing this connection)

        Returns:
            dict with:
                - port_name: Name of the port being connected
                - signal_name: Name of the signal/expression being connected
                - select_msb: MSB of bit/part select in connection (if any)
                - select_lsb: LSB of bit/part select in connection (if any)
                - signal_msb: MSB of signal's declared width
                - signal_lsb: LSB of signal's declared width
                - signal_type: Type of signal ('wire', 'reg', 'input', 'output', etc.)
                - signal_width: Full declared width of the signal
                - connection_width: Actual width being connected (considering bit/part selects)
        """
        # Get basic connection info
        port_name, signal_name, select_msb, select_lsb = self.extract_portarg_info(port_arg)

        # Get signal's declared width
        signal_msb, signal_lsb, signal_type = self.get_signal_width_from_module(module_def, signal_name)

        # Calculate full declared width
        if signal_msb is not None and signal_lsb is not None:
            signal_width = abs(signal_msb - signal_lsb) + 1
        else:
            signal_width = 1

        # Calculate actual connection width
        # If there's a bit/part select in the connection, use that
        # Otherwise, use the full signal width
        if select_msb is not None and select_lsb is not None:
            if isinstance(select_msb, int) and isinstance(select_lsb, int):
                connection_width = abs(select_msb - select_lsb) + 1
            else:
                connection_width = None  # Dynamic/parameter-based width
        else:
            connection_width = signal_width

        return {
            "port_name": port_name,
            "signal_name": signal_name,
            "select_msb": select_msb,  # From connection like A(B[7:0])
            "select_lsb": select_lsb,
            "signal_msb": signal_msb,  # From declaration like wire [7:0] B;
            "signal_lsb": signal_lsb,
            "signal_type": signal_type,
            "signal_width": signal_width,  # Full declared width
            "connection_width": connection_width,  # Actual width being connected
        }

    def add_port_to_module(self, module_def, port_name, direction, msb=None, lsb=None, signed=False):
        """
        Add a port to an existing ModuleDef

        Args:
            module_def: ModuleDef node to modify
            port_name: Name of the port
            direction: 'input', 'output', or 'inout'
            msb: Most significant bit (None for single bit)
            lsb: Least significant bit (None for single bit)
            signed: Whether the port is signed

        Returns:
            Modified ModuleDef
        """
        # Create width specification if needed
        if msb is not None and lsb is not None:
            width = (msb, lsb)
        else:
            width = None

        # Create new port
        new_port = self.create_port(port_name, direction, width, signed)

        # Add to existing portlist
        if module_def.portlist and module_def.portlist.ports:
            current_ports = list(module_def.portlist.ports)
            current_ports.append(new_port)
            module_def.portlist.ports = tuple(current_ports)
        else:
            module_def.portlist = Portlist([new_port])

        return module_def

    def extract_instances_from_module(self, module_def):
        """
        Extract all instances from a ModuleDef

        Args:
            module_def: ModuleDef node

        Returns:
            List of dicts with instance information
        """
        instances = []

        if not module_def.items:
            return instances

        for item in module_def.items:
            if isinstance(item, InstanceList):
                # InstanceList contains module name, parameters, and list of instances
                module_name = item.module

                for inst in item.instances:
                    inst_info = {
                        "module_name": module_name,
                        "instance_name": inst.name,
                        "ports": {},
                        "port_details": [],  # Detailed info including bit selects
                        "parameters": {},
                    }

                    # Extract port connections using the new method
                    if inst.portlist:
                        for port_arg in inst.portlist:
                            if isinstance(port_arg, PortArg):
                                port_name, signal_name, msb, lsb = self.extract_portarg_info(port_arg)

                                # Store simple mapping
                                if port_name and signal_name:
                                    inst_info["ports"][port_name] = signal_name

                                # Store detailed info
                                inst_info["port_details"].append({"port": port_name, "signal": signal_name, "msb": msb, "lsb": lsb})

                    # Extract parameters
                    if inst.parameterlist:
                        for param_arg in inst.parameterlist:
                            if isinstance(param_arg, ParamArg):
                                param_name = param_arg.paramname
                                if param_arg.argname:
                                    if isinstance(param_arg.argname, IntConst):
                                        param_value = param_arg.argname.value
                                    elif isinstance(param_arg.argname, Identifier):
                                        param_value = param_arg.argname.name
                                    else:
                                        param_value = str(param_arg.argname)
                                    inst_info["parameters"][param_name] = param_value

                    instances.append(inst_info)

        return instances

    def add_instance_to_module(self, module_def, module_name, inst_name, port_connections, parameters=None):
        """
        Add an instance to an existing ModuleDef

        Args:
            module_def: ModuleDef node to modify
            module_name: Name of the module to instantiate
            inst_name: Name of the instance
            port_connections: Dict of {port_name: signal_name}
            parameters: Dict of {param_name: value} (optional)

        Returns:
            Modified ModuleDef
        """
        # Create the instance
        instance = self.create_instance(module_name, inst_name, port_connections, parameters)

        # Add to module items
        if module_def.items:
            current_items = list(module_def.items)
            current_items.append(instance)
            module_def.items = tuple(current_items)
        else:
            module_def.items = (instance,)

        return module_def

    def populate_module(self, module_obj, module_def_node):
        """
        Populate a module object with instances, nets, and connections from AST
        This is useful for building an internal representation from parsed Verilog

        Args:
            module_obj: Your custom module object/dict to populate
            module_def_node: ModuleDef AST node from pyverilog
        """
        # Extract and store ports
        if hasattr(module_obj, "ports") or isinstance(module_obj, dict):
            ports = []
            if module_def_node.portlist and module_def_node.portlist.ports:
                for port in module_def_node.portlist.ports:
                    port_info = self.extract_port_info(port)
                    ports.append(port_info)

            if isinstance(module_obj, dict):
                module_obj["ports"] = ports
            else:
                module_obj.ports = ports

        # Extract and store instances
        instances = self.extract_instances_from_module(module_def_node)
        if isinstance(module_obj, dict):
            module_obj["instances"] = instances
        else:
            module_obj.instances = instances

        # Extract nets/wires
        nets = []
        regs = []
        if module_def_node.items:
            for item in module_def_node.items:
                if isinstance(item, Decl):
                    for decl_item in item.list:
                        if isinstance(decl_item, Wire):
                            net_info = {
                                "name": decl_item.name,
                                "type": "wire",
                                "signed": decl_item.signed if hasattr(decl_item, "signed") else False,
                            }
                            if decl_item.width:
                                if isinstance(decl_item.width.msb, IntConst):
                                    net_info["msb"] = int(decl_item.width.msb.value)
                                if isinstance(decl_item.width.lsb, IntConst):
                                    net_info["lsb"] = int(decl_item.width.lsb.value)
                            nets.append(net_info)

                        elif isinstance(decl_item, Reg):
                            reg_info = {
                                "name": decl_item.name,
                                "type": "reg",
                                "signed": decl_item.signed if hasattr(decl_item, "signed") else False,
                            }
                            if decl_item.width:
                                if isinstance(decl_item.width.msb, IntConst):
                                    reg_info["msb"] = int(decl_item.width.msb.value)
                                if isinstance(decl_item.width.lsb, IntConst):
                                    reg_info["lsb"] = int(decl_item.width.lsb.value)
                            regs.append(reg_info)

        if isinstance(module_obj, dict):
            module_obj["nets"] = nets
            module_obj["regs"] = regs
        else:
            module_obj.nets = nets
            module_obj.regs = regs

        return module_obj

    def add_module(self, module_def):
        """
        Add a new module definition to the AST

        Args:
            module_def: ModuleDef node to add
        """
        if not self.ast.description:
            # Create description if it doesn't exist
            self.ast.description = Description([module_def])
        else:
            # Add to existing definitions
            current_defs = list(self.ast.description.definitions)
            current_defs.append(module_def)
            # Replace with new tuple
            self.ast.description = Description(tuple(current_defs))

        return self.ast

    def create_paramlist_from_strlist(self, param_names: List[str]) -> Paramlist:
        """
        Creates a pyverilog Paramlist node from a list of parameter names.
        Each parameter is assigned a safe default value (IntConst('1')).
        """

        # Define a safe default value node for the parameter
        default_value_node = IntConst("1")

        # Use a list comprehension to create the list of vast.Param nodes
        param_nodes = [Parameter(name=name, value=default_value_node, width=None, signed=None) for name in param_names]
        paramlist = Paramlist(param_nodes)
        return paramlist

    def create_module(self, module_name, ports=None, items=None, parameters=None):
        """
        Create a new ModuleDef node

        Args:
            module_name: Name of the module
            ports: List of port declarations (Ioport nodes)
            items: List of module items (Decl, InstanceList, etc.)

        Returns:
            ModuleDef node
        """
        paramlist = self.create_paramlist_from_strlist(parameters or [])
        portlist = Portlist(ports or [])

        module = ModuleDef(name=module_name, paramlist=paramlist, portlist=portlist, items=items or [])

        return module

    def create_port(self, port_name, port_type="input", width=None, signed=False):
        """
        Create a port declaration

        Args:
            port_name: Name of the port
            port_type: 'input', 'output', 'inout'
            width: Width specification [msb:lsb] as tuple (msb, lsb) or None for 1-bit
            signed: Whether the port is signed

        Returns:
            Ioport node
        """
        # Create width if specified
        if width:
            msb, lsb = width
            width_node = Width(msb=IntConst(str(msb)), lsb=IntConst(str(lsb)))
        else:
            width_node = None

        # Create the appropriate port type
        if port_type == "input":
            port = Input(name=port_name, width=width_node, signed=signed)
        elif port_type == "output":
            port = Output(name=port_name, width=width_node, signed=signed)
        elif port_type == "inout":
            port = Inout(name=port_name, width=width_node, signed=signed)
        else:
            raise ValueError(f"Unknown port type: {port_type}")

        return Ioport(port)

    def create_wire(self, wire_name, width=None, signed=False):
        """
        Create a wire declaration

        Args:
            wire_name: Name of the wire
            width: Width specification [msb:lsb] as tuple (msb, lsb)
            signed: Whether the wire is signed

        Returns:
            Decl node with Wire
        """
        if width:
            msb, lsb = width
            width_node = Width(msb=IntConst(str(msb)), lsb=IntConst(str(lsb)))
        else:
            width_node = None

        wire = Wire(name=wire_name, width=width_node, signed=signed)
        return Decl([wire])

    def create_reg(self, reg_name, width=None, signed=False):
        """
        Create a reg declaration

        Args:
            reg_name: Name of the register
            width: Width specification [msb:lsb] as tuple (msb, lsb)
            signed: Whether the register is signed

        Returns:
            Decl node with Reg
        """
        if width:
            msb, lsb = width
            width_node = Width(msb=IntConst(str(msb)), lsb=IntConst(str(lsb)))
        else:
            width_node = None

        reg = Reg(name=reg_name, width=width_node, signed=signed)
        return Decl([reg])

    def create_instance(self, module_name, inst_name, port_connections, parameters=None):
        """
        Create a module instance

        Args:
            module_name: Name of the module to instantiate
            inst_name: Name of the instance
            port_connections: Dict of {port_name: signal_name} or list of PortArg
            parameters: Dict of {param_name: value} or list of ParamArg

        Returns:
            InstanceList node
        """
        # Create parameter list
        param_list = []
        if parameters:
            if isinstance(parameters, dict):
                for name, value in parameters.items():
                    param_list.append(ParamArg(paramname=name, argname=Identifier(value)))
            else:
                param_list = parameters

        # Create port connections
        if isinstance(port_connections, dict):
            port_args = [PortArg(portname=port, argname=Identifier(signal)) for port, signal in port_connections.items()]
        else:
            port_args = port_connections

        # Create instance
        instance = Instance(module=module_name, name=inst_name, portlist=port_args, parameterlist=param_list)

        return InstanceList(module_name, param_list, [instance])

    def add_item_to_module(self, module_name, item):
        """
        Add an item (wire, reg, instance, etc.) to an existing module

        Args:
            module_name: Name of the module to modify
            item: AST node to add (Decl, InstanceList, etc.)
        """
        for definition in self.ast.description.definitions:
            if isinstance(definition, ModuleDef) and definition.name == module_name:
                # Add item to module
                current_items = list(definition.items)
                current_items.append(item)
                definition.items = tuple(current_items)
                return True
        return False

    def generate_code(self):
        """Generate Verilog code from AST"""
        return self.codegen.visit(self.ast)


def example_create_simple_module():
    """Example: Create a simple module from scratch"""
    print("=" * 60)
    print("Example 1: Creating a simple AND gate module")
    print("=" * 60)

    modifier = VerilogModifier(Source("", Description([])))

    # Create ports
    ports = [modifier.create_port("a", "input"), modifier.create_port("b", "input"), modifier.create_port("y", "output")]

    # Create wire for output
    wire_y = modifier.create_wire("y")

    # Create assignment: assign y = a & b;
    assign = Assign(left=Lvalue(Identifier("y")), right=And(Identifier("a"), Identifier("b")))

    # Create module
    module = modifier.create_module(module_name="and_gate", ports=ports, items=[wire_y, assign])

    # Add module to AST
    modifier.add_module(module)

    # Generate code
    code = modifier.generate_code()
    print(code)
    print()


def example_create_complex_module():
    """Example: Create a module with buses and instances"""
    print("=" * 60)
    print("Example 2: Creating a module with buses and instances")
    print("=" * 60)

    modifier = VerilogModifier(Source("", Description([])))

    # Create main module with bus ports
    ports = [
        modifier.create_port("clk", "input"),
        modifier.create_port("rst_n", "input"),
        modifier.create_port("data_in", "input", width=(7, 0)),
        modifier.create_port("data_out", "output", width=(7, 0)),
        modifier.create_port("enable", "input"),
    ]

    # Create internal wires
    wire_internal = modifier.create_wire("internal_data", width=(7, 0))
    wire_valid = modifier.create_wire("valid")

    # Create register
    reg_data = modifier.create_reg("data_reg", width=(7, 0))

    # Create module items list
    items = [wire_internal, wire_valid, reg_data]

    # Create module
    module = modifier.create_module(module_name="data_processor", ports=ports, items=items)

    # Add module to AST
    modifier.add_module(module)

    # Generate code
    code = modifier.generate_code()
    print(code)
    print()


def example_add_to_existing():
    """Example: Parse existing code and add new module"""
    print("=" * 60)
    print("Example 3: Adding module to existing design")
    print("=" * 60)

    # Existing Verilog code
    verilog_code = """
module counter (
    input wire clk,
    input wire rst_n,
    output reg [7:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 8'b0;
        else
            count <= count + 1;
    end
endmodule
"""

    # Parse existing code
    ast, _ = parse([verilog_code], preprocess_include=[], preprocess_define=[])

    modifier = VerilogModifier(ast)

    # Create new module
    ports = [modifier.create_port("in", "input", width=(7, 0)), modifier.create_port("out", "output", width=(7, 0))]

    wire_out = modifier.create_wire("out", width=(7, 0))

    # Create assignment
    assign = Assign(left=Lvalue(Identifier("out")), right=Identifier("in"))

    new_module = modifier.create_module(module_name="passthrough", ports=ports, items=[wire_out, assign])

    # Add new module
    modifier.add_module(new_module)

    # Generate code
    code = modifier.generate_code()
    print(code)
    print()


def example_add_instance():
    """Example: Create new module with instance"""
    print("=" * 60)
    print("Example 4: Creating new module with instance")
    print("=" * 60)

    # Existing Verilog code
    verilog_code = """
module counter (
    input wire clk,
    input wire rst_n,
    output reg [7:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 8'b0;
        else
            count <= count + 1;
    end
endmodule
"""

    # Parse existing code
    ast, _ = parse([verilog_code], preprocess_include=[], preprocess_define=[])

    modifier = VerilogModifier(ast)

    # Create new top module that instantiates counter
    top_ports = [
        modifier.create_port("clk", "input"),
        modifier.create_port("rst_n", "input"),
        modifier.create_port("count_out", "output", width=(7, 0)),
    ]

    # Create internal wire
    wire_internal = modifier.create_wire("internal_count", width=(7, 0))

    # Create counter instance
    counter_inst = modifier.create_instance(
        module_name="counter",
        inst_name="counter_inst",
        port_connections={"clk": "clk", "rst_n": "rst_n", "count": "internal_count"},
    )

    # Create assignment to output
    assign = Assign(left=Lvalue(Identifier("count_out")), right=Identifier("internal_count"))

    # Create new top module with wire, instance, and assignment
    top_module = modifier.create_module(module_name="top", ports=top_ports, items=[wire_internal, counter_inst, assign])

    # Add new module to AST
    modifier.add_module(top_module)

    print("New module 'top' created with counter instance!")

    # Generate code
    code = modifier.generate_code()
    print(code)
    print()


def example_hierarchical_design():
    """Example: Create hierarchical design with multiple modules"""
    print("=" * 60)
    print("Example 5: Creating hierarchical design")
    print("=" * 60)

    modifier = VerilogModifier(Source("", Description([])))

    # Create leaf module (adder)
    adder_ports = [
        modifier.create_port("a", "input", width=(7, 0)),
        modifier.create_port("b", "input", width=(7, 0)),
        modifier.create_port("sum", "output", width=(7, 0)),
    ]

    adder_wire = modifier.create_wire("sum", width=(7, 0))
    adder_assign = Assign(left=Lvalue(Identifier("sum")), right=Plus(Identifier("a"), Identifier("b")))

    adder_module = modifier.create_module(module_name="adder", ports=adder_ports, items=[adder_wire, adder_assign])

    # Create top module
    top_ports = [
        modifier.create_port("clk", "input"),
        modifier.create_port("x", "input", width=(7, 0)),
        modifier.create_port("y", "input", width=(7, 0)),
        modifier.create_port("result", "output", width=(7, 0)),
    ]

    top_wire = modifier.create_wire("result", width=(7, 0))

    # Create instance of adder in top
    adder_inst = modifier.create_instance(
        module_name="adder", inst_name="u_adder", port_connections={"a": "x", "b": "y", "sum": "result"}
    )

    top_module = modifier.create_module(module_name="top", ports=top_ports, items=[top_wire, adder_inst])

    # Add both modules
    modifier.add_module(adder_module)
    modifier.add_module(top_module)

    # Generate code
    code = modifier.generate_code()
    print(code)
    print()


def example_extract_and_add_ports():
    """Example: Extract port info and add ports to modules"""
    print("=" * 60)
    print("Example 6: Extracting and Adding Ports")
    print("=" * 60)

    # Create a module with some ports
    verilog_code = """
module example (
    input wire clk,
    input wire [7:0] data_in,
    output wire [15:0] data_out,
    output wire valid
);
endmodule
"""

    # Parse existing code
    ast, _ = parse([verilog_code], preprocess_include=[], preprocess_define=[])

    modifier = VerilogModifier(ast)

    # Extract port information
    print("Extracted Port Information:")
    print("-" * 40)

    for definition in ast.description.definitions:
        if isinstance(definition, ModuleDef):
            print(f"Module: {definition.name}")
            if definition.portlist and definition.portlist.ports:
                for port in definition.portlist.ports:
                    info = modifier.extract_port_info(port)
                    print(f"  Port: {info['name']}")
                    print(f"    Direction: {info['direction']}")
                    if info["msb"] is not None:
                        print(f"    Width: [{info['msb']}:{info['lsb']}]")
                    else:
                        print(f"    Width: 1-bit")
                    print(f"    Signed: {info['signed']}")
                    print()

            # Now add new ports using the extracted format
            print("Adding new ports to module...")
            modifier.add_port_to_module(definition, "enable", "input")
            modifier.add_port_to_module(definition, "error", "output")
            modifier.add_port_to_module(definition, "status", "output", msb=3, lsb=0)
            modifier.add_port_to_module(definition, "control", "input", msb=7, lsb=0, signed=True)

    print("-" * 40)
    print("\nGenerated Verilog with new ports:")
    print("-" * 40)

    # Generate updated code
    code = modifier.generate_code()
    print(code)
    print()


def example_populate_module():
    """Example: Populate module objects from AST (like your use case)"""
    print("=" * 60)
    print("Example 7: Populating Module Objects from AST")
    print("=" * 60)

    verilog_code = """
module top (
    input wire clk,
    input wire rst_n,
    input wire [7:0] data_in,
    output wire [15:0] result
);
    wire [7:0] internal_sig;
    wire [15:0] intermediate;
    reg [7:0] data_reg;
    
    processor #(
        .WIDTH(8)
    ) u_proc (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_out(internal_sig)
    );
    
    multiplier u_mult (
        .a(internal_sig),
        .b(8'd5),
        .product(intermediate)
    );
    
    assign result = intermediate;
    
endmodule

module processor #(
    parameter WIDTH = 8
) (
    input wire clk,
    input wire rst_n,
    input wire [WIDTH-1:0] data_in,
    output reg [WIDTH-1:0] data_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            data_out <= 0;
        else
            data_out <= data_in;
    end
endmodule
"""

    # Parse the Verilog
    ast, _ = parse([verilog_code], preprocess_include=[], preprocess_define=[])

    modifier = VerilogModifier(ast)

    # Create a dictionary to store module objects (simulating your use case)
    modules = {}

    # First pass: Create module entries
    for node in ast.description.definitions:
        if isinstance(node, ModuleDef):
            module_name = node.name
            modules[module_name] = {"name": module_name, "ports": [], "instances": [], "nets": [], "regs": []}

    # Second pass: Populate modules with instances, nets, and connections
    print("Populating modules with instances, nets, and connections...")
    print("=" * 60)

    for node in ast.description.definitions:
        if isinstance(node, ModuleDef):
            module_name = node.name
            module_obj = modules[module_name]

            # Populate the module object
            modifier.populate_module(module_obj, node)

    # Display the populated module information
    for module_name, module_obj in modules.items():
        print(f"\nModule: {module_name}")
        print("-" * 40)

        print(f"  Ports ({len(module_obj['ports'])}):")
        for port in module_obj["ports"]:
            width_str = f"[{port.get('msb', 0)}:{port.get('lsb', 0)}]" if port.get("msb") is not None else ""
            print(f"    {port['direction']:8s} {port['name']}{width_str}")

        print(f"\n  Nets ({len(module_obj['nets'])}):")
        for net in module_obj["nets"]:
            width_str = f"[{net.get('msb', 0)}:{net.get('lsb', 0)}]" if net.get("msb") is not None else ""
            print(f"    wire {net['name']}{width_str}")

        print(f"\n  Regs ({len(module_obj['regs'])}):")
        for reg in module_obj["regs"]:
            width_str = f"[{reg.get('msb', 0)}:{reg.get('lsb', 0)}]" if reg.get("msb") is not None else ""
            print(f"    reg  {reg['name']}{width_str}")

        print(f"\n  Instances ({len(module_obj['instances'])}):")
        for inst in module_obj["instances"]:
            print(f"    {inst['module_name']} {inst['instance_name']}")
            if inst["parameters"]:
                print(f"      Parameters:")
                for pname, pval in inst["parameters"].items():
                    print(f"        .{pname}({pval})")
            print(f"      Ports:")
            for pname, signal in inst["ports"].items():
                print(f"        .{pname}({signal})")

        print()

    # Now demonstrate adding a new instance to a module
    print("\n" + "=" * 60)
    print("Adding new instance to 'top' module...")
    print("=" * 60)

    # Find the top module AST node
    for node in ast.description.definitions:
        if isinstance(node, ModuleDef) and node.name == "top":
            # Add a new instance
            modifier.add_instance_to_module(
                node,
                module_name="adder",
                inst_name="u_adder",
                port_connections={"a": "internal_sig", "b": "data_reg", "sum": "intermediate"},
            )
            break

    # Re-generate the code to show the new instance
    print("\nUpdated Verilog code:")
    print("-" * 40)
    code = modifier.generate_code()
    print(code)


def example_decode_port_connections():
    """Example: Decode various types of port connections"""
    print("=" * 60)
    print("Example 8: Decoding PortArg Connections with Signal Widths")
    print("=" * 60)

    verilog_code = """
module complex_connections (
    input wire clk,
    input wire [31:0] bus_in,
    input wire [7:0] data_a,
    input wire [7:0] data_b,
    output wire [15:0] result
);
    wire [31:0] internal_bus;
    wire [7:0] byte0, byte1, byte2, byte3;
    wire [15:0] wide_result;
    
    // Instance with simple connections (no bit select)
    simple_module u_simple (
        .clk(clk),
        .data(data_a)
    );
    
    // Instance with bit select in connection
    bit_selector u_bits (
        .in(bus_in[7:0]),
        .out(byte0)
    );
    
    // Instance with concatenation
    combiner u_combine (
        .in({data_a, data_b}),
        .out(wide_result)
    );
    
    // Instance with constant
    constant_user u_const (
        .value(8'hFF),
        .output(byte1)
    );
    
    // Instance with part select
    splitter u_split (
        .data_in(internal_bus[23:16]),
        .data_out(byte2)
    );
    
endmodule
"""

    # Parse the Verilog
    ast, _ = parse([verilog_code], preprocess_include=[], preprocess_define=[])

    modifier = VerilogModifier(ast)

    # Extract and display port connection details
    for node in ast.description.definitions:
        if isinstance(node, ModuleDef):
            print(f"\nModule: {node.name}")
            print("=" * 60)

            # First show what signals are declared
            print("\n  Declared Signals:")
            print("  " + "-" * 56)

            if node.portlist and node.portlist.ports:
                for port in node.portlist.ports:
                    info = modifier.extract_port_info(port)
                    if info["msb"] is not None:
                        print(f"    {info['direction']:8s} [{info['msb']:2d}:{info['lsb']:2d}] {info['name']}")
                    else:
                        print(f"    {info['direction']:8s}         {info['name']}")

            if node.items:
                for item in node.items:
                    if isinstance(item, Decl):
                        for decl_item in item.list:
                            if isinstance(decl_item, Wire):
                                if decl_item.width:
                                    msb = int(decl_item.width.msb.value) if isinstance(decl_item.width.msb, IntConst) else "?"
                                    lsb = int(decl_item.width.lsb.value) if isinstance(decl_item.width.lsb, IntConst) else "?"
                                    print(f"    wire     [{msb:2}:{lsb:2}] {decl_item.name}")
                                else:
                                    print(f"    wire           {decl_item.name}")

            # Now show instances with full width information
            print("\n  Instance Port Connections:")
            print("  " + "-" * 56)

            if node.items:
                for item in node.items:
                    if isinstance(item, InstanceList):
                        for inst in item.instances:
                            print(f"\n    {item.module} {inst.name}:")

                            if inst.portlist:
                                for port_arg in inst.portlist:
                                    if isinstance(port_arg, PortArg):
                                        # Get full width info
                                        info = modifier.extract_portarg_with_width(port_arg, node)

                                        port = info["port_name"]
                                        signal = info["signal_name"]

                                        # Build connection string
                                        conn_str = signal
                                        if info["select_msb"] is not None:
                                            if info["select_msb"] == info["select_lsb"]:
                                                conn_str += f"[{info['select_msb']}]"
                                            else:
                                                conn_str += f"[{info['select_msb']}:{info['select_lsb']}]"

                                        # Show declared width
                                        if info["signal_msb"] is not None:
                                            width_str = f"[{info['signal_msb']}:{info['signal_lsb']}]"
                                        else:
                                            width_str = "[0:0]"

                                        print(
                                            f"      .{port:12s} <- {conn_str:20s} (declared as {info['signal_type']} {width_str}, width={info['signal_width']}, connecting {info['connection_width']} bits)"
                                        )

            print()

    print("=" * 60)
    print("\nKEY POINTS:")
    print("  - 'signal_width' = Full DECLARED width of the signal")
    print("  - 'connection_width' = ACTUAL width being connected to the port")
    print("\n  Examples:")
    print("    .data(data_a) where data_a is [7:0]")
    print("      -> signal_width=8, connection_width=8")
    print("\n    .in(bus_in[15:8]) where bus_in is [31:0]")
    print("      -> signal_width=32, connection_width=8")
    print("      -> Connecting 8 bits (15-8+1) from a 32-bit bus")
    print("\n    .clk(clk) where clk is single bit")
    print("      -> signal_width=1, connection_width=1")
    print("=" * 60)


def main():
    """Run all examples"""
    example_create_simple_module()
    example_create_complex_module()
    example_add_to_existing()
    example_add_instance()
    example_hierarchical_design()
    example_extract_and_add_ports()
    example_populate_module()
    example_decode_port_connections()


if __name__ == "__main__":
    main()
