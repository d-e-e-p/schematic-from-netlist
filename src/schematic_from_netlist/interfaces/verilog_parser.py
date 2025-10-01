import logging as log
import os
import re
import sys

import pyslang

from schematic_from_netlist.interfaces.netlist_database import (
    Bus,
    Instance,
    Module,
    Net,
    NetlistDatabase,
    NetType,
    Pin,
    PinDirection,
)


class VerilogParser:
    def __init__(self):
        self.db = NetlistDatabase()

    def _clean_name(self, name):
        """Strip backslashes and whitespace from names."""
        if isinstance(name, str):
            return name.replace("\\\\", "").strip()
        return name

    def parse_and_store_in_db(self, filename):
        """Parse SystemVerilog file and handle diagnostics"""
        if not os.path.exists(filename):
            raise IOError(f"File not found: {filename}")

        tree = pyslang.SyntaxTree.fromFile(filename)

        bag = pyslang.Bag()
        options = pyslang.CompilationOptions()
        options.languageVersion = pyslang.LanguageVersion.v1800_2017
        bag.compilationOptions = options

        compilation = pyslang.Compilation(bag)
        compilation.addSyntaxTree(tree)

        all_diags = list(compilation.getAllDiagnostics())
        has_errors = False
        for diag in all_diags:
            if diag.isError():
                # Ignore UnknownModule errors, as we expect to see stubbed modules
                if diag.code.name != "UnknownModule":
                    log.error(f"Pyslang error: {diag.code} {diag.args} at {diag.location}")
                    has_errors = True
            else:
                log.warning(str(diag))

        if has_errors:
            raise RuntimeError(f"Pyslang parsing failed with errors for {filename}")

        return self.store_in_db(compilation)

    def store_in_db(self, compilation):
        """Store the compiled design into the netlist database."""
        self._populate_db_from_compilation(compilation)
        self.db.determine_design_hierarchy()
        self.db._build_lookup_tables()
        return self.db

    def _populate_db_from_compilation(self, compilation):
        """Extract all design information and populate the NetlistDatabase."""
        root = compilation.getRoot()

        for instance in root.topInstances:
            self._process_instance_recursive(instance, parent_module_db=None, compilation=compilation)

        if not self.db.top_module and root.topInstances:
            top_module_name = self._clean_name(root.topInstances[0].definition.name)
            if top_module_name in self.db.modules:
                self.db.set_top_module(self.db.modules[top_module_name])

    def _process_instance_recursive(self, instance, parent_module_db, compilation):
        """
        Recursively traverse the design hierarchy, dispatching to Pass 1 (declarations)
        and Pass 2 (connections) helpers.
        """
        definition = instance.definition
        module_name = self._clean_name(definition.name)

        # Pass 1: Process module declarations if this is the first time we see this module.
        module_db = self.db.modules.get(module_name)
        if not module_db:
            module_db = Module(name=module_name)
            self.db.modules[module_name] = module_db
            self._process_module_declarations(module_db, instance)

        # Pass 2: Process instance connections if this instance is instantiated within a parent module.
        if parent_module_db:
            inst_name = self._clean_name(instance.name)
            instance_db = parent_module_db.add_instance(inst_name, module_db, module_name)
            if module_db:
                parent_module_db.child_modules[module_db.name] = module_db
                module_db.parent = parent_module_db
            self._process_instance_connections(instance_db, instance, parent_module_db, compilation)

        # Recurse for sub-instances defined within the current module body.
        for member in instance.body:
            if member.kind == pyslang.SymbolKind.Instance:
                self._process_instance_recursive(member, module_db, compilation)

    def _process_module_declarations(self, module_db, instance):
        """
        PASS 1: Populate the module's busses, nets, and ports from its declarations.
        """
        for member in instance.body:
            if member.kind == pyslang.SymbolKind.Port:
                self._create_bus_and_nets(module_db, member, is_port=True)
            elif member.kind in (pyslang.SymbolKind.Variable, pyslang.SymbolKind.Net):
                self._create_bus_and_nets(module_db, member, is_port=False)

    def _create_bus_and_nets(self, module_db, member, is_port=False):
        """
        Utility to create a Bus object and its constituent Net objects for a given
        declaration (port or internal net).
        """
        name = self._clean_name(member.name)
        is_bus = member.type.isPackedArray or member.type.isUnpackedArray

        parent_bus = None
        if is_bus:
            bit_range = member.type.getBitVectorRange()
            msb, lsb = int(bit_range.left), int(bit_range.right)
            parent_bus = Bus(name=name, bit_range=(msb, lsb), bit_width=abs(msb - lsb) + 1)
            module_db.busses[name] = parent_bus

            # Decompose the bus into single-bit nets
            for i in range(min(lsb, msb), max(lsb, msb) + 1):
                bit_net_name = f"{name}[{i}]"
                if bit_net_name not in module_db.nets:
                    bit_net = module_db.add_net(bit_net_name)
                    bit_net.module = module_db
                    bit_net.parent_bus = parent_bus
                    bit_net.bit_index = i
                    parent_bus.bit_nets[i] = bit_net
        else:
            # It's a scalar wire, create a single net
            if name not in module_db.nets:
                net = module_db.add_net(name)
                net.module = module_db

        if is_port:
            direction = self._get_pyslang_port_direction(member.direction)
            port = module_db.add_port(name, direction)
            port.parent_bus = parent_bus

    def _process_instance_connections(self, instance_db, instance, parent_module_db, compilation):
        """
        PASS 2: Resolve and create pin connections for a given instance.
        """
        for conn in instance.portConnections:
            port_name = self._clean_name(conn.port.name)
            if not conn.expression:
                log.warning(f"Skipping port '{port_name}' on instance '{instance_db.name}' with no expression.")
                continue

            # Resolve the net name from the expression
            net_name = None
            if conn.expression.syntax:
                net_name = self._clean_name(str(conn.expression.syntax))
            else:
                symbol = conn.expression.getSymbolReference()
                if symbol:
                    net_name = self._clean_name(symbol.name)
                else:
                    # Last resort
                    net_name = self._clean_name(str(conn.expression))

            if not net_name:
                log.warning(f"Could not determine net name for port '{port_name}' on instance '{instance_db.name}'.")
                continue

            direction = self._get_pyslang_port_direction(conn.port.direction)
            is_port_bus = conn.port.type.isPackedArray or conn.port.type.isUnpackedArray

            # ... (inside _process_instance_connections)
            if is_port_bus:
                port_range = conn.port.type.getBitVectorRange()
                port_msb, port_lsb = int(port_range.left), int(port_range.right)

                for i in range(abs(port_msb - port_lsb) + 1):
                    port_bit_index = min(port_lsb, port_msb) + i
                    bit_port_name = f"{port_name}[{port_bit_index}]"

                    # Determine the target net name
                    bit_net_name = None
                    match = re.match(r"(\w+)\[(\d+):(\d+)\]", net_name)
                    if match: # Slice: .A(B[7:4])
                        bus_name, msb_str, lsb_str = match.groups()
                        net_lsb = int(lsb_str)
                        bit_net_name = f"{bus_name}[{net_lsb + i}]"
                    else: # Whole bus: .A(B)
                        bit_net_name = f"{net_name}[{port_bit_index}]"

                    net = parent_module_db.nets.get(bit_net_name)
                    if not net:
                        log.warning(f"Net '{bit_net_name}' not found in module '{parent_module_db.name}'. Creating it.")
                        net = parent_module_db.add_net(bit_net_name)
                        net.module = parent_module_db
                    
                    instance_db.add_pin(bit_port_name, direction)
                    instance_db.connect_pin(bit_port_name, net)
            else: # Scalar connection
                net = parent_module_db.nets.get(net_name)
                if not net:
                    log.warning(f"Net '{net_name}' not found in module '{parent_module_db.name}'. Creating it.")
                    net = parent_module_db.add_net(net_name)
                    net.module = parent_module_db

                instance_db.add_pin(port_name, direction)
                instance_db.connect_pin(port_name, net)


    def _get_pyslang_port_direction(self, direction):
        if direction == pyslang.ArgumentDirection.In:
            return PinDirection.INPUT
        elif direction == pyslang.ArgumentDirection.Out:
            return PinDirection.OUTPUT
        elif direction == pyslang.ArgumentDirection.InOut:
            return PinDirection.INOUT
        else:
            return PinDirection.INOUT


def main():
    INFO = "Verilog code parser"
    VERSION = pyslang.__version__
    USAGE = "Usage: python verilog_parser.py file ..."
    log.basicConfig(level=log.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if len(sys.argv) != 2:
        log.error("Please provide a single Verilog file.")
        log.info(USAGE)
        sys.exit(1)

    filename = sys.argv[1]

    parser = VerilogParser()
    try:
        db = parser.parse_and_store_in_db(filename)
        if db:
            log.info("Parsing complete.")
            if db.top_module:
                log.info(f"Top module: {db.top_module.name}")
            else:
                log.warning("Could not determine top module.")
    except (RuntimeError, IOError) as e:
        log.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
