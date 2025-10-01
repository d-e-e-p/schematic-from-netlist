import logging as log
import os
import re
import sys

import pyslang
from pyslang import DefinitionSymbol, InstanceSymbol

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
            return name.replace("\\", "").strip()
        return name

    def parse_and_store_in_db(self, filename):
        """Parse SystemVerilog file and handle diagnostics"""
        if not os.path.exists(filename):
            raise IOError(f"File not found: {filename}")

        tree = pyslang.SyntaxTree.fromFile(filename)
        compilation = pyslang.Compilation()
        compilation.addSyntaxTree(tree)

        all_diags = list(compilation.getAllDiagnostics())
        has_errors = False
        for diag in all_diags:
            if diag.isError():
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

        # Pass 1: Discover all module definitions and declare all nets/ports
        for instance in root.topInstances:
            self._pass1_declare_modules_and_nets(instance)

        # Pass 2: Connect all instances
        for instance in root.topInstances:
            self._pass2_connect_instances(instance, parent_module_db=None, compilation=compilation)

        if not self.db.top_module and root.topInstances:
            top_module_name = self._clean_name(root.topInstances[0].definition.name)
            if top_module_name in self.db.modules:
                self.db.set_top_module(self.db.modules[top_module_name])

    def _pass1_declare_modules_and_nets(self, instance):
        """
        PASS 1: Recursively traverse the design, creating Module objects and populating
                them with all their declared ports, nets, and buses.
        """
        definition = instance.definition
        module_name = self._clean_name(definition.name)

        if module_name in self.db.modules:
            return  # Already processed this module definition

        module_db = Module(name=module_name)
        self.db.modules[module_name] = module_db

        scope = definition.parentScope
        if not scope:
            return

        for member in scope:
            if member.kind == pyslang.SymbolKind.Port:
                self._create_bus_and_nets(module_db, member, is_port=True)
            elif member.kind in (pyslang.SymbolKind.Variable, pyslang.SymbolKind.Net):
                self._create_bus_and_nets(module_db, member, is_port=False)
            elif member.kind == pyslang.SymbolKind.Instance:
                self._pass1_declare_modules_and_nets(member) # Recurse on the instance symbol

    def _pass2_connect_instances(self, instance, parent_module_db, compilation):
        """
        PASS 2: Recursively traverse the design, creating Instance objects and
                connecting their pins to the now-fully-declared nets.
        """
        definition = instance.definition
        module_name = self._clean_name(definition.name)
        module_db = self.db.modules.get(module_name)

        if not module_db:
            log.error(f"Module '{module_name}' not found during Pass 2.")
            return

        if parent_module_db:
            inst_name = self._clean_name(instance.name)
            instance_db = parent_module_db.add_instance(inst_name, module_db, module_name)
            self._process_instance_connections(instance_db, instance, parent_module_db, compilation)

        scope = definition.parentScope
        if not scope:
            return

        for member in scope:
            if member.kind == pyslang.SymbolKind.Instance:
                self._pass2_connect_instances(member, module_db, compilation) # Recurse on the instance symbol

    def _get_int_value(self, val):
        """Safely get an integer value from a pyslang object."""
        if isinstance(val, int):
            return val
        if hasattr(val, 'value'):
            return int(val.value)
        return int(val)

    def _create_bus_and_nets(self, module_db, member, is_port=False):
        name = self._clean_name(member.name)
        is_bus = member.type.isPackedArray or member.type.isUnpackedArray
        parent_bus = None
        if is_bus:
            bit_range = member.type.getBitVectorRange()
            msb = self._get_int_value(bit_range.left)
            lsb = self._get_int_value(bit_range.right)
            parent_bus = Bus(name=name, bit_range=(msb, lsb), bit_width=abs(msb - lsb) + 1)
            module_db.busses[name] = parent_bus
            for i in range(min(lsb, msb), max(lsb, msb) + 1):
                bit_net_name = f"{name}[{i}]"
                if bit_net_name not in module_db.nets:
                    bit_net = module_db.add_net(bit_net_name)
                    bit_net.module = module_db
                    bit_net.parent_bus = parent_bus
                    bit_net.bit_index = i
                    parent_bus.bit_nets[i] = bit_net
        else:
            if name not in module_db.nets:
                net = module_db.add_net(name)
                net.module = module_db
        if is_port:
            direction = self._get_pyslang_port_direction(member.direction)
            port = module_db.add_port(name, direction)
            port.parent_bus = parent_bus

    def _resolve_expression_to_bit_list(self, expression, parent_module_db):
        resolved_nets = []
        if not hasattr(expression, 'kind'):
            resolved_nets.append(self._clean_name(str(expression.syntax)))
            return resolved_nets
        if expression.kind == pyslang.ExpressionKind.Concatenation:
            for operand in reversed(expression.operands):
                resolved_nets.extend(self._resolve_expression_to_bit_list(operand, parent_module_db))
        elif expression.kind == pyslang.ExpressionKind.RangeSelect:
            base_symbol_name = self._clean_name(expression.value.symbol.name)
            msb = self._get_int_value(expression.left)
            lsb = self._get_int_value(expression.right)
            if msb > lsb:
                for i in range(lsb, msb + 1):
                    resolved_nets.append(f"{base_symbol_name}[{i}]")
            else:
                for i in range(msb, lsb + 1):
                    resolved_nets.append(f"{base_symbol_name}[{i}]")
        elif expression.kind == pyslang.ExpressionKind.NamedValue:
            symbol = expression.symbol
            symbol_name = self._clean_name(str(expression.syntax))
            is_bus = symbol and symbol.type.isPackedArray
            if not is_bus and parent_module_db and symbol_name in parent_module_db.busses:
                is_bus = True
            if is_bus:
                bus = parent_module_db.busses[symbol_name]
                msb, lsb = bus.bit_range
                for i in range(min(lsb, msb), max(lsb, msb) + 1):
                    resolved_nets.append(f"{symbol_name}[{i}]")
            else:
                resolved_nets.append(symbol_name)
        elif expression.kind == pyslang.ExpressionKind.IntegerLiteral:
            if self._get_int_value(expression) == 0:
                resolved_nets.append("GND")
            elif self._get_int_value(expression) == 1:
                resolved_nets.append("VCC")
            else:
                resolved_nets.append(self._clean_name(str(expression.syntax)))
        else:
            resolved_nets.append(self._clean_name(str(expression.syntax)))
        return resolved_nets

    def _process_instance_connections(self, instance_db, instance, parent_module_db, compilation):
        for conn in instance.portConnections:
            port_name = self._clean_name(conn.port.name)
            direction = self._get_pyslang_port_direction(conn.port.direction)

            if not conn.expression:
                log.warning(f"Skipping port '{port_name}' on instance '{instance_db.name}' with no expression.")
                continue

            if instance_db.module.is_stub and port_name not in instance_db.module.ports:
                instance_db.module.add_port(port_name, direction)

            is_port_bus = conn.port.type.isPackedArray or conn.port.type.isUnpackedArray
            if not is_port_bus and port_name in instance_db.module.busses:
                 is_port_bus = True

            if is_port_bus:
                port_bus = instance_db.module.busses.get(port_name)
                if port_bus:
                    port_msb, port_lsb = port_bus.bit_range
                    port_width = port_bus.bit_width
                else:
                    port_range = conn.port.type.getBitVectorRange()
                    port_msb = self._get_int_value(port_range.left)
                    port_lsb = self._get_int_value(port_range.right)
                    port_width = abs(port_msb - port_lsb) + 1

                rhs_nets = self._resolve_expression_to_bit_list(conn.expression, parent_module_db)

                if len(rhs_nets) != port_width:
                    log.warning(f"Width mismatch for port '{port_name}' on instance '{instance_db.name}': Port is {port_width} bits, but expression resolves to {len(rhs_nets)} bits ({rhs_nets}). Connection skipped.")
                    continue

                for i in range(port_width):
                    port_bit_index = min(port_lsb, port_msb) + i
                    bit_port_name = f"{port_name}[{port_bit_index}]"
                    bit_net_name = rhs_nets[i]
                    net = parent_module_db.nets.get(bit_net_name)
                    if not net:
                        log.warning(f"Net '{bit_net_name}' not found in module '{parent_module_db.name}'. Creating it.")
                        net = parent_module_db.add_net(bit_net_name)
                        net.module = parent_module_db
                    instance_db.add_pin(bit_port_name, direction)
                    instance_db.connect_pin(bit_port_name, net)
            else:
                net_name = self._clean_name(str(conn.expression.syntax))
                if not net_name:
                    log.warning(f"Could not determine net name for port '{port_name}' on instance '{instance_db.name}'.")
                    continue
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