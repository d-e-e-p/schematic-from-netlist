import os
import sys
import logging
from optparse import OptionParser

import pyverilog
from pyverilog.vparser.ast import *
from pyverilog.vparser.parser import parse

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schematic_from_netlist.interfaces.netlist_database import (
    Instance,
    Module,
    Net,
    NetlistDatabase,
    NetType,
    Pin,
    PinDirection,
)
from schematic_from_netlist.graph.graph_partition import HypergraphPartitioner


class VerilogParser:
    def __init__(self):
        self.db = NetlistDatabase()
        self.module_ports = {}

    def parse_and_store_in_db(self, filelist, include_path=None, define=None):
        ast, directives = parse(filelist, preprocess_include=include_path, preprocess_define=define)
        self.store_ast_in_db(ast)
        return self.db

    def get_signal_name(self, signal):
        """Extract signal name from various AST node types"""
        if isinstance(signal, Identifier):
            return signal.name
        elif isinstance(signal, Pointer):
            var_name = self.get_signal_name(signal.var)
            ptr_str = self.get_signal_name(signal.ptr)
            return f"{var_name}[{ptr_str}]"
        elif isinstance(signal, Partselect):
            var_name = self.get_signal_name(signal.var)
            msb = self.get_signal_name(signal.msb) if signal.msb else ""
            lsb = self.get_signal_name(signal.lsb) if signal.lsb else ""
            return f"{var_name}[{msb}:{lsb}]"
        elif isinstance(signal, Concat):
            parts = [self.get_signal_name(part) for part in signal.list]
            return "{" + ",".join(parts) + "}"
        elif isinstance(signal, IntConst):
            return str(signal.value)
        else:
            return str(signal)

    def get_width_range(self, width_obj):
        """Extract range from Width object"""
        if hasattr(width_obj, "msb") and hasattr(width_obj, "lsb"):
            msb = self.get_signal_name(width_obj.msb)
            lsb = self.get_signal_name(width_obj.lsb)
            try:
                return int(msb), int(lsb)
            except (ValueError, TypeError):
                return msb, lsb
        return None, None

    def store_ast_in_db(self, ast):
        # First pass: Collect module definitions
        for node in ast.description.definitions:
            if isinstance(node, ModuleDef):
                module = Module(name=node.name)
                self.db.modules[node.name] = module
                if not self.db.top_module:
                    self.db.set_top_module(module)

                # Collect ports
                if node.portlist:
                    for port in node.portlist.ports:
                        if isinstance(port, Ioport):
                            port_name = port.first.name
                        else:
                            port_name = port.name
                        port_decl = self._find_decl_for_port(node, port_name)
                        direction = PinDirection.INOUT  # Default
                        if isinstance(port_decl, Input):
                            direction = PinDirection.INPUT
                        elif isinstance(port_decl, Output):
                            direction = PinDirection.OUTPUT

                        module.add_port(port_name, direction)
                        # Also add as a net
                        module.add_net(port_name)

        # Second pass: Populate modules with instances, nets, and connections
        for node in ast.description.definitions:
            if isinstance(node, ModuleDef):
                module = self.db.modules[node.name]
                self._populate_module(module, node)

        # Third pass: Create stub modules for undefined components
        for module in list(self.db.modules.values()):
            for inst in module.instances.values():
                if inst.module_ref not in self.db.modules:
                    logging.warning(f"Creating stub module for undefined component: {inst.module_ref}")
                    stub_module = Module(name=inst.module_ref)
                    # Infer ports from the instance's pins
                    for i, pin in enumerate(inst.pins.values()):
                        # Create a generic port, direction might be unknown (default to INOUT)
                        stub_module.add_port(f"PIN{i}", PinDirection.INOUT)
                    self.db.modules[inst.module_ref] = stub_module
        
        self.db._build_lookup_tables()
        return self.db

    def _find_decl_for_port(self, module_node, port_name):
        for item in module_node.items:
            if isinstance(item, Decl):
                for decl in item.list:
                    if hasattr(decl, "name") and decl.name == port_name:
                        return decl
        return None

    def _populate_module(self, module, module_node):
        # Process declarations (nets)
        for item in module_node.items:
            if isinstance(item, Decl):
                for decl in item.list:
                    if isinstance(decl, Wire):
                        module.add_net(decl.name, NetType.WIRE)
                    elif isinstance(decl, Reg):
                        module.add_net(decl.name, NetType.REG)
                    elif isinstance(decl, Supply):
                        if decl.value.value == "0":
                            module.add_net(decl.name, NetType.SUPPLY0)
                        else:
                            module.add_net(decl.name, NetType.SUPPLY1)

        # Process instances
        for item in module_node.items:
            if isinstance(item, InstanceList):
                for inst in item.instances:
                    module_ref = self.db.modules.get(inst.module)
                    if inst.array:
                        msb, lsb = self.get_width_range(inst.array)
                        for i in range(lsb, msb + 1):
                            inst_name = f"{inst.name}[{i}]"
                            instance = module.add_instance(inst_name, inst.module)
                            for j, port_conn in enumerate(inst.portlist):
                                port_name = port_conn.portname
                                if not port_name:
                                    if module_ref and j < len(module_ref.ports):
                                        port_name = list(module_ref.ports.keys())[j]
                                    else:
                                        port_name = f"PIN{j}"

                                net_name = self.get_signal_name(port_conn.argname)
                                if "[" in net_name and "]" in net_name:
                                    base_name = net_name.split("[")[0]
                                    connected_net_name = f"{base_name}[{i}]"
                                else:
                                    connected_net_name = net_name

                                net = module.nets.get(connected_net_name)
                                if not net:
                                    net = module.add_net(connected_net_name)

                                direction = PinDirection.INOUT
                                if module_ref and port_name in module_ref.ports:
                                    direction = module_ref.ports[port_name].direction

                                pin = instance.add_pin(port_name, direction)
                                instance.connect_pin(port_name, net)
                    else:
                        instance = module.add_instance(inst.name, inst.module)
                        for i, port_conn in enumerate(inst.portlist):
                            port_name = port_conn.portname
                            if not port_name:
                                if module_ref and i < len(module_ref.ports):
                                    port_name = list(module_ref.ports.keys())[i]
                                else:
                                    port_name = f"PIN{i}"

                            if not port_name:
                                continue

                            net_name = self.get_signal_name(port_conn.argname)
                            net = module.nets.get(net_name)
                            if not net:
                                net = module.add_net(net_name)

                            direction = PinDirection.INOUT
                            if module_ref and port_name in module_ref.ports:
                                direction = module_ref.ports[port_name].direction

                            pin = instance.add_pin(port_name, direction)
                            instance.connect_pin(port_name, net)

    def list_connections(self, ast):
        """List all net connections in the format: net inst/pin1 inst/pin2 ..."""
        connections = self.find_net_connections(ast)

        logging.info("\n=== Net Connections ===")
        for net, connected_ports in connections.items():
            if len(connected_ports) > 1:  # Only show nets with multiple connections
                logging.info(f"{net} {' '.join(connected_ports)}")

    def find_net_connections(self, ast):
        """
        Traverse the AST to find net connections between instances and pins
        Returns a dictionary mapping nets to connected instance/pin pairs
        """
        connections = {}

        # First pass: collect module port definitions
        def collect_module_ports(node):
            if isinstance(node, ModuleDef):
                module_name = node.name
                ports = []

                if hasattr(node, "portlist") and node.portlist:
                    for port in node.portlist.ports:
                        if hasattr(port, "name"):
                            ports.append(port.name)

                self.module_ports[module_name] = ports

            if hasattr(node, "children"):
                for child in node.children():
                    if child:
                        collect_module_ports(child)

        def traverse_node(node):
            if isinstance(node, ModuleDef):
                for child in node.children():
                    traverse_node(child)
            elif isinstance(node, InstanceList):
                for instance in node.instances:
                    if hasattr(instance, "portlist") and instance.portlist:
                        traverse_instance_ports(instance)
            elif hasattr(node, "children"):
                for child in node.children():
                    if child:
                        traverse_node(child)

        def traverse_instance_ports(instance):
            instance_name = instance.name if instance.name else "unnamed_inst"
            module_name = instance.module
            is_array_instance = hasattr(instance, "array") and instance.array

            expected_ports = self.module_ports.get(module_name, [])

            if hasattr(instance, "portlist") and instance.portlist:
                for i, port in enumerate(instance.portlist):
                    if isinstance(port, PortArg):
                        port_name = port.portname or (expected_ports[i] if i < len(expected_ports) else f"port{i}")

                        if port.argname:
                            net_name = self.get_signal_name(port.argname)
                            connection_str = f"{instance_name}/{port_name}"
                            if net_name not in connections:
                                connections[net_name] = []
                            connections[net_name].append(connection_str)

        collect_module_ports(ast)
        traverse_node(ast)
        return connections


def main():
    INFO = "Verilog code parser"
    VERSION = pyverilog.__version__
    USAGE = "Usage: python example_parser.py file ..."
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    def showVersion():
        logging.info(INFO)
        logging.info(VERSION)
        logging.info(USAGE)
        sys.exit()

    optparser = OptionParser()
    optparser.add_option("-v", "--version", action="store_true", dest="showversion", default=False, help="Show the version")
    optparser.add_option("-I", "--include", dest="include", action="append", default=[], help="Include path")
    optparser.add_option("-D", dest="define", action="append", default=[], help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = args
    if options.showversion:
        showVersion()

    for f in filelist:
        if not os.path.exists(f):
            raise IOError("file not found: " + f)

    if len(filelist) == 0:
        showVersion()

    parser = VerilogParser()
    db = parser.parse_and_store_in_db(filelist, options.include, options.define)

    hypergraph_data = db.build_hypergraph_data()
    
    # Partition the hypergraph
    k = 2  # Number of partitions
    ini_file = "km1_kKaHyPar_sea20.ini"
    partitioner = HypergraphPartitioner(hypergraph_data)
    partition = partitioner.run_partitioning(k, ini_file)

    # Dump the partitioned graph to a JSON file
    partitioner.dump_graph_to_json(k, partition)


    db.generate_ids()
    import pprint
    logging.debug(pprint.pformat(db.id_by_netname))
    logging.debug(pprint.pformat(db.id_by_instname))


if __name__ == "__main__":
    main()
