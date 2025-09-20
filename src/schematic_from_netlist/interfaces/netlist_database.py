from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from schematic_from_netlist.graph.graph_partition import Edge, HypergraphData


class PinDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class NetType(Enum):
    WIRE = "wire"
    REG = "reg"
    SUPPLY0 = "supply0"
    SUPPLY1 = "supply1"


@dataclass
class Pin:
    """Represents a pin on an instance"""

    name: str
    direction: PinDirection
    instance: "Instance"
    net: Optional["Net"] = None
    bit_width: int = 1
    bit_range: Optional[tuple] = None  # (msb, lsb) for vectors

    def __post_init__(self):
        self.full_name = f"{self.instance.full_name}/{self.name}"

    def __hash__(self):
        return hash(self.full_name)


@dataclass
class Net:
    """Represents a net (wire) in the design"""

    name: str
    module: "Module"
    net_type: NetType = NetType.WIRE
    bit_width: int = 1
    bit_range: Optional[tuple] = None  # (msb, lsb) for vectors
    pins: Set[Pin] = field(default_factory=set)
    id: int = -1
    num_conn: int = 0

    def __post_init__(self):
        self.full_name = f"{self.module.full_name}.{self.name}"
        self.drivers: Set[Pin] = set()  # Output pins driving this net
        self.loads: Set[Pin] = set()  # Input pins loading this net
        self.connections = set()  # drivers + loads

    def add_pin(self, pin: Pin):
        """Add a pin connection to this net"""
        self.pins.add(pin)
        pin.net = self

        self.connections.add(pin)
        self.num_conn += 1
        if pin.direction == PinDirection.OUTPUT:
            self.drivers.add(pin)
        elif pin.direction == PinDirection.INPUT:
            self.loads.add(pin)
        else:  # INOUT
            self.drivers.add(pin)
            self.loads.add(pin)

    def remove_pin(self, pin: Pin):
        """Remove a pin connection from this net"""
        self.pins.discard(pin)
        self.connections.discard(pin)
        self.drivers.discard(pin)
        self.loads.discard(pin)
        self.num_conn -= 1
        pin.net = None
        print(f"after remove_pin {self.name=} {self.num_conn=}")

    def get_fanout(self) -> int:
        """Get the fanout (number of loads) on this net"""
        return len(self.loads)

    def get_connections(self) -> int:
        """Get the num of connected pins for this net"""
        return len(self.connections)

    def is_floating(self) -> bool:
        """Check if net has no drivers"""
        return len(self.drivers) == 0

    def has_multiple_drivers(self) -> bool:
        """Check if net has multiple drivers (potential conflict)"""
        return len(self.drivers) > 1

    def __hash__(self):
        return hash(self.full_name)


@dataclass
class Instance:
    """Represents an instance of a module"""

    name: str
    module_ref: str  # Reference to module definition
    parent_module: "Module"
    pins: Dict[str, Pin] = field(default_factory=dict)
    parameters: Dict[str, any] = field(default_factory=dict)
    id: int = -1

    def __post_init__(self):
        self.full_name = f"{self.parent_module.full_name}/{self.name}"

    def add_pin(self, pin_name: str, direction: PinDirection, net: Optional[Net] = None) -> Pin:
        """Add a pin to this instance"""
        pin = Pin(pin_name, direction, self)
        self.pins[pin_name] = pin
        if net:
            net.add_pin(pin)
        return pin

    def connect_pin(self, pin_name: str, net: Net):
        """Connect a pin to a net"""
        if pin_name in self.pins:
            if self.pins[pin_name].net:
                self.pins[pin_name].net.remove_pin(self.pins[pin_name])
            net.add_pin(self.pins[pin_name])

    def get_connected_nets(self) -> Set[Net]:
        """Get all nets connected to this instance"""
        nets = set()
        for pin in self.pins.values():
            if pin.net:
                nets.add(pin.net)
        return nets

    def __hash__(self):
        return hash(self.full_name)


@dataclass
class Module:
    """Represents a module definition"""

    name: str
    parent_module: Optional["Module"] = None
    instances: Dict[str, Instance] = field(default_factory=dict)
    nets: Dict[str, Net] = field(default_factory=dict)
    ports: Dict[str, Pin] = field(default_factory=dict)  # Module interface ports
    child_modules: Dict[str, "Module"] = field(default_factory=dict)

    def __post_init__(self):
        if self.parent_module:
            self.full_name = f"{self.parent_module.full_name}/{self.name}"
        else:
            self.full_name = self.name  # Top level module

    def add_net(
        self,
        net_name: str,
        net_type: NetType = NetType.WIRE,
        bit_width: int = 1,
        bit_range: Optional[tuple] = None,
    ) -> Net:
        """Add a net to this module"""
        net = Net(
            name=net_name,
            module=self,
            net_type=net_type,
            bit_width=bit_width,
            bit_range=bit_range,
        )
        self.nets[net_name] = net
        return net

    def remove_net(self, net_name):
        if net_name in self.nets:
            del self.nets[net_name]
        return net

    def add_instance(self, inst_name: str, module_ref: str) -> Instance:
        """Add an instance to this module"""
        instance = Instance(name=inst_name, module_ref=module_ref, parent_module=self)
        self.instances[inst_name] = instance
        return instance

    def add_port(self, port_name: str, direction: PinDirection) -> Pin:
        """Add a port to this module"""
        # Create a dummy instance for the module port
        module_inst = Instance(f"__{self.name}__", self.name, self)
        port = Pin(port_name, direction, module_inst)
        self.ports[port_name] = port
        return port

    def get_all_instances(self, recursive: bool = True) -> Dict[str, Instance]:
        """Get all instances in this module (and children if recursive)"""
        all_instances = self.instances.copy()
        if recursive:
            for child_module in self.child_modules.values():
                child_instances = child_module.get_all_instances(recursive=True)
                all_instances.update(child_instances)
        return all_instances

    def get_all_nets(self, recursive: bool = True) -> Dict[str, Net]:
        """Get all nets in this module (and children if recursive)"""
        all_nets = self.nets.copy()
        if recursive:
            for child_module in self.child_modules.values():
                child_nets = child_module.get_all_nets(recursive=True)
                all_nets.update(child_nets)
        return all_nets


class NetlistDatabase:
    """Main database class for the hierarchical netlist"""

    def __init__(self, fanout_threshold: int = 15):
        self.fanout_threshold = fanout_threshold
        self.top_module: Optional[Module] = None
        self.modules: Dict[str, Module] = {}  # Module definitions

        self.inst_by_name: Dict[str, Instance] = {}  # Fast lookup by full name
        self.nets_by_name: Dict[str, Net] = {}  # Fast lookup by full name
        self.pins_by_name: Dict[str, Pin] = {}  # Fast lookup by full name

        self.inst_by_id: Dict[int, Instance] = {}
        self.nets_by_id: Dict[int, Net] = {}

        self.id_by_instname: Dict[str, int] = {}
        self.id_by_netname: Dict[str, int] = {}

        self.instname_by_id: Dict[int, str] = {}
        self.netname_by_id: Dict[int, str] = {}

        self.buffered_nets_log: Dict[str, Dict] = {}

        self.inserted_buf_prefix = "buf⊕_"

    def set_top_module(self, module: Module):
        """Set the top-level module"""
        self.top_module = module
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build fast lookup tables for instances, nets, and pins"""
        if not self.top_module:
            return

        # Clear existing tables
        self.inst_by_name.clear()
        self.nets_by_name.clear()
        self.pins_by_name.clear()

        def traverse_module(module: Module):
            # Add all instances
            for instance in module.instances.values():
                self.inst_by_name[instance.name] = instance
                # Add all pins of this instance
                for pin in instance.pins.values():
                    self.pins_by_name[pin.full_name] = pin

            # Add all nets
            for net in module.nets.values():
                self.nets_by_name[net.name] = net

            # Add module ports as pins
            for port in module.ports.values():
                self.pins_by_name[port.name] = port

            # Recursively traverse child modules
            for child_module in module.child_modules.values():
                traverse_module(child_module)

        traverse_module(self.top_module)

    # Query Methods
    def find_net(self, net_name: str) -> Optional[Net]:
        """Find a net by full name"""
        return self.nets_by_name.get(net_name)

    def find_instance(self, instance_name: str) -> Optional[Instance]:
        """Find an instance by full name"""
        return self.inst_by_name.get(instance_name)

    def find_pin(self, pin_name: str) -> Optional[Pin]:
        """Find a pin by full name"""
        return self.pins_by_name.get(pin_name)

    def get_net_connections(self, net: Net) -> Dict[str, List[Pin]]:
        """Get all connections to a net, organized by type"""
        return {
            "drivers": list(net.drivers),
            "loads": list(net.loads),
            "all_pins": list(net.pins),
        }

    def trace_net_path(self, start_pin: Pin, max_depth: int = 10) -> List[Pin]:
        """Trace the path from a pin through connected nets"""
        path = [start_pin]
        current_pin = start_pin
        depth = 0

        while depth < max_depth and current_pin.net:
            net = current_pin.net
            # Find the next pin in the path
            if current_pin.direction == PinDirection.OUTPUT:
                # Follow to inputs (loads)
                next_pins = list(net.loads)
            else:
                # Follow to outputs (drivers)
                next_pins = list(net.drivers)

            if next_pins:
                current_pin = next_pins[0]  # Take first connection
                path.append(current_pin)
            else:
                break
            depth += 1

        return path

    def get_fanout_tree(self, driver_pin: Pin) -> Dict:
        """Get the complete fanout tree from a driver pin"""
        if driver_pin.direction != PinDirection.OUTPUT or not driver_pin.net:
            return {}

        net = driver_pin.net
        fanout_tree = {
            "driver": driver_pin.full_name,
            "net": net.full_name,
            "loads": [pin.full_name for pin in net.loads],
            "fanout": len(net.loads),
        }

        return fanout_tree

    def find_timing_paths(self, start_instance: Instance, end_instance: Instance) -> List[List[Pin]]:
        """Find timing paths between two instances"""
        # This is a simplified version - real timing analysis is much more complex
        paths = []

        def dfs_path(current_pin: Pin, target_instance: Instance, current_path: List[Pin]):
            if len(current_path) > 20:  # Prevent infinite loops
                return

            if current_pin.instance == target_instance:
                paths.append(current_path + [current_pin])
                return

            if current_pin.net:
                net = current_pin.net
                if current_pin.direction == PinDirection.OUTPUT:
                    next_pins = net.loads
                else:
                    next_pins = net.drivers

                for next_pin in next_pins:
                    if next_pin not in current_path:
                        dfs_path(next_pin, target_instance, current_path + [current_pin])

        # Start from all output pins of start instance
        for pin in start_instance.pins.values():
            if pin.direction == PinDirection.OUTPUT:
                dfs_path(pin, end_instance, [])

        return paths

    def generate_ids(self):
        net_id_counter = 0
        inst_id_counter = 0

        """
        for name, inst in self.inst_by_name.items():
            inst.id = inst_id_counter
            self.id_by_instname[name] = inst_id_counter
            self.inst_by_id[inst_id_counter] = inst
            inst_id_counter += 1
        """

        for name, net in self.nets_by_name.items():
            if net.num_conn > 0 and net.num_conn < self.fanout_threshold:
                net.id = net_id_counter
                self.id_by_netname[name] = net_id_counter
                self.nets_by_id[net_id_counter] = net
                self.netname_by_id[net_id_counter] = name
                net_id_counter += 1
                for pin in net.pins:
                    instname = pin.instance.name
                    if instname not in self.id_by_instname:
                        self.id_by_instname[instname] = inst_id_counter
                        self.instname_by_id[inst_id_counter] = instname
                        self.inst_by_id[inst_id_counter] = pin.instance
                        pin.instance.id = inst_id_counter
                        inst_id_counter += 1
                        pass

    def buffer_multi_fanout_nets(self):
        """Inserts buffers on nets with fanout > 1"""

        if not self.top_module:
            return

        nets_to_buffer = [net for net in self.nets_by_name.values() if net.num_conn > 2 and net.num_conn < self.fanout_threshold]

        for net in nets_to_buffer:
            original_net_name = net.name
            self.buffered_nets_log[original_net_name] = {"pins": net.pins, "buffer_insts": [], "new_nets": []}

            buffer_name = f"{self.inserted_buf_prefix}{original_net_name}"
            buffer_inst = self.top_module.add_instance(buffer_name, "FANOUT_BUFFER")
            self.buffered_nets_log[original_net_name]["buffer_insts"].append(buffer_inst)

            pins_to_buffer = list(net.pins)
            for i, pin in enumerate(pins_to_buffer):
                # New net for buffer output
                new_net_name = f"{original_net_name}_fanout_buffer_{i}"
                new_net = self.top_module.add_net(new_net_name)

                buf_inout_pin = buffer_inst.add_pin(f"IO{i}", PinDirection.INOUT)
                new_net.add_pin(buf_inout_pin)

                # Disconnect load from original net and connect to new net
                net.remove_pin(pin)
                new_net.add_pin(pin)
                self.buffered_nets_log[original_net_name]["new_nets"].append(new_net)

        self._build_lookup_tables()

    def remove_buffers(self):
        """Removes buffers inserted by buffer_multi_fanout_nets"""
        if not self.top_module:
            return

        for original_net_name, log in self.buffered_nets_log.items():
            original_net = self.find_net(f"{self.top_module.name}.{original_net_name}")
            if not original_net:
                continue

            # Reconnect original loads
            for load_pin in log["loads"]:
                original_net.add_pin(load_pin)

            # Delete buffer instances and disconnect their input pins
            for buffer_inst in log["buffer_insts"]:
                for pin in buffer_inst.pins.values():
                    if pin.net:
                        pin.net.remove_pin(pin)
                del self.top_module.instances[buffer_inst.name]

            # Delete new nets
            for new_net in log["new_nets"]:
                del self.top_module.nets[new_net.name]

        self.buffered_nets_log.clear()
        self._build_lookup_tables()

    def get_design_statistics(self) -> Dict:
        """Get overall design statistics"""
        stats = {"total_instances": len(self.inst_by_name), "total_nets": len(self.nets_by_name), "total_pins": len(self.pins_by_name), "modules": len(self.modules), "floating_nets": 0, "multi_driven_nets": 0, "max_fanout": 0, "avg_fanout": 0}

        total_fanout = 0
        for net in self.nets_by_name.values():
            fanout = net.get_fanout()
            total_fanout += fanout
            stats["max_fanout"] = max(stats["max_fanout"], fanout)

            if net.is_floating():
                stats["floating_nets"] += 1
            if net.has_multiple_drivers():
                stats["multi_driven_nets"] += 1

        if len(self.nets_by_name) > 0:
            stats["avg_fanout"] = total_fanout // len(self.nets_by_name)

        return stats

    def build_hypergraph_data(self) -> HypergraphData:
        """Builds the hypergraph data structure for KaHyPar."""
        self.generate_ids()

        num_nodes = len(self.inst_by_id)
        num_edges = len(self.nets_by_id)

        sorted_nets = sorted(self.nets_by_id.values(), key=lambda net: net.id)

        edge_vector = []
        index_vector = [0]

        for net in sorted_nets:
            connected_instance_ids = sorted(list({pin.instance.id for pin in net.pins}))
            connected_instpin_names = [f"{pin.instance.name}/{pin.name}" for pin in net.pins]
            edge_vector.extend(connected_instance_ids)
            index_vector.append(len(edge_vector))
            # print(f"connecting {net.name=} conn {net.num_conn}: {connected_instance_ids=} {connected_instpin_names=}")
            # print(f"{len(edge_vector)=} {len(index_vector)=} {edge_vector=} {index_vector=}")

            if index_vector[-1] != len(edge_vector):
                print(f"Bad sentinel: {index_vector[-1]=} != {len(edge_vector)=}")
        if len(index_vector) != num_edges + 1:
            print(f"Mismatch: {len(index_vector)=} vs {num_edges + 1=}")

        return HypergraphData(
            num_nodes=num_nodes,
            num_edges=num_edges,
            index_vector=index_vector,
            edge_vector=edge_vector,
        )

    def get_edges_between_nodes(self, nodes):
        def add_edge(name, src, dst, instlist):
            if src in instlist and dst in instlist:
                # skip buffer nets
                if name.startswith(self.inserted_buf_prefix):
                    edge = Edge(src=src.name, dst=dst.name)
                else:
                    edge = Edge(src=src.name, dst=dst.name, name=name)
                return edge
            return None

        print(nodes)
        inst_list = []
        for id in nodes:
            inst_list.append(self.inst_by_id[id])

        # find all nets connected to these instances
        net_list = []
        seen_nets = set()
        for inst in inst_list:
            for net in inst.get_connected_nets():
                if net.name not in seen_nets:
                    net_list.append(net)
                    seen_nets.add(net.name)

        # print(f"{seen_nets=}")
        edges = []
        for net in net_list:
            if net.num_conn < 2:
                continue
            elif net.num_conn > self.fanout_threshold:
                continue
            else:
                # multi-pin net so generate all pairwise connections
                conn_inst_list = [pin.instance for pin in net.pins]
                for i in range(len(conn_inst_list)):
                    for j in range(i + 1, len(conn_inst_list)):
                        src, dst = conn_inst_list[i], conn_inst_list[j]
                        if edge := add_edge(net.name, src, dst, inst_list):
                            edges.append(edge)

        return edges


# Example usage and helper functions
def create_example_netlist():
    """Create an example netlist for testing"""
    db = NetlistDatabase()

    # Create top module
    top_module = Module("TOP")
    db.set_top_module(top_module)
    db.modules["TOP"] = top_module

    # Add nets
    val_net = top_module.add_net("VAL", NetType.WIRE, 4, (3, 0))
    led0_net = top_module.add_net("LED0", NetType.WIRE, 4, (3, 0))

    # Add instances
    sub0_inst = top_module.add_instance("inst_sub0", "SUB")
    sub0_inst.add_pin("VAL", PinDirection.INPUT, val_net)
    sub0_inst.add_pin("LED", PinDirection.OUTPUT, led0_net)

    return db


# Usage examples
if __name__ == "__main__":
    # Create example database
    db = create_example_netlist()

    # Query examples
    print("Design Statistics:")
    stats = db.get_design_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nNet connections:")
    for net_name, net in db.nets_by_name.items():
        connections = db.get_net_connections(net)
        print(f"  {net_name}: {len(connections['drivers'])} drivers, {len(connections['loads'])} loads")
