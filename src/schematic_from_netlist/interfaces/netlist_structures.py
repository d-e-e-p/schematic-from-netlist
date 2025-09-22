from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set


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
        self.connections = set()  # drivers + loads TODO: use pins instead

    def add_pin(self, pin: Pin):
        """Add a pin connection to this net"""
        if pin.net is not None:
            if pin.net == self:
                return
            print(f"instrumentation: Pin {pin.full_name} is on {pin.net.full_name}, removing it.")
            pin.net.remove_pin(pin)

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
        print(f"after add_pin {self.name=} {self.num_conn=} {pin.full_name=}")

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
    parent_module: Module
    pins: Dict[str, Pin] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
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
            pin = self.pins[pin_name]
            if pin.net:
                pin.net.remove_pin(pin)
            net.add_pin(pin)

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

    def remove_net(self, net_name: str) -> Optional[Net]:
        """Remove a net from this module"""
        if net_name in self.nets:
            net = self.nets[net_name]
            del self.nets[net_name]
            return net
        return None

    def add_instance(self, inst_name: str, module_ref: str) -> Instance:
        """Add an instance to this module"""
        instance = Instance(name=inst_name, module_ref=module_ref, parent_module=self)
        self.instances[inst_name] = instance
        return instance

    def add_port(self, port_name: str, direction: PinDirection) -> Pin:
        """Add a port to this module"""
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
