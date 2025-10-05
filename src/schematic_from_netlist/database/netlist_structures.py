from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

from schematic_from_netlist.database.physical_structures import (
    DesignPhysical,
    InstancePhysical,
    ModulePhysical,
    NetPhysical,
    PinPhysical,
    PortPhysical,
)


# -----------------------------
# Enums
# -----------------------------
class PinDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class NetType(Enum):
    WIRE = "wire"
    REG = "reg"
    SUPPLY0 = "supply0"
    SUPPLY1 = "supply1"


# -----------------------------
# Bus
# -----------------------------
@dataclass
class Bus:
    name: str = ""
    msb: Optional[int] = None
    lsb: Optional[int] = None


# -----------------------------
# Port
# -----------------------------
@dataclass
class Port:
    """Represents ports on modules"""

    name: str
    direction: PinDirection
    module: Module
    bus: Optional[Bus] = None  # Reference to the full Bus object (Layer 1)
    draw: PortPhysical = field(default_factory=PortPhysical)


# -----------------------------
# Pin
# -----------------------------
@dataclass
class Pin:
    """Represents a pin on an instance"""

    name: str
    direction: PinDirection
    instance: Instance
    net: Optional["Net"] = None
    bus: Optional[Bus] = None  # Reference to the full Bus object (Layer 1)
    bit_index: Optional[int] = None  # The specific index within the parent bus (e.g., 3)
    draw: PinPhysical = field(default_factory=PinPhysical)

    def __post_init__(self):
        self.full_name = f"{self.instance.full_name}/{self.name}"

    def __hash__(self):
        return hash(self.full_name)


# -----------------------------
# Net
# -----------------------------
@dataclass
class Net:
    """Represents a net (wire) in the design"""

    name: str
    module: "Module"
    net_type: NetType = NetType.WIRE
    bus: Optional[Bus] = field(default=None)

    pins: Set[Pin] = field(default_factory=set)
    id: int = -1
    num_conn: int = 0
    is_buffer_wire: bool = False
    buffer_original_netname: Optional[str] = None
    draw: NetPhysical = field(default_factory=NetPhysical)

    def __post_init__(self):
        self.full_name = f"{self.module.full_name}.{self.name}"
        self.drivers: Set[Pin] = set()
        self.loads: Set[Pin] = set()
        self.connections: Set[Pin] = set()

    def add_pin(self, pin: Pin):
        if pin.net is not None and pin.net != self:
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
        logging.debug(f"after add_pin {self.name=} {self.num_conn=} {pin.full_name=}")

    def remove_pin(self, pin: Pin):
        self.pins.discard(pin)
        self.connections.discard(pin)
        self.drivers.discard(pin)
        self.loads.discard(pin)
        self.num_conn -= 1
        pin.net = None
        logging.debug(f"after remove_pin {self.name=} {self.num_conn=}")

    def get_fanout(self) -> int:
        return len(self.loads)

    def get_connections(self) -> int:
        return len(self.connections)

    def is_floating(self) -> bool:
        return len(self.drivers) == 0

    def has_multiple_drivers(self) -> bool:
        return len(self.drivers) > 1

    def __hash__(self):
        return hash(self.full_name)


# -----------------------------
# Instance
# -----------------------------
@dataclass
class Instance:
    """Represents an instance of a module"""

    name: str
    module: Module
    module_ref: str
    parent_module: Module
    pins: Dict[str, Pin] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    bus: Optional[Bus] = None
    id: int = -1
    partition: int = -1
    orient: str = "R0"
    is_buffer: bool = False
    buffer_original_netname: Optional[str] = None
    module_ref_uniq: Optional[str] = None
    draw: InstancePhysical = field(default_factory=InstancePhysical)

    def __post_init__(self):
        self.full_name = f"{self.parent_module.full_name}/{self.name}"

    def add_pin(self, pin_name: str, direction: PinDirection, net: Optional[Net] = None) -> Pin:
        pin = Pin(pin_name, direction, self)
        self.pins[pin_name] = pin
        if net:
            net.add_pin(pin)
        return pin

    def connect_pin(self, pin_name: str, net: Net):
        if pin_name in self.pins:
            pin = self.pins[pin_name]
            if pin.net:
                pin.net.remove_pin(pin)
            net.add_pin(pin)

    def get_connected_nets(self) -> Set[Net]:
        return {pin.net for pin in self.pins.values() if pin.net}

    def __hash__(self):
        return hash(self.full_name)


# -----------------------------
# Module
# -----------------------------
@dataclass
class Module:
    name: str
    parent_module: Optional[Module] = None
    instances: Dict[str, Instance] = field(default_factory=dict)
    nets: Dict[str, Net] = field(default_factory=dict)
    ports: Dict[str, Port] = field(default_factory=dict)
    pins: Dict[str, Pin] = field(default_factory=dict)
    child_modules: Dict[str, Module] = field(default_factory=dict)
    busses: Dict[str, Bus] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    is_leaf: bool = False
    is_stub: bool = True
    draw: ModulePhysical = field(default_factory=ModulePhysical)

    def __post_init__(self):
        self.full_name = f"{self.parent_module.full_name}/{self.name}" if self.parent_module else self.name

    def add_net(
        self,
        net_name: str,
        net_type: NetType = NetType.WIRE,
    ) -> Net:
        net = Net(
            name=net_name,
            module=self,
            net_type=net_type,
        )
        self.nets[net_name] = net
        return net

    def remove_net(self, net_name: str) -> Optional[Net]:
        return self.nets.pop(net_name, None)

    def add_instance(self, inst_name: str, module: Module, module_ref: str) -> Instance:
        instance = Instance(name=inst_name, module=module, module_ref=module_ref, parent_module=self)
        self.instances[inst_name] = instance

        # Automatically create pins for the instance based on the module's ports
        for port in module.ports.values():
            if port.bus:
                for i in range(port.bus.lsb, port.bus.msb + 1):
                    pin_name = f"{port.name}[{i}]"
                    instance.add_pin(pin_name, port.direction)
            else:
                instance.add_pin(port.name, port.direction)

        return instance

    def remove_instance(self, inst_name: str) -> Optional[Instance]:
        instance = self.instances.pop(inst_name, None)
        if instance:
            for pin in instance.pins.values():
                if pin.net:
                    pin.net.remove_pin(pin)
        return instance

    def add_port(self, port_name: str, direction: PinDirection) -> Port:
        port = Port(port_name, direction, self)
        self.ports[port_name] = port
        return port

    def get_all_instances(self, recursive: bool = True) -> Dict[str, Instance]:
        all_instances = self.instances.copy()
        if recursive:
            for child in self.child_modules.values():
                all_instances.update(child.get_all_instances(True))
        return all_instances

    def get_all_nets(self, recursive: bool = True) -> Dict[str, Net]:
        all_nets = self.nets.copy()
        if recursive:
            for child in self.child_modules.values():
                all_nets.update(child.get_all_nets(True))
        return all_nets

    def get_all_pins(self, recursive: bool = True) -> Dict[str, Pin]:
        """Return all pins in this module (and optionally from all submodules)."""
        all_pins = {}

        # Add pins from instances
        for inst in self.instances.values():
            for pin in inst.pins.values():
                all_pins[pin.full_name] = pin

        # Recurse into children if requested
        if recursive:
            for child in self.child_modules.values():
                all_pins.update(child.get_all_pins(True))
        return all_pins


# -----------------------------
# Module
# -----------------------------
@dataclass
class Design:
    modules: Dict[str, Module] = field(default_factory=dict)
    top_module: Optional[Module] = None
    draw: DesignPhysical = field(default_factory=DesignPhysical)
