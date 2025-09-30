from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from shapely.geometry import LineString, MultiLineString, Point, Polygon, box


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
# Port
# -----------------------------
@dataclass
class Port:
    """Represents ports on modules"""

    name: str
    direction: PinDirection
    module: Module
    bit_width: int = 1
    bit_range: Optional[Tuple[int, int]] = None  # (msb, lsb)

    fig: Optional[Tuple[float, float]] = None
    shape: Optional[Tuple[int, int]] = None
    geom: Optional[Point] = None


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
    bit_width: int = 1
    bit_range: Optional[Tuple[int, int]] = None  # (msb, lsb)

    fig: Optional[Tuple[float, float]] = None
    shape: Optional[Tuple[int, int]] = None
    geom: Optional[Point] = None

    def shape2geom(self):
        if self.shape is None:
            self.geom = None
            return None
        x, y = self.shape
        self.geom = Point(round(x), round(y))
        return self.geom

    def geom2shape(self):
        if self.geom is None:
            self.shape = None
            return None
        self.shape = (int(self.geom.x), int(self.geom.y))
        return self.shape

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
    bit_width: int = 1
    bit_range: Optional[Tuple[int, int]] = None
    pins: Set[Pin] = field(default_factory=set)
    id: int = -1
    num_conn: int = 0
    is_buffer_wire: bool = False
    buffer_original_netname: Optional[str] = None
    buffer_patch_points: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)

    fig: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    shape: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    geom: Optional[MultiLineString] = None

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

    def shape2geom(self):
        if not self.shape:
            self.geom = None
            return None
        # Base geometry
        lines = [LineString(seg) for seg in self.shape]
        # Optionally add patch lines
        lines.extend(LineString(seg) for seg in self.buffer_patch_points)

        # Merge into one MultiLineString
        self.geom = MultiLineString(lines)
        return self.geom

    def geom2shape(self):
        if self.geom is None:
            self.shape.clear()
            return None
        # Convert MultiLineString back to list of 2-point segments
        self.shape = [
            ((int(line.coords[0][0]), int(line.coords[0][1])), (int(line.coords[1][0]), int(line.coords[1][1])))
            for line in self.geom.geoms
        ]
        return self.shape

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
    id: int = -1
    partition: int = -1
    orient: str = "R0"
    is_buffer: bool = False
    buffer_original_netname: Optional[str] = None
    module_ref_uniq: Optional[str] = None

    fig: Optional[Tuple[float, float, float, float]] = None
    shape: Optional[Tuple[int, int, int, int]] = None
    geom: Optional[Polygon] = None

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

    def shape2geom(self):
        if not self.shape or len(self.shape) != 4:
            self.geom = None
            return None
        x1, y1, x2, y2 = self.shape
        self.geom = box(x1, y1, x2, y2)
        return self.geom

    def geom2shape(self):
        if self.geom is None:
            self.shape = None
            return None
        x_min, y_min, x_max, y_max = self.geom.bounds
        self.shape = (int(x_min), int(y_min), int(x_max), int(y_max))
        return self.shape

    def __hash__(self):
        return hash(self.full_name)


# -----------------------------
# Cluster
# -----------------------------
@dataclass
class Cluster:
    id: int
    instances: List[Instance] = field(default_factory=list)
    size: Optional[Tuple[int, int]] = None
    offset: Optional[Tuple[int, int]] = None
    pins: Dict[str, Pin] = field(default_factory=dict)
    shape: Optional[Tuple[int, int, int, int]] = None
    size_float: Optional[Tuple[float, float]] = None
    offset_float: Optional[Tuple[float, float]] = None

    def add_pin(self, pin: Pin):
        self.pins[pin.full_name] = pin


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
    child_modules: Dict[str, Module] = field(default_factory=dict)
    depth: int = 0
    is_leaf: bool = False

    fig: Optional[Tuple[float, float, float, float]] = None
    shape: Optional[Tuple[int, int, int, int]] = None
    geom: Optional[Polygon] = None

    def __post_init__(self):
        self.full_name = f"{self.parent_module.full_name}/{self.name}" if self.parent_module else self.name

    def add_net(
        self,
        net_name: str,
        net_type: NetType = NetType.WIRE,
        bit_width: int = 1,
        bit_range: Optional[Tuple[int, int]] = None,
    ) -> Net:
        net = Net(name=net_name, module=self, net_type=net_type, bit_width=bit_width, bit_range=bit_range)
        self.nets[net_name] = net
        return net

    def remove_net(self, net_name: str) -> Optional[Net]:
        return self.nets.pop(net_name, None)

    def add_instance(self, inst_name: str, module: Module, module_ref: str) -> Instance:
        instance = Instance(name=inst_name, module=module, module_ref=module_ref, parent_module=self)
        self.instances[inst_name] = instance
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
