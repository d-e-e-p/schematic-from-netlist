from typing import Dict, List, Optional, Tuple

from .netlist_operations import NetlistOperationsMixin
from .netlist_structures import Instance, Module, Net, NetType, Pin, PinDirection


class NetlistDatabase(NetlistOperationsMixin):
    """Main database class for the hierarchical netlist"""

    def __init__(self, fanout_threshold: int = 55):
        self.fanout_threshold = fanout_threshold

        self.debug: bool = False
        self.geom_db: Optional[object] = None
        self.schematic_db: Optional[object] = None
        self.groups: list[tuple[int, list[Instance]]] = []
        self.stage: Optional[str] = None

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
        self.inserted_net_suffix = "_fanout_buffer_"

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
