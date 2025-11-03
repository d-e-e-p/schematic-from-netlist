import logging as log
from typing import Dict, List, Optional, Tuple

from .netlist_operations import NetlistOperationsMixin
from .netlist_structures import Bus, Design, Instance, Module, Net, NetType, Pin, PinDirection, Port


class NetlistDatabase(NetlistOperationsMixin):
    """Main database class for the hierarchical netlist"""

    def __init__(self, fanout_threshold: int = 150, skip_nets: List[str] = []):
        self.fanout_threshold = fanout_threshold
        self.skip_nets = skip_nets
        # design > module > instance
        self.current_design = "initial"
        self.design = Design(self.current_design)

        self.debug: bool = False
        self.geom_db: Optional[object] = None
        self.schematic_db: Optional[object] = None
        self.stage: Optional[str] = None

        self.inst_by_name: Dict[str, Instance] = {}  # Fast lookup by full name
        self.nets_by_name: Dict[str, Net] = {}  # Fast lookup by full name
        self.pins_by_name: Dict[str, Pin] = {}  # Fast lookup by full name
        self.ports_by_name: Dict[str, Port] = {}  # Fast lookup by full name

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
        self.design.top_module = module
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build fast lookup tables for instances, nets, and pins"""

        # Clear existing tables
        self.inst_by_name.clear()
        self.nets_by_name.clear()
        self.pins_by_name.clear()

        for module in self.design.modules.values():
            # Add all instances
            module.hash2instance.clear()
            module.hash2net.clear()
            module.hash2pin.clear()
            for instance in module.instances.values():
                module.hash2instance[str(hash(instance))] = instance
                self.inst_by_name[instance.full_name] = instance

                # Add all pins of this instance
                for pin in instance.pins.values():
                    module.hash2pin[str(hash(pin))] = pin
                    self.pins_by_name[pin.full_name] = pin

            # Add all nets
            for net in module.nets.values():
                module.hash2net[str(hash(net))] = net
                self.nets_by_name[net.name] = net

            # Add module ports as pins
            for port in module.ports.values():
                self.ports_by_name[port.name] = port

        """
        instances_by_partition = defaultdict(list)
        for _, inst in self.inst_by_name.items():
            instances_by_partition[inst.partition].append(inst)

        self.design.top_module.clusters.clear()
        for part_id, inst_list in instances_by_partition.items():
            self.design.top_module.clusters[part_id] = Cluster(id=part_id, instances=inst_list)
        """

    def elaborate(self):
        """
        Create a shadow flat database module under design called flat_module,
        which has all the instances and nets module instantiation, and connecting ports
        so that every net and instance has a concrete, fully qualified name
        (e.g. top/u_alu/foo ).
        """
        if not self.design.top_module:
            log.error("Top module not set. Cannot elaborate.")
            return

        flat_module_name = f"{self.design.top_module.name}_flat"
        self.design.flat_module = Module(name=flat_module_name)
        flat_module = self.design.flat_module

        # Map from a flat net to its representative net in a merged set
        self.flat_net_representatives: Dict[Net, Net] = {}

        self._elaborate_recursive(self.design.top_module, self.design.top_module.name, flat_module)

        # Post-processing: clean up merged nets
        all_nets = list(flat_module.nets.values())
        for net in all_nets:
            if net in self.flat_net_representatives:
                rep = self._find_net_rep(net)
                if rep != net:
                    # This net has been merged into another.
                    # Its pins should have been moved already during merge.
                    if net.pins:
                        log.warning(f"Net {net.name} was merged but still has pins. Moving them.")
                        for pin in list(net.pins.values()):
                            rep.connect_pin(pin)
                    flat_module.remove_net(net.name)

        del self.flat_net_representatives  # Clean up

    def _elaborate_recursive(self, module: Module, prefix: str, flat_module: Module):
        """Recursively traverse the design hierarchy to flatten it."""

        # Create flat nets for all nets defined in this module
        for net in module.nets.values():
            flat_net_name = f"{prefix}/{net.name}"
            if flat_net_name not in flat_module.nets:
                flat_net = flat_module.add_net(flat_net_name)
                flat_net.hier_module = module
                self.flat_net_representatives[flat_net] = flat_net

        # Process instances in the current module
        for inst in module.instances.values():
            inst_full_name = f"{prefix}/{inst.name}"
            child_module = inst.module

            if not child_module.instances:  # It's a leaf module
                flat_inst = flat_module.add_instance(inst_full_name, child_module, inst.module_ref)
                flat_inst.hier_module = module
                flat_inst.hier_prefix = prefix

                for pin in inst.pins.values():
                    if pin.net:
                        hier_net = pin.net
                        flat_net_name = f"{prefix}/{hier_net.name}"
                        net_to_connect = flat_module.nets[flat_net_name]
                        rep_net = self._find_net_rep(net_to_connect)
                        flat_inst.connect_pin(pin.name, rep_net)
            else:  # It's a hierarchical module
                self._elaborate_recursive(child_module, inst_full_name, flat_module)

                # Stitch nets at the boundary
                for pin in inst.pins.values():
                    if not pin.net:
                        continue

                    net_outside = pin.net
                    flat_net_name_outside = f"{prefix}/{net_outside.name}"

                    port_name = pin.name
                    if port_name not in child_module.ports:
                        log.warning(
                            f"Instance '{inst.name}' of module '{module.name}' has pin '{port_name}' "
                            f"that is not a port on module '{child_module.name}'"
                        )
                        continue

                    # Assuming port implies a net of the same name inside the module
                    if port_name not in child_module.nets:
                        log.warning(f"Module '{child_module.name}' has port '{port_name}' but no internal net of the same name.")
                        continue

                    net_inside = child_module.nets[port_name]
                    flat_net_name_inside = f"{inst_full_name}/{net_inside.name}"

                    if flat_net_name_outside in flat_module.nets and flat_net_name_inside in flat_module.nets:
                        fnet_out = flat_module.nets[flat_net_name_outside]
                        fnet_in = flat_module.nets[flat_net_name_inside]
                        self._merge_flat_nets(fnet_out, fnet_in)
                    else:
                        log.warning(f"Could not find nets for stitching: {flat_net_name_outside} or {flat_net_name_inside}")

    def _find_net_rep(self, net: Net) -> Net:
        """Find the representative of a net in a disjoint set (for merging nets)."""
        if net not in self.flat_net_representatives:
            self.flat_net_representatives[net] = net
            return net

        path = []
        curr = net
        while self.flat_net_representatives[curr] != curr:
            path.append(curr)
            curr = self.flat_net_representatives[curr]

        rep = curr
        for n in path:
            self.flat_net_representatives[n] = rep
        return rep

    def _merge_flat_nets(self, net1: Net, net2: Net):
        """Merge two flat nets using union-find."""
        rep1 = self._find_net_rep(net1)
        rep2 = self._find_net_rep(net2)

        if rep1 == rep2:
            return

        # Merge rep2 into rep1. Heuristic: keep the one with more pins.
        if len(rep1.pins) < len(rep2.pins):
            rep1, rep2 = rep2, rep1  # swap

        for pin in list(rep2.pins.values()):
            rep1.connect_pin(pin)

        self.flat_net_representatives[rep2] = rep1


# Example usage and helper functions
def create_example_netlist(self):
    """Create an example netlist for testing"""
    db = NetlistDatabase()

    # Create top module
    top_module = Module("TOP")
    db.set_top_module(top_module)
    db.design.modules["TOP"] = top_module

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
    # Set up basic logging for standalone script execution
    log.basicConfig(level=log.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
    # Create example database
    db = create_example_netlist()

    # Query examples
    log.info("Design Statistics:")
    stats = db.design.get_design_statistics()
    for key, value in stats.items():
        log.info(f"  {key}: {value}")
