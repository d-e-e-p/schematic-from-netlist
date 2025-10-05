import csv
import json
import logging
import os
import re
from collections import defaultdict

# from pprint import pprint
from typing import Dict, List, Optional

from tabulate import tabulate

from schematic_from_netlist.graph.graph_partition import Edge, HypergraphData

from .netlist_structures import Design, Instance, Module, Net, Pin, PinDirection


class NetlistOperationsMixin:
    """Mixin class for netlist operations"""

    # Query Methods
    def find_net(self, net_name: str) -> "Optional[Net]":
        """Find a net by full name"""
        return self.nets_by_name.get(net_name)

    def find_instance(self, instance_name: str) -> "Optional[Instance]":
        """Find an instance by full name"""
        return self.inst_by_name.get(instance_name)

    def find_pin(self, pin_name: str) -> "Optional[Pin]":
        """Find a pin by full name"""
        return self.pins_by_name.get(pin_name)

    def get_net_connections(self, net: "Net") -> "Dict[str, List[Pin]]":
        """Get all connections to a net, organized by type"""
        return {
            "drivers": list(net.drivers),
            "loads": list(net.loads),
            "all_pins": list(net.pins),
        }

    def trace_net_path(self, start_pin: "Pin", max_depth: int = 10) -> "List[Pin]":
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

    def get_fanout_tree(self, driver_pin: "Pin") -> "Dict":
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

    def find_timing_paths(self, start_instance: "Instance", end_instance: "Instance") -> "List[List[Pin]]":
        """Find timing paths between two instances"""
        # This is a simplified version - real timing analysis is much more complex
        paths = []

        def dfs_path(current_pin: "Pin", target_instance: "Instance", current_path: "List[Pin]"):
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

        for name, net in self.nets_by_name.items():
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

        # ok check we got everyone
        for name, inst in self.inst_by_name.items():
            if inst.id == -1:
                logging.warning(f"Inst {name} not assigned an ID")
                self.id_by_instname[inst.name] = inst_id_counter
                self.instname_by_id[inst_id_counter] = inst.name
                self.inst_by_id[inst_id_counter] = inst
                inst.id = inst_id_counter
                inst_id_counter += 1
        for name, net in self.nets_by_name.items():
            if net.id == -1:
                logging.warning(f"Net {name} not assigned an ID")

        # pprint(self.id_by_instname)

    def buffer_multi_fanout_nets(self):
        """Inserts buffers on nets with fanout > 1"""
        self.print_design_stats("before buffering")
        sorted_modules = sorted(self.design.modules.values(), key=lambda m: m.depth, reverse=True)
        for module in sorted_modules:
            if not module.is_leaf:
                self.buffer_multi_fanout_nets_for_module(module)

        self._build_lookup_tables()
        self.print_design_stats("before buffering")

    def buffer_multi_fanout_nets_for_module(self, module):
        """Inserts buffers on nets with fanout > 1"""

        nets_to_buffer = [net for net in module.nets.values() if net.num_conn > 2 and net.num_conn < self.fanout_threshold]
        if len(nets_to_buffer) == 0:
            return

        module_ref = "FANOUT_BUFFER"
        stub_module = Module(name=module_ref)
        stub_module.is_leaf = True

        logging.info(f"instrumentation: in {module.name} Found {len(nets_to_buffer)} nets to buffer.")
        for net in nets_to_buffer:
            original_net_name = net.name
            net.draw.shape = []
            logging.debug(f"instrumentation: in {module.name} Buffering net {original_net_name} with {net.num_conn} connections.")
            i = 0
            buffer_name = f"{self.inserted_buf_prefix}{i}_{original_net_name}"
            buffer_inst = module.add_instance(buffer_name, stub_module, module_ref)
            buffer_inst.is_buffer = True
            buffer_inst.buffer_original_netname = original_net_name

            pins_to_buffer = list(net.pins.values())
            for i, pin in enumerate(pins_to_buffer):
                new_net_name = f"{original_net_name}{self.inserted_net_suffix}{i}"
                new_net = module.add_net(new_net_name)
                new_net.is_buffered_net = True
                new_net.buffer_original_netname = original_net_name
                logging.debug(f"instrumentation: created net {new_net.name} derived from {new_net.buffer_original_netname}")

                buf_inout_pin = buffer_inst.add_pin(f"IO{i}", PinDirection.INOUT)
                new_net.connect_pin(buf_inout_pin)

                logging.debug(f"instrumentation: Moving pin {pin.full_name} from {net.name} to {new_net.name}")
                net.remove_pin(pin)
                new_net.connect_pin(pin)

    def remove_multi_fanout_buffers(self):
        """Removes all buffers and restores original connectivity."""
        if not self.design.top_module:
            return

        self.print_design_stats("before removing buffers")

        # Restore original nets from the log
        for original_net_name, log in self.buffered_nets_log.items():
            original_net = self.find_net(original_net_name)
            if not original_net:
                logging.warning(f"instrumentation: Could not find original net {original_net_name} during restore.")
                continue

            logging.debug(f"instrumentation: Restoring pins for {original_net_name}")
            for pin in log.get("old_pins", []):
                if pin.net != original_net:
                    original_net.add_pin(pin)

        # Proactively find and delete all buffer instances and nets
        instances_to_delete = [
            inst_name for inst_name in self.design.top_module.instances if inst_name.startswith(self.inserted_buf_prefix)
        ]
        nets_to_delete = [
            net_name
            for net_name in self.design.top_module.nets
            if self.inserted_net_suffix in net_name or net_name.startswith(f"top_{self.inserted_buf_prefix}")
        ]

        for inst_name in instances_to_delete:
            instance = self.design.top_module.instances.get(inst_name)
            if instance:
                logging.debug(f"instrumentation: Deleting buffer instance {instance.name}")
                for pin in instance.pins.values():
                    if pin.net:
                        pin.net.remove_pin(pin)
                del self.design.top_module.instances[inst_name]

        for net_name in nets_to_delete:
            if net_name in self.design.top_module.nets:
                logging.debug(f"instrumentation: Deleting buffer net {net_name}")
                # clone wires before delete
                net = self.design.top_module.nets[net_name]
                logging.debug(f"looking for original of {net_name=} =  {net.buffer_original_netname}")
                original_net = self.nets_by_name.get(net.buffer_original_netname)
                original_net.draw.shape.extend(net.draw.shape)
                del self.design.top_module.nets[net_name]

        self.buffered_nets_log.clear()
        self._build_lookup_tables()

        self.print_design_stats("after removing buffers")

    def create_buffering_for_groups(self, net, ordering, collections, cluster_id):
        """deal with fanout routing"""
        original_net_name = net.name
        table_output_dir = "data/tables"
        self.dump_to_table(f"pre_buffering_{original_net_name}")

        buffer_insts_map = {}

        for i, collection in enumerate(collections):
            if original_net_name not in self.buffered_nets_log:
                self.buffered_nets_log[original_net_name] = {"old_pins": set(), "buffer_insts": [], "new_nets": []}

            log = self.buffered_nets_log[original_net_name]
            if not log["old_pins"]:
                log["old_pins"] = net.pins.copy()

            buffer_name = f"{self.inserted_buf_prefix}{i}{cluster_id}_{original_net_name}"
            buffer_inst = self.design.top_module.add_instance(buffer_name, "FANOUT_BUFFER")
            buffer_inst.partition = cluster_id
            buffer_inst.is_buffer = True
            buffer_insts_map[i] = buffer_inst
            log["buffer_insts"].append(buffer_inst)
            logging.debug(f" connecting {buffer_name=} to collection {collection=}")

            pins_to_buffer = [self.find_pin(pinname) for pinname in collection]
            for j, pin in enumerate(pins_to_buffer):
                if pin is None:
                    logging.warning(f"instrumentation: WARNING - Could not find pin for name {collection[j]}")
                    continue
                new_net_name = f"{original_net_name}{self.inserted_net_suffix}{i}_{j}_{cluster_id}"
                new_net = self.design.top_module.add_net(new_net_name)
                new_net.is_buffered_net = True
                new_net.buffer_original_netname = original_net_name

                buf_inout_pin = buffer_inst.add_pin(f"IO{j}", PinDirection.INOUT)
                new_net.add_pin(buf_inout_pin)
                logging.debug(f" buffer pin {buf_inout_pin.full_name} in {cluster_id=} now drives {collection=}")

                net.remove_pin(pin)
                new_net.add_pin(pin)
                log["new_nets"].append(new_net)

        if len(ordering) > 1:
            for k in range(len(ordering) - 1):
                src_buf_idx, dst_buf_idx = ordering[k], ordering[k + 1]
                src_buffer_inst, dst_buffer_inst = buffer_insts_map.get(src_buf_idx), buffer_insts_map.get(dst_buf_idx)
                if src_buffer_inst and dst_buffer_inst:
                    chain_net_name = f"top_{self.inserted_buf_prefix}{src_buf_idx}_{dst_buf_idx}_{cluster_id}_{original_net_name}"
                    chain_net = self.design.top_module.add_net(chain_net_name)
                    chain_net.is_buffered_net = True
                    chain_net.buffer_original_netname = original_net_name

                    src_pin_num, dst_pin_num = len(src_buffer_inst.pins), len(dst_buffer_inst.pins)
                    src_pin = src_buffer_inst.add_pin(f"IO{src_pin_num}", PinDirection.INOUT)
                    dst_pin = dst_buffer_inst.add_pin(f"IO{dst_pin_num}", PinDirection.INOUT)

                    chain_net.add_pin(src_pin)
                    chain_net.add_pin(dst_pin)
                    logging.debug(
                        f"instrumentation: Chaining buffer {src_buffer_inst.name} to {dst_buffer_inst.name} with net {chain_net.name}"
                    )

        self._build_lookup_tables()
        # self.dump_to_table(table_output_dir, f"post_buffering_{original_net_name}", -1)

    def dump_to_table(self, stage_name: str):
        """Dumps the netlist database to a CSV file."""
        # TODO: make this only operate in debug mode
        current_value = getattr(self, "_table_counter", 0)
        self._table_counter = current_value + 1

        stage_name = f"{self._table_counter}_{stage_name}"
        output_dir = os.path.join("data", "tables")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"{stage_name}.csv")
        log_filename = os.path.join(output_dir, f"{stage_name}_buffer_log.json")

        logging.debug(f"instrumentation: Dumping netlist DB to {filename}")

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["net.module.name", "net.name", "net.id", "net.num_conn", "pin.full_name", "pin.instance.name", "pin.instance.id"]
            )
            # Sort modules by name for consistent output
            sorted_modules = sorted(self.design.modules.values(), key=lambda m: m.name)

            for module in sorted_modules:
                # Optional: write a header for the module
                # writer.writerow([f"Module: {module.name}", "", "", "", "", "", ""])

                sorted_nets = sorted(module.nets.values(), key=lambda n: n.name)
                for net in sorted_nets:
                    if not net.pins:
                        writer.writerow([module.name, net.name, net.id, net.num_conn, "", "", ""])
                    else:
                        sorted_pins = sorted(list(net.pins.values()), key=lambda p: p.full_name)
                        for pin in sorted_pins:
                            writer.writerow(
                                [module.name, net.name, net.id, net.num_conn, pin.full_name, pin.instance.name, pin.instance.id]
                            )

        dump_log = {
            net_name: {
                "old_pins": [p.full_name for p in log_data.get("old_pins", [])],
                "buffer_insts": [i.name for i in log_data.get("buffer_insts", [])],
                "new_nets": [n.name for n in log_data.get("new_nets", [])],
            }
            for net_name, log_data in self.buffered_nets_log.items()
        }
        with open(log_filename, "w") as f:
            json.dump(dump_log, f, indent=2)

    def count_with_children(self, module):
        """Recursively count module + all descendants, return aggregated totals."""
        # Start with this module's own counts
        totals = {
            "ports": len(module.ports.values()),
            "instances": len(module.instances.values()),
            "pins": len(module.pins.values()),
            "nets": len(module.nets.values()),
        }

        # Add counts from all children
        for child in module.child_modules.values():
            child_totals = self.count_with_children(child)
            totals["ports"] += child_totals["ports"]
            totals["instances"] += child_totals["instances"]
            totals["pins"] += child_totals["pins"]
            totals["nets"] += child_totals["nets"]

        return totals

    def print_design_stats(self, title: str = ""):
        """Prints detailed design statistics in a hierarchical table."""

        if not self.design.top_module:
            logging.error("No top module set.")
            return

        headers = ["Hierarchy", "Module", "Instances", "Ports", "Pins", "Nets", "Max Fanout", "Buffers"]
        table_data = []
        memo_stats = {}

        def get_stats(module):
            if module.name in memo_stats:
                return memo_stats[module.name]

            # Direct stats from this module
            totals = {
                "ports": len(module.ports),
                "instances": len(module.instances),
                "pins": len(module.pins),
                "nets": len(module.nets),
                "max_fanout": max([net.num_conn for net in module.nets.values()], default=0),
                "buffers": sum(1 for inst in module.instances.values() if inst.is_buffer),
            }

            # Recursively add stats from child modules
            for instance in module.instances.values():
                child_module = self.design.modules.get(instance.module_ref)
                if child_module:
                    child_totals = get_stats(child_module)
                    totals["instances"] += child_totals["instances"]
                    totals["pins"] += child_totals["pins"]
                    totals["nets"] += child_totals["nets"]
                    totals["ports"] += child_totals["ports"]

            memo_stats[module.name] = totals
            return totals

        def build_table_recursive(module, prefix="", is_top=True):
            stats = get_stats(module)

            if is_top:
                # Top module entry
                table_data.append(
                    [
                        module.name,
                        module.name,
                        stats["instances"],
                        stats["ports"],
                        stats["pins"],
                        stats["nets"],
                        stats["max_fanout"],
                        stats["buffers"],
                    ]
                )

            # Sort child instances by their total instance count
            sorted_instances = sorted(
                module.instances.values(),
                key=lambda inst: get_stats(self.design.modules.get(inst.module_ref, Module("dummy"))).get("instances", 0),
                reverse=True,
            )

            for i, instance in enumerate(sorted_instances):
                child_module = self.design.modules.get(instance.module_ref)
                if not child_module or child_module.is_leaf:
                    continue

                is_last = i == len(sorted_instances) - 1
                connector = "└── " if is_last else "├── "

                child_stats = get_stats(child_module)
                table_data.append(
                    [
                        f"{prefix}{connector}{instance.name}",
                        child_module.name,
                        child_stats["instances"],
                        child_stats["ports"],
                        child_stats["pins"],
                        child_stats["nets"],
                        child_stats["max_fanout"],
                        child_stats["buffers"],
                    ]
                )

                new_prefix = prefix + ("    " if is_last else "│   ")
                build_table_recursive(child_module, new_prefix, is_top=False)

        build_table_recursive(self.design.top_module)
        logging.info(title + "\n" + tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    def build_hypergraph_data(self) -> "HypergraphData":
        """Builds the hypergraph data structure for KaHyPar."""
        self.generate_ids()
        num_nodes, num_edges = len(self.inst_by_id), len(self.nets_by_id)
        sorted_nets = sorted(self.nets_by_id.values(), key=lambda net: net.id)
        edge_vector, index_vector = [], [0]

        for net in sorted_nets:
            connected_instance_ids = sorted(list({pin.instance.id for pin in net.pins}))
            edge_vector.extend(connected_instance_ids)
            index_vector.append(len(edge_vector))
            logging.debug(
                f"hyper: {net.name=} conn {net.num_conn}: {connected_instance_ids=} {[f'{pin.instance.name}/{pin.name}' for pin in net.pins]=}"
            )

        return HypergraphData(num_nodes, num_edges, index_vector, edge_vector)

    def assign_to_groups(self, partitions):
        instances_by_partition = defaultdict(list)
        for inst_id, part_id in partitions.items():
            if inst := self.inst_by_id.get(inst_id):
                inst.partition = part_id
                instances_by_partition[part_id].append(inst)
            else:
                logging.warning(f"warning: instance not found matching {id=}")

        self.design.top_module.clusters.clear()
        for part_id, inst_list in instances_by_partition.items():
            self.design.top_module.clusters[part_id] = Cluster(id=part_id, instances=inst_list)

    def get_edges_between_nodes(self, nodes):
        inst_list = [self.inst_by_id[id] for id in nodes]
        net_list = list({net for inst in inst_list for net in inst.get_connected_nets()})
        edges = []
        for net in net_list:
            if 2 <= net.num_conn <= self.fanout_threshold:
                conn_inst_list = [pin.instance for pin in net.pins]
                for i in range(len(conn_inst_list)):
                    for j in range(i + 1, len(conn_inst_list)):
                        src, dst = conn_inst_list[i], conn_inst_list[j]
                        if src in inst_list and dst in inst_list:
                            edge = Edge(
                                src.name, dst.name, name=None if net.name.startswith(self.inserted_buf_prefix) else net.name
                            )
                            edges.append(edge)
        return edges

    def clear_all_shapes(self):
        for inst in self.design.top_module.get_all_instances().values():
            inst.draw.shape = None

        for net in self.design.top_module.get_all_nets().values():
            net.draw.shape.clear()
            net.draw.buffer_patch_points.clear()

        for pin in self.design.top_module.get_all_pins().values():
            pin.draw.shape = None

    def geom2shape(self):
        """Convert all geom objects to shape."""
        self.clear_all_shapes()
        for collection in (
            self.design.top_module.get_all_instances().values(),
            self.design.top_module.get_all_nets().values(),
            self.design.top_module.get_all_pins().values(),
        ):
            for obj in collection:
                obj.geom2shape()

    def shape2geom(self):
        """Convert all shape objects to geom."""
        for collection in (
            self.design.top_module.get_all_instances().values(),
            self.design.top_module.get_all_nets().values(),
            self.design.top_module.get_all_pins().values(),
        ):
            for obj in collection:
                obj.shape2geom()

    def uniquify_module_names(self):
        """needed to have each ref having different shape"""
        seen = {}
        for inst in self.design.top_module.get_all_instances().values():
            if seen.get(inst.module_ref):
                counter = seen[inst.module_ref]
            else:
                counter = 0

            name = f"{inst.module_ref}{counter}"
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            inst.module_ref_uniq = safe_name
            seen[inst.module_ref] = counter + 1

    def determine_design_hierarchy(self) -> List[Module]:
        """
        Determines the design hierarchy, prints it, and returns a list of modules
        sorted from bottom to top.
        """
        if not self.design.top_module:
            logging.error("No top module set.")
            return []

        # Build parent-child relationships
        for module in self.design.modules.values():
            for instance in module.instances.values():
                child_module_name = instance.module_ref
                if child_module_name in self.design.modules:
                    child_module = self.design.modules[child_module_name]
                    child_module.parent_module = module
                    module.child_modules[child_module.name] = child_module
                    logging.debug(f"  {module.name} -> {child_module.name}")

        # Helper to print the hierarchy tree
        def print_hierarchy(module, prefix=""):
            print(f"{prefix}- {module.name}")
            for child in module.child_modules.values():
                print_hierarchy(child, prefix + "  ")

        print("Design Hierarchy:")
        print_hierarchy(self.design.top_module)

        # Sort modules by depth (bottom-up)
        depth_map = {}

        def get_depth(module):
            if module.name in depth_map:
                return depth_map[module.name]
            if not module.parent_module:
                depth = 0
            else:
                depth = get_depth(module.parent_module) + 1
            depth_map[module.name] = depth
            return depth

        for module in self.design.modules.values():
            module.depth = get_depth(module)

        sorted_modules = sorted(self.design.modules.values(), key=lambda m: depth_map[m.name], reverse=True)

        # Find leaf modules by checking instances
        leaf_modules = []
        for module in self.design.modules.values():
            has_child_modules = any(instance.module_ref in self.design.modules for instance in module.instances.values())
            if not has_child_modules:
                leaf_modules.append(module)
                module.is_leaf = True
        print(f"\nLeaf modules: {[module.name for module in leaf_modules]}")

        return sorted_modules
