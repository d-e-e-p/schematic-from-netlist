import csv
import json
import os
from pprint import pprint
from typing import Dict, List, Optional

from schematic_from_netlist.graph.graph_partition import Edge, HypergraphData

from .netlist_structures import Instance, Net, Pin, PinDirection


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
            print(f"{name} {net.num_conn=}")
            if net.num_conn > 0 and net.num_conn < self.fanout_threshold:
                net.id = net_id_counter
                self.id_by_netname[name] = net_id_counter
                print(f"net {name} now has {net.id=}")
                self.nets_by_id[net_id_counter] = net
                self.netname_by_id[net_id_counter] = name
                net_id_counter += 1
                for pin in net.pins:
                    instname = pin.instance.name
                    if instname in self.id_by_instname:
                        print(f"looking at {pin.full_name} with instname {instname=} already has id = {self.id_by_instname[instname]=}")
                    if instname not in self.id_by_instname:
                        self.id_by_instname[instname] = inst_id_counter
                        self.instname_by_id[inst_id_counter] = instname
                        self.inst_by_id[inst_id_counter] = pin.instance
                        pin.instance.id = inst_id_counter
                        print(f"created id of {pin.full_name} with instname {instname=} has id = {self.id_by_instname[instname]=}")
                        inst_id_counter += 1

        # ok check we got everyone
        for name, inst in self.inst_by_name.items():
            if inst.id == -1:
                print(f"Inst {name} not assigned an ID")
                self.id_by_instname[inst.name] = inst_id_counter
                self.instname_by_id[inst_id_counter] = inst.name
                self.inst_by_id[inst_id_counter] = inst
                inst.id = inst_id_counter
                inst_id_counter += 1
        for name, net in self.nets_by_name.items():
            if net.id == -1:
                print(f"Net {name} not assigned an ID")

        pprint(self.id_by_instname)

    def buffer_multi_fanout_nets(self):
        """Inserts buffers on nets with fanout > 1"""
        print("--- Before buffering ---")
        pprint(self.get_design_statistics())

        if not self.top_module:
            return

        nets_to_buffer = [net for net in self.nets_by_name.values() if net.num_conn > 2 and net.num_conn < self.fanout_threshold]
        print(f"instrumentation: Found {len(nets_to_buffer)} nets to buffer.")

        for net in nets_to_buffer:
            original_net_name = net.name
            print(f"instrumentation: Buffering net {original_net_name} with {net.num_conn} connections.")
            self.buffered_nets_log[original_net_name] = {"old_pins": set(), "buffer_insts": [], "new_nets": []}

            log = self.buffered_nets_log[original_net_name]
            log["old_pins"] = net.pins.copy()

            buffer_name = f"{self.inserted_buf_prefix}{original_net_name}"
            buffer_inst = self.top_module.add_instance(buffer_name, "FANOUT_BUFFER")
            log["buffer_insts"].append(buffer_inst)

            pins_to_buffer = list(net.pins)
            for i, pin in enumerate(pins_to_buffer):
                new_net_name = f"{original_net_name}{self.inserted_net_suffix}{i}"
                new_net = self.top_module.add_net(new_net_name)

                buf_inout_pin = buffer_inst.add_pin(f"IO{i}", PinDirection.INOUT)
                new_net.add_pin(buf_inout_pin)

                print(f"instrumentation: Moving pin {pin.full_name} from {net.name} to {new_net.name}")
                net.remove_pin(pin)
                new_net.add_pin(pin)
                log["new_nets"].append(new_net)

        self._build_lookup_tables()
        print("--- After buffering ---")
        pprint(self.get_design_statistics())

    def remove_multi_fanout_buffers(self):
        """Removes all buffers and restores original connectivity."""
        if not self.top_module:
            return

        print("--- Before removing buffers ---")
        pprint(self.get_design_statistics())

        # Restore original nets from the log
        for original_net_name, log in self.buffered_nets_log.items():
            original_net = self.find_net(original_net_name)
            if not original_net:
                print(f"instrumentation: Could not find original net {original_net_name} during restore.")
                continue

            print(f"instrumentation: Restoring pins for {original_net_name}")
            for pin in log.get("old_pins", []):
                if pin.net != original_net:
                    original_net.add_pin(pin)

        # Proactively find and delete all buffer instances and nets
        instances_to_delete = [inst_name for inst_name in self.top_module.instances if inst_name.startswith(self.inserted_buf_prefix)]
        nets_to_delete = [net_name for net_name in self.top_module.nets if self.inserted_net_suffix in net_name or net_name.startswith(f"top_{self.inserted_buf_prefix}")]

        for inst_name in instances_to_delete:
            instance = self.top_module.instances.get(inst_name)
            if instance:
                print(f"instrumentation: Deleting buffer instance {instance.name}")
                for pin in instance.pins.values():
                    if pin.net:
                        pin.net.remove_pin(pin)
                del self.top_module.instances[inst_name]

        for net_name in nets_to_delete:
            if net_name in self.top_module.nets:
                print(f"instrumentation: Deleting buffer net {net_name}")
                del self.top_module.nets[net_name]

        self.buffered_nets_log.clear()
        self._build_lookup_tables()

        print("--- After removing buffers ---")
        pprint(self.get_design_statistics())

    def create_buffering_for_groups(self, net, ordering, clusters):
        """deal with fanout routing"""
        original_net_name = net.name
        table_output_dir = "data/tables"
        self.dump_netlist_db_to_table(table_output_dir, f"pre_buffering_{original_net_name}", -1)

        buffer_insts_map = {}

        for i, cluster in enumerate(clusters):
            if original_net_name not in self.buffered_nets_log:
                self.buffered_nets_log[original_net_name] = {"old_pins": set(), "buffer_insts": [], "new_nets": []}

            log = self.buffered_nets_log[original_net_name]
            if not log["old_pins"]:
                log["old_pins"] = net.pins.copy()

            buffer_name = f"{self.inserted_buf_prefix}{i}_{original_net_name}"
            buffer_inst = self.top_module.add_instance(buffer_name, "FANOUT_BUFFER")
            buffer_insts_map[i] = buffer_inst
            log["buffer_insts"].append(buffer_inst)
            print(f" connecting {buffer_name=} to cluster {cluster=}")

            pins_to_buffer = [self.find_pin(pinname) for pinname in cluster]
            for j, pin in enumerate(pins_to_buffer):
                if pin is None:
                    print(f"instrumentation: WARNING - Could not find pin for name {cluster[j]}")
                    continue
                new_net_name = f"{original_net_name}{self.inserted_net_suffix}{i}_{j}"
                new_net = self.top_module.add_net(new_net_name)

                buf_inout_pin = buffer_inst.add_pin(f"IO{j}", PinDirection.INOUT)
                new_net.add_pin(buf_inout_pin)

                net.remove_pin(pin)
                new_net.add_pin(pin)
                log["new_nets"].append(new_net)

        if len(ordering) > 1:
            for k in range(len(ordering) - 1):
                src_buf_idx, dst_buf_idx = ordering[k], ordering[k + 1]
                src_buffer_inst, dst_buffer_inst = buffer_insts_map.get(src_buf_idx), buffer_insts_map.get(dst_buf_idx)

                if src_buffer_inst and dst_buffer_inst:
                    chain_net_name = f"top_{self.inserted_buf_prefix}{src_buf_idx}_{dst_buf_idx}_{original_net_name}"
                    chain_net = self.top_module.add_net(chain_net_name)

                    src_pin_num, dst_pin_num = len(src_buffer_inst.pins), len(dst_buffer_inst.pins)
                    src_pin = src_buffer_inst.add_pin(f"IO{src_pin_num}", PinDirection.INOUT)
                    dst_pin = dst_buffer_inst.add_pin(f"IO{dst_pin_num}", PinDirection.INOUT)

                    chain_net.add_pin(src_pin)
                    chain_net.add_pin(dst_pin)
                    print(f"instrumentation: Chaining buffer {src_buffer_inst.name} to {dst_buffer_inst.name} with net {chain_net.name}")

        self._build_lookup_tables()
        self.dump_netlist_db_to_table(table_output_dir, f"post_buffering_{original_net_name}", -1)

    def dump_netlist_db_to_table(self, output_dir: str, stage_name: str, iteration: int = -1):
        """Dumps the netlist database to a CSV file."""
        os.makedirs(output_dir, exist_ok=True)

        base_filename = f"{stage_name}_iteration_{iteration}" if iteration >= 0 else stage_name
        filename = os.path.join(output_dir, f"{base_filename}.csv")
        log_filename = os.path.join(output_dir, f"{base_filename}_buffer_log.json")

        print(f"instrumentation: Dumping netlist DB to {filename}")

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["net.name", "net.id", "net.num_conn", "pin.full_name", "pin.instance.name", "pin.instance.id"])
            sorted_nets = sorted(self.nets_by_name.values(), key=lambda n: n.name)
            for net in sorted_nets:
                if not net.pins:
                    writer.writerow([net.name, net.id, net.num_conn, "", "", ""])
                else:
                    sorted_pins = sorted(list(net.pins), key=lambda p: p.full_name)
                    for pin in sorted_pins:
                        writer.writerow([net.name, net.id, net.num_conn, pin.full_name, pin.instance.name, pin.instance.id])

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

    def get_design_statistics(self) -> "Dict":
        """Get overall design statistics"""
        stats = {"total_instances": len(self.inst_by_name), "total_nets": len(self.nets_by_name), "total_pins": len(self.pins_by_name), "modules": len(self.modules), "floating_nets": 0, "multi_driven_nets": 0, "max_fanout": 0, "avg_fanout": 0}
        total_fanout = sum(net.get_fanout() for net in self.nets_by_name.values())

        for net in self.nets_by_name.values():
            stats["max_fanout"] = max(stats["max_fanout"], net.get_fanout())
            if net.is_floating():
                stats["floating_nets"] += 1
            if net.has_multiple_drivers():
                stats["multi_driven_nets"] += 1

        if self.nets_by_name:
            stats["avg_fanout"] = total_fanout // len(self.nets_by_name)
        return stats

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
            print(f"connecting {net.name=} conn {net.num_conn}: {connected_instance_ids=} {[f'{pin.instance.name}/{pin.name}' for pin in net.pins]=}")

        return HypergraphData(num_nodes, num_edges, index_vector, edge_vector)

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
                            edge = Edge(src.name, dst.name, name=None if net.name.startswith(self.inserted_buf_prefix) else net.name)
                            edges.append(edge)
        return edges
