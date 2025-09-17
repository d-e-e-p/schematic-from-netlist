import argparse
import os

from schematic_from_netlist.graph.create_schematic import CreateSchematic
from schematic_from_netlist.graph.graph_partition import HypergraphPartitioner
from schematic_from_netlist.interfaces.json_graph import ParseJson
from schematic_from_netlist.interfaces.ltspice_writer import LTSpiceWriter
from schematic_from_netlist.interfaces.verilog_parser import VerilogParser


def main():
    """Main entry point for the command-line tool."""
    parser = argparse.ArgumentParser(description="Generate a schematic from a Verilog netlist.")
    parser.add_argument("netlist_file", help="Path to the Verilog netlist file.")
    parser.add_argument("-k", type=int, default=2, help="Number of partitions for the hypergraph.")
    parser.add_argument("--config", default="data/config/hyper.ini", help="Path to the KaHyPar configuration file.")
    parser.add_argument("--output_dir", default="data/ltspice", help="Directory to save the generated LTspice files.")

    args = parser.parse_args()

    if not os.path.exists(args.netlist_file):
        print(f"Error: File not found: {args.netlist_file}")
        return

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    print(f"Processing netlist file: {args.netlist_file}")
    print(f"Using configuration file: {args.config}")

    # --- Core Logic ---
    # 1. Parse Verilog netlist into a database
    verilog_parser = VerilogParser()
    db = verilog_parser.parse_and_store_in_db([args.netlist_file])
    # insert fanout buffers
    db.buffer_multi_fanout_nets()

    # 2. Build hypergraph data for partitioning
    hypergraph_data = db.build_hypergraph_data()

    # 3. Partition the hypergraph
    partitioner = HypergraphPartitioner(hypergraph_data, db.id_by_instname)
    partition = partitioner.run_partitioning(args.k, args.config)

    # 4. Dump the partitioned graph to JSON, DOT, and PNG
    partitioner.dump_graph_to_json(args.k, partition, "data")

    # 5. Parse the generated JSON to extract schematic data
    if partitioner.graph_json_data:
        json_parser = ParseJson(partitioner.graph_json_data)
        geom_db = json_parser.parse()
        
        # 6. Associate geometry with the netlist
        schematic_db = CreateSchematic(geom_db, db)
        schematic_db.generate_schematic_info()
        
        # 7. Patch buffers and write the final schematic
        schematic_db.patch_and_remove_buffers()
        ltspice_writer = LTSpiceWriter(db, schematic_db)
        ltspice_writer.produce_schematic(args.output_dir)

    print("Schematic generation process completed.")


if __name__ == "__main__":
    main()
