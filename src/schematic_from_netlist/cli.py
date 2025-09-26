import argparse
import os

from schematic_from_netlist.graph.gen_sch_data import GenSchematicData
from schematic_from_netlist.graph.graph_partition import HypergraphPartitioner
from schematic_from_netlist.graph.group_maker import GroupMaker
from schematic_from_netlist.interfaces.graphviz import Graphviz
from schematic_from_netlist.interfaces.ltspice_writer import LTSpiceWriter
from schematic_from_netlist.interfaces.verilog_parser import VerilogParser

# ---------------- Pipeline Stages ---------------- #


def load_netlist(netlist_file: str, debug: bool):
    """Parse Verilog netlist into database."""
    if not os.path.exists(netlist_file):
        raise FileNotFoundError(f"Netlist file not found: {netlist_file}")

    verilog_parser = VerilogParser()
    db = verilog_parser.parse_and_store_in_db([netlist_file])
    db.debug = debug
    db.dump_to_table("0_initial_parse")

    db.buffer_multi_fanout_nets()  # Insert fanout buffers
    db.dump_to_table("1_after_initial_buffering")

    return db


def partition_hypergraph(db, k: int, config_file: str):
    """Partition hypergraph and assign groups."""
    hypergraph_data = db.build_hypergraph_data()
    partitioner = HypergraphPartitioner(hypergraph_data, db)
    partition = partitioner.run_partitioning(k, config_file)
    db.assign_to_groups(partition)


def build_geometry(db):
    """Build Graphviz layouts for groups and top-level interconnect."""
    gv = Graphviz(db)
    db.geom_db = gv.get_layout_geom()


def generate_schematic(db, output_dir: str):
    """Generate schematic info and produce LTSpice output."""
    schematic_db = GenSchematicData(db)
    schematic_db.generate_schematic()

    """
    group_maker = GroupMaker(db, schematic_db)
    group_maker.insert_route_guide_buffers()
    db.dump_to_table("4_after_route_guide_insertion")
    """

    writer = LTSpiceWriter(db, schematic_db)
    writer.produce_schematic(output_dir)


# ---------------- CLI Entrypoint ---------------- #


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a schematic from a Verilog netlist.")
    parser.add_argument("netlist_file", help="Path to the Verilog netlist file.")
    parser.add_argument("-k", type=int, default=2, help="Number of partitions for the hypergraph.")
    parser.add_argument("--config", default="data/config/hyper.ini", help="Path to the KaHyPar configuration file.")
    parser.add_argument("--output_dir", default="data/ltspice", help="Directory to save the generated LTSpice files.")
    # Logging flags
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--debug", action="store_true", help="Enable debug side info.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Processing netlist file: {args.netlist_file}")

    db = load_netlist(args.netlist_file, args.debug)
    partition_hypergraph(db, args.k, args.config)

    db.stage = "init"
    build_geometry(db)
    generate_schematic(db, args.output_dir)

    db.remove_multi_fanout_buffers()
    db.dump_to_table("5_final_state_after_buffer_removal")

    print("Run Complete.")


if __name__ == "__main__":
    main()
