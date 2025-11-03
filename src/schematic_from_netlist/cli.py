import argparse
import logging as log
import os

import colorlog

from schematic_from_netlist.global_placer.global_placer import GlobalPlacer
from schematic_from_netlist.global_router.global_router import GlobalRouter
from schematic_from_netlist.graph.gen_sch_data import GenSchematicData
from schematic_from_netlist.graph.group_maker import SteinerGroupMaker
from schematic_from_netlist.interfaces.graphviz import Graphviz
from schematic_from_netlist.interfaces.ltspice_writer import LTSpiceWriter
from schematic_from_netlist.interfaces.verilog_parser import VerilogParser
from schematic_from_netlist.sastar_router.sastar_router import AstarRouter
from schematic_from_netlist.utils.config import setup_logging

# ---------------- Pipeline Stages ---------------- #


def load_netlist(netlist_file: str, debug: bool):
    """Parse Verilog netlist into database."""
    if not os.path.exists(netlist_file):
        raise FileNotFoundError(f"Netlist file not found: {netlist_file}")

    # vr = VerilogReorder()
    # processed_file, top_module = vr.parse_and_create_stubs(netlist_file)

    vp = VerilogParser()
    db = vp.parse_and_store(netlist_file)
    db.debug = debug
    db.dump_to_table("initial_parse")
    db.determine_design_hierarchy()
    db.elaborate()
    return db


def produce_graph(db):
    """Build Graphviz layouts for groups and top-level interconnect."""

    gv = Graphviz(db)
    gp = GlobalPlacer(db)
    gr = GlobalRouter(db)
    ar = AstarRouter(db)
    # router = AstarRouter(db)

    # graphviz -> extract macro and port locations -> remove buffers
    db.buffer_multi_fanout_nets()  # Insert fanout buffers
    db.dump_to_table("multi_fanout_insertion")
    gv.generate_layout_figures(phase="initial", flat=True)
    db.remove_multi_fanout_buffers()
    db.fig2geom()
    gp.place_design()
    exit()

    junctions = gr.insert_routing_junctions()
    # db.dump_to_table("route_guide_insertion")
    # dr.reroute()

    bypass_phase2 = True
    if bypass_phase2:
        ar.route_design()
        # db.remove_multi_fanout_buffers()
        db.geom2shape()
    else:
        db.insert_route_guide_buffers(junctions)
        db.geom2fig()
        gv.generate_layout_figures(phase="placed")


def generate_schematic(db, output_dir: str):
    """Generate schematic info and produce LTSpice output."""
    db.schematic_db = GenSchematicData(db)
    bypass_phase2 = True
    if not bypass_phase2:
        db.schematic_db.generate_schematic()

    writer = LTSpiceWriter(db)
    writer.produce_schematic(output_dir)


def generate_steiner_buffers(db):
    db.remove_multi_fanout_buffers()
    db.dump_to_table("final_state_after_buffer_removal")

    group_maker = SteinerGroupMaker(db)
    group_maker.insert_route_guide_buffers()
    db.dump_to_table("after_route_guide_insertion")


"""
def optimize_layout(db):
    db.shape2geom()
    optimizer = LayoutOptimizer(db)
    optimizer.optimize_layout()
"""


# ---------------- CLI Entrypoint ---------------- #


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a schematic from a Verilog netlist.")
    parser.add_argument("netlist_file", help="Path to the Verilog netlist file.")
    parser.add_argument("--max_fanout", type=int, default=20, help="fanout threshold for connecting nodes in a module")
    parser.add_argument("--output_dir", default="data/ltspice", help="Directory to save the generated LTSpice files.")
    # Logging flags
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--debug", action="store_true", help="Enable debug side info.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging
    setup_logging(args.debug)

    db = load_netlist(args.netlist_file, args.debug)
    produce_graph(db)
    generate_schematic(db, args.output_dir)

    log.info("Run Complete.")


if __name__ == "__main__":
    main()
