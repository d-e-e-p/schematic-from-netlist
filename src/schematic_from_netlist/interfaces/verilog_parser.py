import logging as log
import os
import re
import sys

import pyverilog
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer

from schematic_from_netlist.interfaces.netlist_database import (
    Bus,
    Instance,
    Module,
    Net,
    NetlistDatabase,
    NetType,
    Pin,
    PinDirection,
)


class VerilogParser:
    def __init__(self):
        self.db = NetlistDatabase()

    def _clean_name(self, name):
        """Strip backslashes and whitespace from names."""
        if isinstance(name, str):
            return name.replace("\\", "").strip()
        return name

    def parse_and_store_in_db(self, filename, topmodule):
        analyzer = VerilogDataflowAnalyzer(
            [filename],
            topmodule,
            noreorder=False,
            nobind=False,
        )
        analyzer.generate()

        self.store_all_instances(analyzer.getInstances())

    def store_all_instances(self, vinstances):
        for vchain, master in vinstances:
            print(f"{master=} {vchain.get_module_list()=} {vchain.tocode()=} {vchain.scopechain}")
        breakpoint()


def main():
    INFO = "Verilog code parser"
    VERSION = pyslang.__version__
    USAGE = "Usage: python verilog_parser.py file ..."
    log.basicConfig(level=log.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if len(sys.argv) != 2:
        log.error("Please provide a single Verilog file.")
        log.info(USAGE)
        sys.exit(1)

    filename = sys.argv[1]

    parser = VerilogParser()
    try:
        db = parser.parse_and_store_in_db(filename)
        if db:
            log.info("Parsing complete.")
            if db.top_module:
                log.info(f"Top module: {db.top_module.name}")
            else:
                log.warning("Could not determine top module.")
    except (RuntimeError, IOError) as e:
        log.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

