import logging as log
import sys
import time
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree

from schematic_from_netlist.database.netlist_structures import Module, Net
from schematic_from_netlist.interfaces.symbol_library import SymbolLibrary
from schematic_from_netlist.sastar_router.models import CostBuckets, CostEstimator, PNet, RoutingContext
from schematic_from_netlist.sastar_router.sim_router import SimultaneousRouter
from schematic_from_netlist.sastar_router.test_cases import create_hard_test_case
from schematic_from_netlist.sastar_router.visualization import plot_result

log.basicConfig(level=log.INFO)


class GlobalPlacer:
    def __init__(self, db):
        self.db = db
        self.symbol_outlines = SymbolLibrary().get_symbol_outlines()

    def place_design(self):
        log.info("Placing design")
        components = set(self.symbol_outlines.keys())
        module = self.db.design.flat_module
        for inst in module.instances.values():
            if inst.module.name in components:
                inst.draw.fixed_size = True
            log.info(
                f"inst {inst.name} with {inst.hier_prefix=} {inst.hier_module.name=}  seed location {inst.draw.geom=} {inst.draw.fixed_size=}"
            )

    def create_testcase(self, type):
        return db


if __name__ == "__main__":
    # Define nets
    log.basicConfig(level=log.INFO)

    # Create test cases assuming db is not populated
    db = create_testcase("precision")
    global_placer = GlobalPlacer(db)
    global_placer.place_design()
