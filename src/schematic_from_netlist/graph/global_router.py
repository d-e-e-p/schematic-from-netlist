# improve wire bundling by inserting buffers at steiner points

from __future__ import annotations

import logging as log
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import shapely
from rtree import index
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import linemerge, unary_union

from schematic_from_netlist.database.netlist_structures import Module, Net, Pin
from schematic_from_netlist.graph.geom_utils import Geom


@dataclass
class Topology:
    net: Net
    junctions: Dict[str, Junction] = field(default_factory=dict)


@dataclass
class Junction:
    location: Tuple[int, int]
    children: List[Junction | Pin] = field(default_factory=list)


class GlobalRouter:
    def __init__(self, db):
        self.db = db
        self.junctions: Dict[Module, List[Topology]] = {}

    def insert_routing_junctions(self):
        # Process groups
        for module in self.db.design.modules.values():
            sorted_nets = sorted(module.nets.values(), key=lambda net: net.num_conn)
            for net in sorted_nets:
                if 2 < net.num_conn < self.db.fanout_threshold:
                    self.process_net(module, net)
        return self.junctions

    def process_net(self, module, net):
        breakpoint()
