# improve wire bundling by inserting buffers at steiner points

import logging as log
import math
import os

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx

from schematic_from_netlist.graph.geom_utils import Geom
from schematic_from_netlist.graph.grouper import Grouper


class SteinerGroupMaker:
    def __init__(self, db):
        self.db = db
        self.schematic_db = db.schematic_db

    def find_target_endpoints(self, net):
        endpoint2pin = {}
        for pin in net.pins.values():
            endpoint2pin[pin.draw.shape] = pin
        return endpoint2pin

    def create_buffer_groups(self, module, net, groups, ordering, endpoint2pin):
        """create groups for each new net"""
        buf_groups = []
        for i, group in enumerate(groups):
            endpoint_pins = [endpoint2pin[point] for point in group]
            buf_groups.append(endpoint_pins)
        self.db.create_buffering_for_groups(module, net, ordering, buf_groups)

    def insert_route_guide_buffers(self):
        # Process groups
        grouper = Grouper()
        for module in self.db.design.modules.values():
            sorted_nets = sorted(module.nets.values(), key=lambda net: net.num_conn, reverse=True)
            for net in sorted_nets:
                if net.num_conn > 2:
                    if net.draw.shape:
                        endpoint2pin = self.find_target_endpoints(net)
                        log.debug(f"Grouping net {net.name} with endpoints {endpoint2pin.keys()}")
                        groups, ordering = grouper.group_endpoints(list(endpoint2pin.keys()))
                        self.create_buffer_groups(module, net, groups, ordering, endpoint2pin)
