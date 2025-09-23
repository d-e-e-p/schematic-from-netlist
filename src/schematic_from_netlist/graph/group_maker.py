# improve wire bundling by inserting buffers at steiner points

import math
import os

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx

from schematic_from_netlist.graph.geom_utils import Geom
from schematic_from_netlist.graph.grouper import Grouper


class GroupMaker:
    def __init__(self, db, schematic_db):
        self.db = db
        self.schematic_db = schematic_db

    def find_target_endpoints(self, net):
        endpoint_map = {}
        for pin in net.pins:
            endpoint_map[pin.shape] = pin.full_name
        return endpoint_map

    def create_buffer_groups(self, net, groups, ordering, endpoint_map):
        """create groups for each new net"""
        clusters = []
        for i, group in enumerate(groups):
            endpoint_names = [endpoint_map[point] for point in group]
            clusters.append(endpoint_names)
            # print(f"Creating group {i}: {group}")
            print(f"Creating group {i}: {endpoint_names}")
        self.db.create_buffering_for_groups(net, ordering, clusters)

    def insert_route_guide_buffers(self):
        # Process groups
        grouper = Grouper()
        sorted_nets = sorted(self.db.nets_by_name.values(), key=lambda net: net.num_conn, reverse=True)
        for net in sorted_nets:
            if net.num_conn > 1:
                if net.shape:
                    endpoint_map = self.find_target_endpoints(net)
                    print(f"Grouping net {net.name} with endpoints {endpoint_map}")
                    groups, ordering = grouper.group_endpoints(list(endpoint_map.keys()))
                    self.create_buffer_groups(net, groups, ordering, endpoint_map)
