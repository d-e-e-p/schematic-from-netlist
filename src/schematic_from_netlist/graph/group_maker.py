# improve wire bundling by inserting buffers at steiner points

import math
import os
import logging

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx

from schematic_from_netlist.graph.geom_utils import Geom
from schematic_from_netlist.graph.grouper import Grouper


class SteinerGroupMaker:
    def __init__(self, db):
        self.db = db
        self.schematic_db = db.schematic_db

    def find_target_endpoints(self, net, cluster_id):
        endpoint_map = {}
        for pin in net.pins:
            if pin.instance.partition == cluster_id:
                endpoint_map[pin.shape] = pin.full_name
        return endpoint_map

    def create_buffer_groups(self, net, groups, ordering, endpoint_map, cluster_id):
        """create groups for each new net"""
        buf_groups = []
        for i, group in enumerate(groups):
            endpoint_names = [endpoint_map[point] for point in group]
            buf_groups.append(endpoint_names)
            logging.debug(f"Creating group {i}: {cluster_id=} {endpoint_names=}")
        self.db.create_buffering_for_groups(net, ordering, buf_groups, cluster_id)

    def insert_route_guide_buffers(self):
        # Process groups
        grouper = Grouper()
        sorted_nets = sorted(self.db.nets_by_name.values(), key=lambda net: net.num_conn, reverse=True)
        for net in sorted_nets:
            if net.num_conn > 2:
                if net.shape:
                    for cluster_id, cluster in self.db.top_module.clusters.items():
                        endpoint_map = self.find_target_endpoints(net, cluster_id)
                        logging.debug(f"Grouping net {net.name} in cluster {cluster_id} with endpoints {endpoint_map}")
                        groups, ordering = grouper.group_endpoints(list(endpoint_map.keys()))
                        self.create_buffer_groups(net, groups, ordering, endpoint_map, cluster_id)
