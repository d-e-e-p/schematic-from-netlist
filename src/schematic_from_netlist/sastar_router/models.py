import logging as log
import sys
import time
from collections import namedtuple
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree


class Net:
    def __init__(self, name, pins, routed_paths=None):
        self.name = name
        self.pins = pins
        self.routed_paths = routed_paths
        self.total_cost = 0  # Add this line to track the total routing cost
        self.step_costs = {}  # Change from list to dict {(x,y): cost}


@dataclass
class RoutingContext:
    obstacles_index: STRtree
    obstacles: list
    existing_paths_index: STRtree
    existing_paths: list
    halo_geoms: list
    halo_index: STRtree
    terminal_set: set  # terminal locations


@dataclass(frozen=True)
class CostBucket:
    length: int
    via: int
    crossing: int


class CostBuckets:
    BASE = CostBucket(1, 5, 10)
    HALO = CostBucket(4, 5, 10)
    MACRO = CostBucket(100, 5, 10)


class CostEstimator:
    def __init__(self, grid_spacing=1.0):
        self.grid_spacing = grid_spacing

    def _region_costs(self, node, context):
        """Return the cost bucket based on location priority."""
        if self.is_in_macro(node, context):
            return CostBuckets.MACRO
        if self.is_in_halo(node, context):
            return CostBuckets.HALO
        return CostBuckets.BASE

    def get_move_cost(self, current_node, neighbor_node, parent_node, context):
        """Calculate the cost of moving to a neighbor node."""

        costs = self._region_costs(neighbor_node, context)

        total_cost = costs.length

        # Optional penalties depending on geometry
        if self.is_intersecting(neighbor_node, context):
            total_cost += costs.crossing

        if self.is_bend(parent_node, current_node, neighbor_node):
            total_cost += costs.via

        return total_cost

    def is_bend(self, parent_node, current_node, neighbor_node):
        """Detect if three points form a bend (not collinear)."""
        if parent_node is None:
            return False

        # Vectors from parent to current, and current to neighbor
        v1 = (current_node[0] - parent_node[0], current_node[1] - parent_node[1])
        v2 = (neighbor_node[0] - current_node[0], neighbor_node[1] - current_node[1])

        # Cross product: if zero, points are collinear (no bend)
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        return cross != 0

    def is_in_halo(self, point, context):
        """Check if a point is in any halo region."""
        if not context.halo_index:
            return False
        point_geom = Point(point)
        for idx in context.halo_index.query(point_geom, predicate="intersects"):
            halo_geom = context.halo_geoms[idx]
            if halo_geom.contains(point_geom):
                return True
        return False

    # obstacle handling: allow entry but with high cost
    def is_in_macro(self, to_node, context):
        to_point = Point(to_node)
        if to_point in context.terminal_set:
            return False

        if context.obstacles_index:
            intersecting_obs_indices = context.obstacles_index.query(to_point, predicate="intersects")
            if intersecting_obs_indices.size > 0:
                if to_node not in context.terminal_set:
                    # Apply high cost instead of blocking completely
                    return True
        return False

    def is_intersecting(self, to_node, context):
        to_point = Point(to_node)
        # Existing nets check remains unchanged
        if context.existing_paths and context.existing_paths_index:
            intersecting_net_indices = context.existing_paths_index.query(to_point, predicate="intersects")
            if intersecting_net_indices.size > 0:
                return True

        return False
