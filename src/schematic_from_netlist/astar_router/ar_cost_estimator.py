# src/schematic_from_netlist/router/cost.py
import logging as log
from dataclasses import dataclass, field

from shapely.geometry import Point


@dataclass
class Base:
    base_length: float = 0
    base_via: float = 0
    halo_length: float = 0
    halo_via: float = 0
    crossings: float = 0


@dataclass
class Metric(Base):
    pass


@dataclass
class Cost(Base):
    total: float = 0


@dataclass
class CostFactors(Base):
    pass


class CostEstimator:
    """
    Calculates routing costs.
    """

    def __init__(self, occupancy_map):
        self.occupancy_map = occupancy_map

        self.cf = CostFactors()
        self.cf.base_length = 1.0
        self.cf.base_via = 2.0
        self.cf.halo_length = 2.0
        self.cf.halo_via = 100.0
        self.cf.crossings = 5

    def get_cost(self, p1: Point, p2: Point) -> float:
        """
        Calculates the cost of a path segment between two points.
        """
        wire_length = p1.distance(p2)
        congestion = self.occupancy_map.get_congestion_for_segment(p1, p2)

        # TODO
        return cost

    def is_bend(self, nodes):
        """Detect if three points form a bend (not collinear)."""

        parent, current, neighbor = nodes

        if parent is None:
            return False

        # Vectors from parent to current, and current to neighbor
        v1 = (current[0] - parent[0], current[1] - parent[1])
        v2 = (neighbor[0] - current[0], neighbor[1] - current[1])

        # Cross product: if zero, points are collinear (no bend)
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        return cross != 0

    def get_neighbor_move_cost(self, current, neighbor, parent, macro_center, start_node, end_node):
        """Calculate the cost of moving to a neighbor node."""

        # deterine bending into macro
        if current == start_node:
            nodes = [macro_center, current, neighbor]
        elif neighbor == end_node:
            nodes = [current, neighbor, macro_center]
        else:
            nodes = [parent, current, neighbor]
        is_bend = self.is_bend(nodes)

        via_in_halo = self.occupancy_map.grid_via[neighbor]

        metric = Metric()
        metric.base_length = 1.0
        metric.base_via = int(is_bend)
        metric.halo_length = via_in_halo
        metric.halo_via = int(is_bend) * via_in_halo
        metric.crossings = self.occupancy_map.grid[neighbor]

        cost = Cost()
        cost.base_length = metric.base_length * self.cf.base_length
        cost.base_via = metric.base_via * self.cf.base_via
        cost.halo_length = metric.halo_length * self.cf.halo_length
        cost.halo_via = metric.halo_via * self.cf.halo_via
        cost.crossings = metric.crossings * self.cf.crossings
        cost.total = cost.base_length + cost.base_via + cost.halo_length + cost.halo_via + cost.crossings

        log.trace(f"  Neighbor {neighbor}: {metric=}, {cost=}")
        return cost
