# src/schematic_from_netlist/router/cost.py
import logging as log

from shapely.geometry import Point


class Cost:
    """
    Calculates routing costs.
    """

    def __init__(self, occupancy_map, wire_length_weight=1.0, crossing_weight=10.0, bend_penalty=4.0, halo_cost=20.0):
        self.occupancy_map = occupancy_map
        self.wire_length_weight = wire_length_weight
        self.crossing_weight = crossing_weight
        self.bend_penalty = bend_penalty
        self.halo_cost = halo_cost

    def get_cost(self, p1: Point, p2: Point) -> float:
        """
        Calculates the cost of a path segment between two points.
        """
        wire_length = p1.distance(p2)
        congestion = self.occupancy_map.get_congestion_for_segment(p1, p2)

        cost = self.wire_length_weight * wire_length + self.crossing_weight * congestion
        return cost

    def is_bend(self, current, neighbor, parent):
        """Detect if three points form a bend (not collinear)."""
        if parent is None:
            return False

        # Vectors from parent to current, and current to neighbor
        v1 = (current[0] - parent[0], current[1] - parent[1])
        v2 = (neighbor[0] - current[0], neighbor[1] - current[1])

        # Cross product: if zero, points are collinear (no bend)
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        return cross != 0

    def get_neighbor_move_cost(self, current, neighbor, parent, macro_center):
        cost = self.wire_length_weight
        cost += self.occupancy_map.grid[neighbor] * self.crossing_weight
        if not parent:
            parent = macro_center
        is_bend = self.is_bend(current, neighbor, parent)
        if is_bend:
            cost += self.bend_penalty
            cost += self.occupancy_map.grid_via[neighbor] * self.bend_penalty
        log.debug(
            f"  Neighbor {neighbor}: route_occupancy={self.occupancy_map.grid[neighbor]} via_occupancy={self.occupancy_map.grid_via[neighbor]} bend:{is_bend} {cost=}"
        )
        return cost
