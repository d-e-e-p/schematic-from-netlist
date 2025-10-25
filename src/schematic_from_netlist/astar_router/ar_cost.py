# src/schematic_from_netlist/router/cost.py
from shapely.geometry import Point


class Cost:
    """
    Calculates routing costs.
    """

    def __init__(self, occupancy_map, wire_length_weight=1.0, crossing_weight=50.0, bend_penalty=4.0, halo_cost=20.0):
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

    def get_base_cost(self, p1: Point) -> float:
        """
        Calculates the base cost of a path segment, without congestion.
        """
        return self.wire_length_weight * self.occupancy_map.grid_size

