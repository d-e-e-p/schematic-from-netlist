from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rtree import index
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from schematic_from_netlist.database.netlist_structures import Module, Net, Pin


@dataclass
class Topology:
    net: Net
    junctions: List[Junction] = field(default_factory=list)
    metrics: Metrics | None = None


@dataclass
class Junction:
    name: str
    location: Point
    children: Set[Junction | Pin] = field(default_factory=set)

    def __hash__(self):
        return hash((self.name, self.location))


@dataclass
class Metrics:
    # Geometric parameters
    wirelength: float = 0.0
    macro_overlap: float = 0.0
    halo_overlap: float = 0.0
    intersecting_length: float = 0.0
    intersecting_crossings: int = 0

    # Individual weighted costs
    cost_wirelength: float = 0.0
    cost_macro: float = 0.0
    cost_halo: float = 0.0
    cost_congestion: float = 0.0
    cost_crossing: float = 0.0
    cost_track_overlap: float = 0.0
    cost_macro_junction_penalty: float = 0.0
    cost_halo_junction_penalty: float = 0.0

    # Aggregate total
    total_cost: float = 0.0

    def to_dict(self):
        """Return the metrics as a plain dictionary."""
        return asdict(self)

    def __str__(self):
        """Return a short summary string for logging."""
        return (
            f"wl={self.cost_wirelength:.1f}, "
            f"macro={self.cost_macro:.1f}, "
            f"halo={self.cost_halo:.1f}, "
            f"cong={self.cost_congestion:.1f}, "
            f"cross={self.cost_crossing:.1f}, "
            f"track={self.cost_track_overlap:.1f}, "
            f"pen={self.cost_macro_junction_penalty + self.cost_halo_junction_penalty:.1f}, "
            f"total={self.total_cost:.1f}"
        )


@dataclass
class RoutingContext:
    macros: Polygon | BaseGeometry
    halos: Polygon | BaseGeometry
    congestion_idx: index.Index
    other_nets_geoms: List[LineString]
    h_tracks: Dict[float, List[Tuple[float, float]]]
    v_tracks: Dict[float, List[Tuple[float, float]]]
    pin_macros: Dict[Pin, Polygon]
    module: Optional[Module] = None
