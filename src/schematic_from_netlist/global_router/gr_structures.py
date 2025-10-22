from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rtree import index
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from schematic_from_netlist.database.netlist_structures import Module, Net, Pin


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
    macros: BaseGeometry = field(default_factory=Polygon)
    halos: BaseGeometry = field(default_factory=Polygon)
    congestion_idx: index.Index = field(default_factory=index.Index)
    other_nets_geoms: List[LineString] = field(default_factory=list)
    h_tracks: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    v_tracks: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    pin_macros: Dict[Pin, Polygon] = field(default_factory=dict)
    module: Module = field(default_factory=Module)
    net: Net = field(default_factory=Net)


@dataclass
class Topology:
    net: Net
    junctions: List[Junction] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    context: RoutingContext = field(default_factory=RoutingContext)


@dataclass
class Junction:
    name: str
    location: Point
    children: Set[Junction | Pin] = field(default_factory=set)
    geom: Optional[MultiLineString] = None

    def __hash__(self):
        return hash((self.name, self.location))
