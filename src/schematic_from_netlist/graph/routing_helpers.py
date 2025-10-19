from __future__ import annotations

from typing import List

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from schematic_from_netlist.database.netlist_structures import Module


def get_macro_geometries(module: Module):
    """Get all macro geometries in a module."""
    geoms = []
    for i in module.get_all_instances().values():
        if hasattr(i.draw, "geom") and i.draw.geom:
            if isinstance(i.draw.geom, Polygon):
                geoms.append(i.draw.geom)
            elif isinstance(i.draw.geom, MultiPolygon):
                geoms.extend(list(i.draw.geom.geoms))
    return unary_union(geoms) if geoms else Polygon()


def get_halo_geometries(macros, buffer_dist: int = 10) -> Polygon:
    """Get halo geometries around macros."""
    if macros.is_empty:
        return Polygon()
    return macros.buffer(buffer_dist)


def generate_l_paths(p1: Point, p2: Point) -> List[LineString]:
    """Generate two L-shaped paths between two points."""
    if not all(isinstance(p, Point) for p in [p1, p2]):
        return []
    path1 = LineString([(p1.x, p1.y), (p1.x, p2.y), (p2.x, p2.y)])
    path2 = LineString([(p1.x, p1.y), (p2.x, p1.y), (p2.x, p2.y)])
    return [path1, path2]


def get_l_path_corner(path: LineString) -> Point:
    """Get the corner point of an L-shaped path."""
    return Point(path.coords[1])
