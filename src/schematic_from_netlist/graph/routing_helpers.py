from __future__ import annotations

from typing import List

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from schematic_from_netlist.database.netlist_structures import Module


def generate_candidate_paths(
    p1: Point | None,
    p2: Point | None,
    context,
) -> List[LineString]:
    """
    Generate candidate L- and Z-shaped paths between two points.
    Ignores halo escape logic (even if pins lie inside macros).

    Returns unique LineStrings.
    """
    paths: List[LineString] = []
    if not p1 or not p2:
        return paths

    # 1. Generate all L + Z variants
    paths.extend(generate_lz_paths(p1, p2))

    # 2. Remove duplicates by WKT
    unique_paths = []
    seen = set()
    for p in paths:
        if p.wkt not in seen:
            unique_paths.append(p)
            seen.add(p.wkt)

    return unique_paths


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





def generate_lz_paths(p1: Point, p2: Point) -> List[LineString]:
    """Generate L- and Z-shaped paths between two points."""
    if not all(isinstance(p, Point) for p in [p1, p2]):
        return []

    paths = []

    # --- L-shapes (2 options)
    # Go vertical first, then horizontal
    path_L1 = LineString([(p1.x, p1.y), (p1.x, p2.y), (p2.x, p2.y)])
    # Go horizontal first, then vertical
    path_L2 = LineString([(p1.x, p1.y), (p2.x, p1.y), (p2.x, p2.y)])
    paths.extend([path_L1, path_L2])

    # --- Z-shapes (2 options)
    # Midpoints
    mid_x = (p1.x + p2.x) / 2
    mid_y = (p1.y + p2.y) / 2

    # Z1: horizontal–vertical–horizontal, bending through mid_y
    path_Z1 = LineString(
        [
            (p1.x, p1.y),
            (mid_x, p1.y),
            (mid_x, p2.y),
            (p2.x, p2.y),
        ]
    )

    # Z2: vertical–horizontal–vertical, bending through mid_x
    path_Z2 = LineString(
        [
            (p1.x, p1.y),
            (p1.x, mid_y),
            (p2.x, mid_y),
            (p2.x, p2.y),
        ]
    )

    paths.extend([path_Z1, path_Z2])

    return paths


def get_l_path_corner(path: LineString) -> Point:
    """Get the corner point of an L-shaped path."""
    return Point(path.coords[1])
