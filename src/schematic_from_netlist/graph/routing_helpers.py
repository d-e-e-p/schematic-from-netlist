from __future__ import annotations

import logging as log
from typing import List

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from schematic_from_netlist.database.netlist_structures import Module


def generate_candidate_paths(p1: Point | None, p2: Point | None, context) -> List[LineString]:
    """
    Generate candidate L- and Z-shaped paths between two points.
    If a path overlaps with existing tracks, it attempts to find a clear path by offsetting.
    """

    if not p1 or not p2:
        return []

    initial_paths = generate_lz_paths(p1, p2)
    final_paths = []
    OFFSET_STEP = 5.0
    MAX_OFFSET_ATTEMPTS = 5

    for path in initial_paths:
        has_overlap = False
        coords = list(path.coords)
        for i in range(len(coords) - 1):
            if _check_segment_overlap(Point(coords[i]), Point(coords[i + 1]), context):
                has_overlap = True
                break

        if not has_overlap:
            final_paths.append(path)
        else:
            # Path has overlap, try to find a detour
            for i in range(1, MAX_OFFSET_ATTEMPTS + 1):
                # Try positive and negative offsets
                for sign in [1, -1]:
                    offset = sign * i * OFFSET_STEP
                    # Horizontal segment overlap -> offset vertically
                    if abs(p1.y - path.coords[1][1]) < 1e-9:  # L-path, horizontal first
                        new_y = round(p1.y + offset)
                        detour_path = LineString([p1, Point(p1.x, new_y), Point(p2.x, new_y), p2])
                        if not _check_segment_overlap(Point(p1.x, new_y), Point(p2.x, new_y), context):
                            final_paths.append(detour_path)
                            break
                    # Vertical segment overlap -> offset horizontally
                    else:  # L-path, vertical first
                        new_x = round(p1.x + offset)
                        detour_path = LineString([p1, Point(new_x, p1.y), Point(new_x, p2.y), p2])
                        if not _check_segment_overlap(Point(new_x, p1.y), Point(new_x, p2.y), context):
                            final_paths.append(detour_path)
                            break
                if len(final_paths) > len(initial_paths):  # Found a detour
                    break

    # Remove duplicates
    unique_paths = []
    seen = set()
    for p in final_paths:
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
    mid_x = round((p1.x + p2.x) / 2)
    mid_y = round((p1.y + p2.y) / 2)

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


def _check_segment_overlap(p1: Point, p2: Point, context) -> bool:
    """Check if a segment overlaps with existing tracks, ignoring overlaps inside specified macros."""
    segment = LineString([p1, p2])
    # Ignore check if the segment is fully inside one of the ignored macros
    for macro in context.pin_macros.values():
        if macro.contains(segment):
            return False
    # Horizontal segment
    if abs(p1.y - p2.y) < 1e-9:
        y = p1.y
        seg_x1, seg_x2 = sorted((p1.x, p2.x))
        if y in context.h_tracks:
            for track_x1, track_x2 in context.h_tracks[y]:
                if max(seg_x1, track_x1) < min(seg_x2, track_x2):
                    return True
    # Vertical segment
    elif abs(p1.x - p2.x) < 1e-9:
        x = p1.x
        seg_y1, seg_y2 = sorted((p1.y, p2.y))
        if x in context.v_tracks:
            for track_y1, track_y2 in context.v_tracks[x]:
                if max(seg_y1, track_y1) < min(seg_y2, track_y2):
                    return True
    return False
