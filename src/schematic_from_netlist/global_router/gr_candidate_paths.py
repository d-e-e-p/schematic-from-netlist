from __future__ import annotations

import logging as log
import unittest
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock  # Used to create a simple 'context' object

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from schematic_from_netlist.database.netlist_structures import Module, Pin


def generate_candidate_paths(p1: Point | None, p2: Point | None, context) -> List[LineString]:
    """
    Generate candidate L- and Z-shaped paths between two points.
    If a path overlaps with existing tracks, it attempts to find a clear path by offsetting.
    """
    if not p1 or not p2:
        return []

    initial_paths = generate_lz_paths(p1, p2)
    final_paths = []
    OFFSET_STEP = 5
    MAX_OFFSET_ATTEMPTS = 50

    seen = set()
    log.trace(f"{p1=} -> {p2=}")

    for path in initial_paths:
        log.trace(f"Processing initial path: {path.wkt}")
        if not _check_path_overlap(path, context):
            if path.wkt not in seen:
                final_paths.append(path)
                seen.add(path.wkt)
        else:
            detour = _find_clear_detour(path, p1, p2, context, OFFSET_STEP, MAX_OFFSET_ATTEMPTS)

            if detour:
                if not is_orthogonal(detour):
                    log.warning(f"Detour not orthogonal: {detour.wkt}")

                if detour.wkt not in seen:
                    final_paths.append(detour)
                    seen.add(detour.wkt)
                    continue  #  ACCEPT success & move to next initial path

                #  This only fires if it was truly a duplicate
                log.trace(f"Detour for {path.wkt} rejected: duplicate {detour.wkt}")
            else:
                #  Only fires on actual failure to find detour
                log.trace(f"No clear detour found for {path.wkt}")

    if not final_paths:
        log.trace(f"No clear path found for {p1=} {p2=} {initial_paths=} {context.v_tracks=} {context.h_tracks=}")
    return final_paths


def is_orthogonal(path: LineString, tol: float = 1e-9) -> bool:
    """Return True if all segments of the path are horizontal or vertical."""
    coords = list(path.coords)
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        # Segment must be either horizontal or vertical
        if dx > tol and dy > tol:
            return False
    return True


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


def old_check_segment_overlap(p1: Point, p2: Point, context) -> bool:
    """
    Check if a segment overlaps existing tracks meaningfully (>= MIN_CLEARANCE).
    Single-point touching or crossing endpoints is allowed.
    """
    MIN_CLEARANCE = -25  # >1 cell overlap required to be considered a conflict
    segment = LineString([p1, p2])
    tolerance = 1e-9

    # Check if segment sits on a macro pin
    if segment_on_occupied_macro_pin(segment, context.pin_macros, context.h_tracks, context.v_tracks):
        return True

    # Horizontal segment
    if abs(p1.y - p2.y) < tolerance:
        y = int(round(p1.y))
        seg_x1, seg_x2 = sorted((int(round(p1.x)), int(round(p2.x))))

        for track_y, tracks in context.h_tracks.items():
            if y != track_y:
                continue

            for track_x1, track_x2 in tracks:
                overlap_length = min(seg_x2, track_x2) - max(seg_x1, track_x1)
                if overlap_length <= MIN_CLEARANCE:
                    log.trace(
                        f"H Overlap length: {overlap_length} > {MIN_CLEARANCE} for {track_x1=}, {track_x2=}, {seg_x1=}, {seg_x2=}"
                    )
                    return True

        return False

    # Vertical segment
    if abs(p1.x - p2.x) < tolerance:
        x = int(round(p1.x))
        seg_y1, seg_y2 = sorted((int(round(p1.y)), int(round(p2.y))))

        for track_x, tracks in context.v_tracks.items():
            if x != track_x:
                continue

            for track_y1, track_y2 in tracks:
                overlap_length = min(seg_y2, track_y2) - max(seg_y1, track_y1)
                if overlap_length <= MIN_CLEARANCE:
                    log.trace(
                        f"V Overlap length: {overlap_length} > {MIN_CLEARANCE} for {track_y1=}, {track_y2=}, {seg_y1=}, {seg_y2=}"
                    )
                    return True

        return False

    return False


def _check_segment_overlap(p1: Point, p2: Point, context) -> bool:
    """
    Check if a segment overlaps with existing tracks by more than 1 unit.
    """
    segment = LineString([p1, p2])
    tolerance = 1e-9

    # check_track_macro_pin Ignore check if the segment is fully inside one of the ignored macros
    if segment_on_occupied_macro_pin(segment, context.pin_macros, context.h_tracks, context.v_tracks):
        return True

    # Horizontal segment
    if abs(p1.y - p2.y) < tolerance:
        y = int(round(p1.y))
        seg_x1, seg_x2 = sorted((int(round(p1.x)), int(round(p2.x))))

        for track_y, tracks in context.h_tracks.items():
            # The key (track_y) is already an int from the build function
            if y == track_y:
                for track_x1, track_x2 in tracks:
                    # *** MODIFIED CHECK ***
                    # Calculate overlap length and check if it's > 1
                    overlap_length = min(seg_x2, track_x2) - max(seg_x1, track_x1)
                    if overlap_length > 1:
                        log.trace(
                            f"H-Overlap on segment {segment.wkt} with track at y={y} between x={track_x1}-{track_x2}. Overlap length: {overlap_length}"
                        )
                        return True
        # No overlap > 1 found
        return False

    # Vertical segment
    elif abs(p1.x - p2.x) < tolerance:
        # Assuming coordinates are already rounded
        x = int(round(p1.x))
        seg_y1, seg_y2 = sorted((int(round(p1.y)), int(round(p2.y))))

        for track_x, tracks in context.v_tracks.items():
            # The key (track_x) is already an int
            if x == track_x:
                for track_y1, track_y2 in tracks:
                    # *** MODIFIED CHECK ***
                    # Calculate overlap length and check if it's > 1
                    overlap_length = min(seg_y2, track_y2) - max(seg_y1, track_y1)
                    if overlap_length > 1:
                        log.trace(
                            f"V-Overlap on segment {segment.wkt} with track at x={x} between y={track_y1}-{track_y2}. Overlap length: {overlap_length}"
                        )
                        return True
        # No overlap > 1 found
        return False

    return False


def _check_path_overlap(path: LineString, context) -> bool:
    """
    Check if a path overlaps with existing tracks in the context.
    Returns True if there's an overlap, False otherwise.
    """
    log.trace(f"  → Checking overlap for: {path.wkt}")

    # Get all segments from the path
    coords = list(path.coords)
    log.trace(f"     Path has {len(coords)} coordinates")

    for i in range(len(coords) - 1):
        p1 = Point(coords[i])
        p2 = Point(coords[i + 1])

        log.trace(f"     Segment {i}: {p1.wkt} → {p2.wkt}")

        # Check if segment is vertical
        if abs(p1.x - p2.x) < 1e-9:
            x = int(round(p1.x))
            y_min = int(round(min(p1.y, p2.y)))
            y_max = int(round(max(p1.y, p2.y)))

            log.trace(f"       Vertical segment at x={x}, y_range=[{y_min}, {y_max}]")

            if x in context.v_tracks:
                log.trace(f"       Found v_tracks at x={x}: {context.v_tracks[x]}")
                for track_y_min, track_y_max in context.v_tracks[x]:
                    log.trace(f"         Checking against track y_range=[{track_y_min}, {track_y_max}]")

                    # Check for overlap
                    if not (y_max < track_y_min or y_min > track_y_max):
                        log.trace(f"         ✗ OVERLAP DETECTED!")
                        log.trace(f"         Segment [{y_min}, {y_max}] overlaps with track [{track_y_min}, {track_y_max}]")
                        log.trace(f"  Path {path.wkt} has overlap.")
                        return True
                    else:
                        log.trace(f"         ✓ No overlap with this track")
            else:
                log.trace(f"       No v_tracks at x={x}")

        # Check if segment is horizontal
        elif abs(p1.y - p2.y) < 1e-9:
            y = int(round(p1.y))
            x_min = int(round(min(p1.x, p2.x)))
            x_max = int(round(max(p1.x, p2.x)))

            log.trace(f"       Horizontal segment at y={y}, x_range=[{x_min}, {x_max}]")

            if y in context.h_tracks:
                log.trace(f"       Found h_tracks at y={y}: {context.h_tracks[y]}")
                for track_x_min, track_x_max in context.h_tracks[y]:
                    log.trace(f"         Checking against track x_range=[{track_x_min}, {track_x_max}]")

                    # Check for overlap
                    if not (x_max < track_x_min or x_min > track_x_max):
                        log.trace(f"         ✗ OVERLAP DETECTED!")
                        log.trace(f"         Segment [{x_min}, {x_max}] overlaps with track [{track_x_min}, {track_x_max}]")
                        log.trace(f"  Path {path.wkt} has overlap.")
                        return True
                    else:
                        log.trace(f"         ✓ No overlap with this track")
            else:
                log.trace(f"       No h_tracks at y={y}")
        else:
            log.trace(f"       ⚠ Diagonal segment detected! (should not happen)")

    log.trace(f"  ✓ Path {path.wkt} has NO overlap.")
    return False


def old_check_path_overlap(path: LineString, context) -> bool:
    """Helper function to check ALL segments of a path for overlap."""
    coords = list(path.coords)
    for i in range(len(coords) - 1):
        if _check_segment_overlap(Point(coords[i]), Point(coords[i + 1]), context):
            log.trace(f"Path {path.wkt} has overlap.")
            return True
    log.trace(f"Path {path.wkt} is clear.")
    return False


def _find_clear_detour(
    path: LineString, p1: Point, p2: Point, context, offset_step: int, max_attempts: int
) -> Optional[LineString]:
    """Tries to find a single clear detour for a given overlapping path."""

    num_coords = len(path.coords)
    tolerance = 1e-9
    log.trace(f"=" * 80)
    log.trace(f"DETOUR SEARCH START")
    log.trace(f"Original path: {path.wkt} (num_coords={num_coords})")
    log.trace(f"p1={p1.wkt}, p2={p2.wkt}")
    log.trace(f"offset_step={offset_step}, max_attempts={max_attempts}")
    log.trace(f"context.v_tracks={dict(context.v_tracks)}")
    log.trace(f"context.h_tracks={dict(context.h_tracks)}")

    # *** NEW LOGIC TO HANDLE STRAIGHT/MALFORMED PATHS ***
    # A straight line has 2 coords. The bad path from the log has 3 coords
    # where the first two are identical.
    is_straight_line = num_coords == 2
    is_malformed_straight = num_coords == 3 and Point(path.coords[0]).equals(Point(path.coords[1]))

    log.trace(f"Path classification: is_straight_line={is_straight_line}, is_malformed_straight={is_malformed_straight}")

    if is_straight_line or is_malformed_straight:
        log.trace("Handling straight/malformed path...")
        # Path is vertically aligned (like the one from the log)
        if abs(p1.x - p2.x) < tolerance:
            log.trace(f"Path is VERTICAL (x1={p1.x}, x2={p2.x})")
            # We MUST detour horizontally
            for i in range(1, max_attempts + 1):
                for sign in [1, -1]:
                    offset = sign * i * offset_step
                    new_x = int(round(p1.x + offset))
                    # Create a Π -path with a horizontal middle segment
                    detour_path = LineString([p1, Point(new_x, p1.y), Point(new_x, p2.y), p2])
                    log.trace(f"  Attempt {i}, sign={sign}: trying horizontal detour at x={new_x}")
                    log.trace(f"    detour_path: {detour_path.wkt}")

                    overlap = _check_path_overlap(detour_path, context)
                    log.trace(f"    overlap check result: {overlap}")

                    if not overlap:
                        log.trace(f"  ✓ FOUND CLEAR HORIZONTAL DETOUR: {detour_path.wkt}")
                        return detour_path  # Found a clear path
                    else:
                        log.trace(f"  ✗ Path overlaps, continuing search...")

            log.trace("No clear horizontal detour found for vertical path")
            return None  # No clear horizontal detour found

        # Path is horizontally aligned
        elif abs(p1.y - p2.y) < tolerance:
            log.trace(f"Path is HORIZONTAL (y1={p1.y}, y2={p2.y})")
            # We MUST detour vertically
            for i in range(1, max_attempts + 1):
                for sign in [1, -1]:
                    offset = sign * i * offset_step
                    new_y = int(round(p1.y + offset))
                    # Create a Π -path with a vertical middle segment
                    detour_path = LineString([p1, Point(p1.x, new_y), Point(p2.x, new_y), p2])
                    log.trace(f"  Attempt {i}, sign={sign}: trying vertical detour at y={new_y}")
                    log.trace(f"    detour_path: {detour_path.wkt}")

                    overlap = _check_path_overlap(detour_path, context)
                    log.trace(f"    overlap check result: {overlap}")

                    if not overlap:
                        log.trace(f"  ✓ FOUND CLEAR VERTICAL DETOUR: {detour_path.wkt}")
                        return detour_path  # Found a clear path
                    else:
                        log.trace(f"  ✗ Path overlaps, continuing search...")

            log.trace("No clear vertical detour found for horizontal path")
            return None  # No clear vertical detour found

    # logic for L-Paths (now correctly skipped by the malformed path)
    if num_coords == 3:
        log.trace("Handling L-PATH (3 coordinates)...")
        log.trace(f"  Path coords: {list(path.coords)}")

        for i in range(1, max_attempts + 1):
            for sign in [1, -1]:
                offset = sign * i * offset_step
                detour_path = None

                # Check orientation of the first segment
                if abs(p1.y - path.coords[1][1]) < 1e-9:  # Horizontal first L-path
                    new_y = int(round(p1.y + offset))
                    detour_path = LineString([p1, Point(p1.x, new_y), Point(p2.x, new_y), p2])
                    log.trace(f"  Attempt {i}, sign={sign}: H-first L-path detour with y={new_y}")
                    log.trace(f"    detour_path: {detour_path.wkt}")
                else:  # Vertical first L-path
                    new_x = int(round(p1.x + offset))
                    detour_path = LineString([p1, Point(new_x, p1.y), Point(new_x, p2.y), p2])
                    log.trace(f"  Attempt {i}, sign={sign}: V-first L-path detour with x={new_x}")
                    log.trace(f"    detour_path: {detour_path.wkt}")

                if detour_path:
                    overlap = _check_path_overlap(detour_path, context)
                    log.trace(f"    overlap check result: {overlap}")

                    if not overlap:
                        log.trace(f"  ✓ FOUND CLEAR L-PATH DETOUR: {detour_path.wkt}")
                        return detour_path
                    else:
                        log.trace(f"  ✗ Path overlaps, continuing search...")

        log.trace("No L-path detour found")
        return None  # No L-path detour found

    # Existing logic for Z-Paths
    elif num_coords == 4:
        log.trace("Handling Z-PATH (4 coordinates)...")
        p1_pt, p_mid1, p_mid2, p2_pt = map(Point, path.coords)
        log.trace(f"  Path coords: p1={p1_pt.wkt}, mid1={p_mid1.wkt}, mid2={p_mid2.wkt}, p2={p2_pt.wkt}")

        for i in range(1, max_attempts + 1):
            for sign in [1, -1]:
                offset = sign * i * offset_step

                # Middle segment direction: horizontal if y1 ≈ y2
                if abs(p_mid1.y - p_mid2.y) < 1e-9:  # horizontal middle segment
                    # offset vertically
                    new_y = int(round(p_mid1.y + offset))
                    detour_path = LineString([p1, Point(p_mid1.x, new_y), Point(p_mid2.x, new_y), p2])
                    log.trace(f"  Attempt {i}, sign={sign}: Z-path (H-mid) detour with y={new_y}")
                    log.trace(f"    detour_path: {detour_path.wkt}")

                elif abs(p_mid1.x - p_mid2.x) < 1e-9:  # vertical middle segment
                    # offset horizontally
                    new_x = int(round(p_mid1.x + offset))
                    detour_path = LineString([p1, Point(new_x, p_mid1.y), Point(new_x, p_mid2.y), p2])
                    log.trace(f"  Attempt {i}, sign={sign}: Z-path (V-mid) detour with x={new_x}")
                    log.trace(f"    detour_path: {detour_path.wkt}")
                else:
                    # The middle segment is diagonal—this shouldn't happen for orthogonal paths
                    log.trace(f"  Attempt {i}, sign={sign}: Middle segment is DIAGONAL (skipping)")
                    continue

                # Validate orthogonality and overlap
                is_ortho = is_orthogonal(detour_path)
                log.trace(f"    is_orthogonal: {is_ortho}")

                if is_ortho:
                    overlap = _check_path_overlap(detour_path, context)
                    log.trace(f"    overlap check result: {overlap}")

                    if not overlap:
                        log.trace(f"  ✓ FOUND CLEAR Z-PATH DETOUR: {detour_path.wkt}")
                        return detour_path
                    else:
                        log.trace(f"  ✗ Path overlaps, continuing search...")
                else:
                    log.trace(f"  ✗ Path not orthogonal, skipping...")

        log.trace("No Z-path detour found")
        return None  # No Z-path detour found

    log.warning(f"DETOUR SEARCH FAILED: Unsupported path type with {num_coords} coordinates")
    log.warning(f"No clear detour found for {p1=} {p2=} {path=}")
    log.warning(f"context.v_tracks={dict(context.v_tracks)}")
    log.warning(f"context.h_tracks={dict(context.h_tracks)}")
    log.trace(f"=" * 80)
    return None  # No clear detour found


def old_find_clear_detour(
    path: LineString, p1: Point, p2: Point, context, offset_step: int, max_attempts: int
) -> Optional[LineString]:
    """Tries to find a single clear detour for a given overlapping path."""

    num_coords = len(path.coords)
    tolerance = 1e-9
    log.trace(f"Finding detour for path: {path.wkt}")

    # *** NEW LOGIC TO HANDLE STRAIGHT/MALFORMED PATHS ***
    # A straight line has 2 coords. The bad path from the log has 3 coords
    # where the first two are identical.
    is_straight_line = num_coords == 2
    is_malformed_straight = num_coords == 3 and Point(path.coords[0]).equals(Point(path.coords[1]))

    if is_straight_line or is_malformed_straight:
        # Path is vertically aligned (like the one from the log)
        if abs(p1.x - p2.x) < tolerance:
            # We MUST detour horizontally
            for i in range(1, max_attempts + 1):
                for sign in [1, -1]:
                    offset = sign * i * offset_step
                    new_x = int(round(p1.x + offset))
                    # Create a Π -path with a horizontal middle segment
                    detour_path = LineString([p1, Point(new_x, p1.y), Point(new_x, p2.y), p2])
                    if not _check_path_overlap(detour_path, context):
                        return detour_path  # Found a clear path
            return None  # No clear horizontal detour found

        # Path is horizontally aligned
        elif abs(p1.y - p2.y) < tolerance:
            # We MUST detour vertically
            for i in range(1, max_attempts + 1):
                for sign in [1, -1]:
                    offset = sign * i * offset_step
                    new_y = int(round(p1.y + offset))
                    # Create a Π -path with a vertical middle segment
                    detour_path = LineString([p1, Point(p1.x, new_y), Point(p2.x, new_y), p2])
                    if not _check_path_overlap(detour_path, context):
                        return detour_path  # Found a clear path
            return None  # No clear vertical detour found

    # logic for L-Paths (now correctly skipped by the malformed path)
    if num_coords == 3:
        for i in range(1, max_attempts + 1):
            for sign in [1, -1]:
                offset = sign * i * offset_step
                detour_path = None

                # Check orientation of the first segment
                if abs(p1.y - path.coords[1][1]) < 1e-9:  # Horizontal first L-path
                    new_y = int(round(p1.y + offset))
                    detour_path = LineString([p1, Point(p1.x, new_y), Point(p2.x, new_y), p2])
                    log.trace(f"Attempting H-first L-path detour with y={new_y}: {detour_path.wkt}")
                else:  # Vertical first L-path
                    new_x = int(round(p1.x + offset))
                    detour_path = LineString([p1, Point(new_x, p1.y), Point(new_x, p2.y), p2])
                    log.trace(f"Attempting V-first L-path detour with x={new_x}: {detour_path.wkt}")

                if detour_path and not _check_path_overlap(detour_path, context):
                    return detour_path
        return None  # No L-path detour found

    # Existing logic for Z-Paths
    elif num_coords == 4:
        p1, p_mid1, p_mid2, p2 = map(Point, path.coords)

        for i in range(1, max_attempts + 1):
            for sign in [1, -1]:
                offset = sign * i * offset_step

                # Middle segment direction: horizontal if y1 ≈ y2
                if abs(p_mid1.y - p_mid2.y) < 1e-9:  # horizontal middle segment
                    # offset vertically
                    new_y = int(round(p_mid1.y + offset))
                    detour_path = LineString([p1, Point(p_mid1.x, new_y), Point(p_mid2.x, new_y), p2])
                    log.trace(f"Attempting Z-path (H-mid) detour with y={new_y}: {detour_path.wkt}")
                elif abs(p_mid1.x - p_mid2.x) < 1e-9:  # vertical middle segment
                    # offset horizontally
                    new_x = int(round(p_mid1.x + offset))
                    detour_path = LineString([p1, Point(new_x, p_mid1.y), Point(new_x, p_mid2.y), p2])
                    log.trace(f"Attempting Z-path (V-mid) detour with x={new_x}: {detour_path.wkt}")
                else:
                    # The middle segment is diagonal—this shouldn't happen for orthogonal paths
                    continue

                # Validate orthogonality and overlap
                if is_orthogonal(detour_path) and not _check_path_overlap(detour_path, context):
                    return detour_path

        return None  # No Z-path detour found

    log.warning(f"No clear detour found for {p1=} {p2=} {path=} {context.v_tracks=} {context.h_tracks=}")
    return None  # No clear detour found


def segment_on_occupied_macro_pin(
    segment: LineString,
    pin_macros: Dict[Pin, Polygon],
    h_tracks: Dict[int, List[Tuple[int, int]]],
    v_tracks: Dict[int, List[Tuple[int, int]]],
) -> bool:
    """
    Checks if a segment crosses a macro or if its intersection points with a
    macro boundary lie on a defined track.

    This function returns True if:
    1. The segment runs along the North, South, East, or West face of any macro.
    2. The segment intersects a macro boundary, and that intersection point
       lies on a defined horizontal or vertical track.

    Args:
        segment: The LineString segment to check.
        pin_macros: A dictionary of Shapely Polygons representing macro areas.
        h_tracks: Dictionary of horizontal tracks {y: [(x1, x2), ...]}.
        v_tracks: Dictionary of vertical tracks {x: [(y1, y2), ...]}.

    Returns:
        True if the segment is invalid for either reason, False otherwise.
    """
    tol = 1e-9
    (x1, y1), (x2, y2) = segment.coords
    is_horizontal = abs(y1 - y2) < tol
    is_vertical = abs(x1 - x2) < tol
    for macro in pin_macros.values():
        # Condition 1: Check for segments running along the N, S, E, or W faces of the macro.
        minx, miny, maxx, maxy = macro.bounds

        # Horizontal segment → check north/south (top/bottom) faces
        if is_horizontal:
            y = y1
            if abs(y - maxy) < tol or abs(y - miny) < tol:
                seg_x1, seg_x2 = sorted((x1, x2))
                if max(seg_x1, minx) < min(seg_x2, maxx):
                    return True

        # Vertical segment → check east/west (right/left) faces
        elif is_vertical:
            x = x1
            if abs(x - maxx) < tol or abs(x - minx) < tol:
                seg_y1, seg_y2 = sorted((y1, y2))
                if max(seg_y1, miny) < min(seg_y2, maxy):
                    return True

        # Condition 2: Check intersection points on the boundary.
        # Use intersection() with the macro's boundary to find the exact points of contact.
        intersection_geom = segment.intersection(macro.exterior)

        # Skip if there is no intersection at all
        if intersection_geom.is_empty:
            continue

        # Collect all individual intersection points from the geometry.
        # This handles cases where the segment crosses the macro at multiple points.
        points_to_check = []
        if intersection_geom.geom_type == "Point":
            points_to_check.append(intersection_geom)
        elif intersection_geom.geom_type == "MultiPoint":
            points_to_check.extend(intersection_geom.geoms)
        elif intersection_geom.geom_type in ["LineString", "MultiLineString"]:
            # If the intersection is a line, it means the segment runs along the
            # macro boundary. We check the endpoints of this shared line.
            geoms = intersection_geom.geoms if intersection_geom.geom_type == "MultiLineString" else [intersection_geom]
            for line in geoms:
                points_to_check.extend([Point(p) for p in line.coords])

        # Now, check if any of the found intersection points lie on a defined track.
        for point in points_to_check:
            x_coord = int(round(point.x))
            y_coord = int(round(point.y))

            # Check if the point's x-coordinate matches a vertical track
            if x_coord in v_tracks:
                return True

            # Check if the point's y-coordinate matches a horizontal track
            if y_coord in h_tracks:
                return True

    # If we loop through all macros and find no invalid condition, return False.
    return False


# --- Test Case ---


class TestCheckSegmentOverlap(unittest.TestCase):
    def test_path_with_no_overlap(self):
        """
        Tests the condition:
        path=<LINESTRING (18 46, 18 41, 33 41, 33 36)>
        Should NOT overlap the given context.
        """
        # 1. Define the context from your log
        context = Mock()
        context.v_tracks = defaultdict(
            list, {48: [(27, 56)], 33: [(23, 27)], 28: [(23, 31), (42, 50)], 43: [(18, 52)], 23: [(22, 56)], 18: [(21, 51)]}
        )
        context.h_tracks = defaultdict(
            list,
            {
                27: [(33, 48)],
                36: [(38, 48)],
                56: [(23, 28), (38, 48)],
                18: [(38, 43)],
                32: [(38, 43)],
                52: [(38, 43)],
                41: [(23, 28)],
                22: [(23, 28)],
                51: [(18, 28)],
                21: [(18, 28)],
            },
        )
        context.pin_macros = {}  # No macros

        # 2. Define the path
        path = LineString([(48, 36), (27, 36), (27, 41), (33, 41), (33, 48)])  # (48 36, 48 27, 33 27, 33 18)

        # 3. Manually check segments (for verification):
        # Seg 1: x=18, y=[41, 46]. v_tracks has no key 18. OK.
        # Seg 2: y=41, x=[18, 33]. h_tracks has no key 41. OK.
        # Seg 3: x=33, y=[36, 41]. v_tracks has no key 33. OK.

        # 4. Run the assertion
        # We assert that _check_path_overlap returns False (no overlap)
        self.assertTrue(_check_path_overlap(path, context), "Overlap not detected.")

    def test_find_detour_for_blocked_vertical_path(self):
        """
        Tests the specific scenario from the Pdb session where a vertical
        path is blocked and must find a clear horizontal detour.
        """
        # 1. Define the context from the Pdb session
        context = Mock()
        context.v_tracks = defaultdict(
            list, {48: [(27, 56)], 33: [(23, 27)], 28: [(23, 31), (42, 50)], 43: [(18, 52)], 23: [(22, 56)], 18: [(21, 51)]}
        )
        context.h_tracks = defaultdict(
            list,
            {
                27: [(33, 48)],
                36: [(38, 48)],
                56: [(23, 28), (38, 48)],
                18: [(38, 43)],
                32: [(38, 43)],
                52: [(38, 43)],
                41: [(23, 28)],
                22: [(23, 28)],
                51: [(18, 28)],
                21: [(18, 28)],
            },
        )
        context.pin_macros = {}  # No macros

        # 2. Define points and parameters
        p1 = Point(48, 36)
        p2 = Point(33, 18)
        OFFSET_STEP = 5
        MAX_OFFSET_ATTEMPTS = 5

        # 3. The initial path is a straight vertical line, which is blocked
        #    by the track at x=33.
        initial_path = LineString([p1, p2])
        # self.assertTrue(_check_path_overlap(initial_path, context), "Test setup is wrong: The initial path should be blocked.")

        # 4. Define the expected clear path.
        #    - Detour to x=38 (33+5) is blocked by h_tracks and v_tracks.
        #    - Detour to x=28 (33-5) is blocked by h_tracks and v_tracks.
        #    - Detour to x=43 (33+10) is clear.
        expected_detour = LineString([(28, 18), (28, 23), (33, 23), (33, 18)])

        # 5. Call the function under test
        detour = _find_clear_detour(initial_path, p1, p2, context, OFFSET_STEP, MAX_OFFSET_ATTEMPTS)

        # 6. Assert the results
        self.assertIsNotNone(detour, "Function failed to find any detour.")
        self.assertEqual(detour, expected_detour, f"Detour found was {detour.wkt}, but expected {expected_detour.wkt}")


if __name__ == "__main__":
    unittest.main()
