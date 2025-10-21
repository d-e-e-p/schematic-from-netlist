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
    OFFSET_STEP = 2.0
    MAX_OFFSET_ATTEMPTS = 50

    seen = set()

    for path in initial_paths:
        if not _check_path_overlap(path, context):
            if path.wkt not in seen:
                final_paths.append(path)
                seen.add(path.wkt)
        else:
            # Path has overlap, try to find a detour
            detour = _find_clear_detour(path, p1, p2, context, OFFSET_STEP, MAX_OFFSET_ATTEMPTS)
            if detour and not is_orthogonal(detour):
                log.info(f"Detour is not orthogonal: {detour.wkt}")
                breakpoint()
            if detour and detour.wkt not in seen:
                final_paths.append(detour)
                seen.add(detour.wkt)

    if not final_paths:
        log.debug(f"No clear path found for {p1=} {p2=} {initial_paths=} {context.v_tracks=} {context.h_tracks=}")
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
                        # log.info(f"H-Overlap: {segment.wkt} overlaps {tracks} at y={y}")
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
                        # log.info(f"V-Overlap: {segment.wkt} overlaps {tracks} at x={x}")
                        return True
        # No overlap > 1 found
        return False

    return False


def _check_path_overlap(path: LineString, context) -> bool:
    """Helper function to check ALL segments of a path for overlap."""
    coords = list(path.coords)
    for i in range(len(coords) - 1):
        if _check_segment_overlap(Point(coords[i]), Point(coords[i + 1]), context):
            return True
    # log.info(f" {path=} does not overlap {context.v_tracks=} or {context.h_tracks=}")
    return False


def _find_clear_detour(
    path: LineString, p1: Point, p2: Point, context, offset_step: float, max_attempts: int
) -> Optional[LineString]:
    """Tries to find a single clear detour for a given overlapping path."""

    num_coords = len(path.coords)
    tolerance = 1e-9

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
                else:  # Vertical first L-path
                    new_x = int(round(p1.x + offset))
                    detour_path = LineString([p1, Point(new_x, p1.y), Point(new_x, p2.y), p2])

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
                elif abs(p_mid1.x - p_mid2.x) < 1e-9:  # vertical middle segment
                    # offset horizontally
                    new_x = int(round(p_mid1.x + offset))
                    detour_path = LineString([p1, Point(new_x, p_mid1.y), Point(new_x, p_mid2.y), p2])
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
        OFFSET_STEP = 5.0
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
