# src/schematic_from_netlist/router/occupancy.py

import logging as log

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point

log.basicConfig(level=log.DEBUG)


class TrackOccupancyMonitor:
    """
    Monitors track occupancy for each layer.
    """

    def __init__(self, nx, ny, n_layers=1):
        self.occupancy = np.zeros((nx, ny, n_layers))

    def is_occupied(self, x, y, layer=1):
        return self.occupancy[x, y, layer] > 0

    def add_occupancy(self, x, y, layer=1):
        self.occupancy[x, y, layer] += 1

    def remove_occupancy(self, x, y, layer=1):
        self.occupancy[x, y, layer] -= 1


class OccupancyMap:
    """
    Tracks routing resource occupancy on a grid.
    """

    def __init__(self, bounds, grid_size):
        self.bounds = bounds  # (minx, miny, maxx, maxy)
        self.grid_size = grid_size
        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.nx = max(1, int((self.maxx - self.minx) / grid_size) + 1)
        self.ny = max(1, int((self.maxy - self.miny) / grid_size) + 1)
        self.grid = np.zeros((self.nx, self.ny))
        self.grid_via = np.zeros((self.nx, self.ny))
        self.track_occupancy_monitor = TrackOccupancyMonitor(self.nx, self.ny)
        log.debug(f"Initialized occupancy map with grid size {self.nx} x {self.ny}")

    def _world_to_grid(self, x, y):
        # Check if the point is within the bounds
        if int(x) < self.minx or int(x) > self.maxx or int(y) < self.miny or int(y) > self.maxy:
            """
            log.warning(
                f"Point ({x:.2f}, {y:.2f}) is outside occupancy map bounds: ({self.minx:.2f}, {self.miny:.2f}) to ({self.maxx:.2f}, {self.maxy:.2f})"
            )
            """
            # Clamp to the nearest edge
            x = max(self.minx, min(x, self.maxx))
            y = max(self.miny, min(y, self.maxy))

        ix = int((x - self.minx) / self.grid_size)
        iy = int((y - self.miny) / self.grid_size)
        ix = max(0, min(ix, self.nx - 1))
        iy = max(0, min(iy, self.ny - 1))
        log.trace(
            f"Converting world ({x:.2f}, {y:.2f}) to grid ({ix}, {iy}) with bounds ({self.minx:.2f}, {self.miny:.2f}) to ({self.maxx:.2f}, {self.maxy:.2f})"
        )
        log.trace(f"Grid size: {self.grid_size}, nx: {self.nx}, ny: {self.ny}")
        return ix, iy

    def _grid_to_world(self, ix, iy):
        x = self.minx + ix * self.grid_size
        y = self.miny + iy * self.grid_size
        return Point(x, y)

    def update_occupancy(self, geom):
        """
        Updates the occupancy grid for a given tree
        """
        log.debug(f"update_occupancy called with {geom}")
        if isinstance(geom, list):
            geom = MultiLineString(geom)

        # Handle simple LineString
        if isinstance(geom, LineString):
            for x, y in geom.coords:
                self.grid[int(x), int(y)] += 1

            # Handle MultiLineString
        if isinstance(geom, MultiLineString):
            for i, seg in enumerate(geom.geoms):
                for x, y in seg.coords:
                    self.grid[int(x), int(y)] += 1

    def _rasterize_line(self, line):
        """Rasterize a line onto an integer grid."""
        # Skip degenerate lines
        if line.length == 0:
            return

        # Get start and end in world space
        (x1, y1), (x2, y2) = line.coords[0], line.coords[-1]

        # Convert world coordinates to grid coordinates (assumed int grid)
        gx1, gy1 = map(int, self._world_to_grid(x1, y1))
        gx2, gy2 = map(int, self._world_to_grid(x2, y2))

        # Use Bresenham's integer line algorithm
        dx = abs(gx2 - gx1)
        dy = -abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx + dy

        x, y = gx1, gy1
        while True:
            # Only update valid grid cells
            if 0 <= x < self.nx and 0 <= y < self.ny:
                old_value = self.grid[x, y]
                self.grid[x, y] += 1
                self.track_occupancy_monitor.add_occupancy(x, y, 0)
                log.info(f"Grid({x}, {y}) updated {old_value} → {self.grid[x, y]}")
            else:
                log.warning(f"({x}, {y}) is outside grid bounds.")

            if x == gx2 and y == gy2:
                break

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def old_rasterize_line(self, line):
        # Simple line rasterization. A more robust method like Bresenham's might be needed.
        length = line.length
        if length == 0:
            return
        num_steps = max(2, int(length / self.grid_size) * 2 + 2)
        log.info(f"Rasterizing line with {num_steps} steps, length: {length}")
        for i in range(num_steps + 1):
            point = line.interpolate(i / num_steps, normalized=True)
            ix, iy = self._world_to_grid(point.x, point.y)
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                old_value = self.grid[ix, iy]
                self.grid[ix, iy] += 1
                self.track_occupancy_monitor.add_occupancy(ix, iy, 0)  # only 1 layer...
                log.info(
                    f"Updated occupancy at grid ({ix}, {iy}) for point ({point.x:.2f}, {point.y:.2f}): {old_value} -> {self.grid[ix, iy]}"
                )
            else:
                log.warning(f"Point ({point.x}, {point.y}) is outside the grid bounds: ({ix}, {iy})")

    def add_blockage(self, polygon):
        """
        Adds a routing blockage to the occupancy map.
        """
        self._rasterize_polygon(polygon, self.grid, np.inf)

    def add_halo(self, polygon):
        """
        Adds a via blockage to the occupancy map.
        """
        self._rasterize_polygon(polygon, self.grid_via, 1)
        self._rasterize_polygon(polygon, self.grid, 1)

    def _rasterize_polygon(self, polygon, grid_ptr, value):
        minx, miny, maxx, maxy = polygon.bounds
        min_ix, min_iy = self._world_to_grid(minx, miny)
        max_ix, max_iy = self._world_to_grid(maxx, maxy)

        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                # Check if the center of the grid cell is within the polygon
                world_x = self.minx + (ix + 0.5) * self.grid_size
                world_y = self.miny + (iy + 0.5) * self.grid_size
                if polygon.contains(Point(world_x, world_y)):
                    if 0 <= ix < self.nx and 0 <= iy < self.ny:
                        grid_ptr[ix, iy] = value

    def get_congestion_for_segment(self, p1: Point, p2: Point) -> float:
        """
        Gets the maximum congestion for a line segment between two points.
        """
        line = LineString([p1, p2])
        length = line.length
        if length == 0:
            ix, iy = self._world_to_grid(p1.x, p1.y)
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                return self.grid[ix, iy]
            return 0.0

        num_steps = int(length / self.grid_size) + 1
        max_congestion = 0
        for i in range(num_steps + 1):
            point = line.interpolate(i / num_steps, normalized=True)
            ix, iy = self._world_to_grid(point.x, point.y)
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                if self.grid[ix, iy] == np.inf:
                    return np.inf  # Blocked path
                max_congestion = max(max_congestion, self.grid[ix, iy])
        return max_congestion

    def get_congested_nets(self, threshold=2.0):
        # This would require linking grid cells back to nets, which is more complex.
        # For now, this is a placeholder.
        return []

    def get_vertical_track_congestion(self, ix: int) -> float:
        """
        Gets the total congestion for a vertical track.
        """
        if 0 <= ix < self.nx:
            # Sum the occupancy of the entire column to represent track congestion
            return np.sum(self.grid[ix, :])
        return 0.0
