import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

scaling_from_graph_to_sch = 0.24  # from graphviz to LTspice, about 72/0.24 = 300dpi


# --------------------------------------
# Common geometry mixin for rectangle objects
# --------------------------------------
class RectanglePhysical:
    fig: Optional[Tuple[float, float, float, float]] = None
    shape: Optional[Tuple[int, int, int, int]] = None
    geom: Optional[Polygon] = None

    def fig2shape(self):
        if self.fig is None:
            self.shape = None
            return self.shape
        s = scaling_from_graph_to_sch
        x1, y1, x2, y2 = self.fig
        self.shape = (int(round(x1 * s)), int(round(y1 * s)), int(round(x2 * s)), int(round(y2 * s)))
        return self.shape

    def shape2geom(self):
        if not self.shape or len(self.shape) != 4:
            self.geom = None
            return None
        x1, y1, x2, y2 = self.shape
        self.geom = box(x1, y1, x2, y2)
        return self.geom

    def geom2shape(self):
        if self.geom is None:
            self.shape = None
            return None
        x_min, y_min, x_max, y_max = self.geom.bounds
        self.shape = (int(x_min), int(y_min), int(x_max), int(y_max))
        return self.shape


class PointPhysical:
    fig: Optional[Tuple[float, float]] = None
    shape: Optional[Tuple[int, int]] = None
    geom: Optional[Point] = None

    def fig2shape(self):
        if self.fig is None:
            self.shape = None
            return self.shape
        s = scaling_from_graph_to_sch
        x, y = self.fig
        self.shape = (int(round(x * s)), int(round(y * s)))
        return self.shape

    def shape2geom(self):
        if self.shape is None:
            self.geom = None
            return None
        x, y = self.shape
        self.geom = Point(round(x), round(y))
        return self.geom

    def geom2shape(self):
        if self.geom is None:
            self.shape = None
            return None
        self.shape = (int(self.geom.x), int(self.geom.y))
        return self.shape


@dataclass
class PortPhysical(PointPhysical):
    pass


# -----------------------------
# Pin
# -----------------------------
@dataclass
class PinPhysical(PointPhysical):
    pass


# -----------------------------
# Net
# -----------------------------
@dataclass
class NetPhysical:
    fig: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    shape: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    geom: Optional[MultiLineString] = None
    buffer_patch_points: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)

    def fig2shape(self):
        if not self.fig:
            self.shape = []
            return self.shape

        def scale_point(point: tuple[float, float]) -> tuple[int, int]:
            """Scale a single points (x,y)"""
            s = scaling_from_graph_to_sch
            x, y = point
            return (int(round(x * s)), int(round(y * s)))

        segments = self.fig
        for seg_start, seg_end in segments:
            pt_start = scale_point(seg_start)
            pt_end = scale_point(seg_end)
            if pt_start != pt_end:
                self.shape.append((pt_start, pt_end))
        return self.shape

    def shape2geom(self):
        if not self.shape:
            self.geom = None
            return None
        # Base geometry
        lines = [LineString(seg) for seg in self.shape]
        # also add patch lines
        lines.extend(LineString(seg) for seg in self.buffer_patch_points)

        # Merge into one MultiLineString
        self.geom = MultiLineString(lines)
        return self.geom

    def geom2shape(self):
        if self.geom is None:
            self.shape.clear()
            return None
        # Convert MultiLineString back to list of 2-point segments
        self.shape = [
            ((int(line.coords[0][0]), int(line.coords[0][1])), (int(line.coords[1][0]), int(line.coords[1][1])))
            for line in self.geom.geoms
        ]
        return self.shape


# -----------------------------
# Instance
# -----------------------------
@dataclass
class InstancePhysical(RectanglePhysical):
    pass


# -----------------------------
# Module
# -----------------------------
@dataclass
class ModulePhysical(RectanglePhysical):
    pass


# -----------------------------
# Design
# -----------------------------
@dataclass
class DesignPhysical(RectanglePhysical):
    def clear_all_shapes(self):
        self.shape = None
        for module in super().modules.values():
            module.draw.shape = None
            for inst in module.instances.values():
                inst.draw.shape = None
            for net in module.nets.values():
                net.draw.shape = []
            for pin in module.pins.values():
                pin.draw.shape = None

    def geom2shape(self):
        """Convert all geom objects to shape."""
        self.clear_all_shapes()
        for module in super().modules.values():
            for collection in (
                module.ports.values(),
                module.instances.values(),
                module.nets.values(),
                module.pins.values(),
            ):
                for obj in collection:
                    obj.geom2shape()

    def shape2geom(self):
        """Convert all shape objects to geom."""
        for module in super().modules.values():
            for collection in (
                module.ports.values(),
                module.instances.values(),
                module.nets.values(),
                module.pins.values(),
            ):
                for obj in collection:
                    obj.shape2geom()
