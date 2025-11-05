from __future__ import annotations

import logging as log
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

if TYPE_CHECKING:
    from schematic_from_netlist.database.netlist_structures import Design  # type hints only

# TODO: scaling from gv should depend on pin-to-pin or wire-to-wire spacing
# scaling_from_gv_to_dr = 0.24  # from graphviz to LTspice, about 72/0.24 = 300dpi
scaling_from_gv_to_dr = 1.0 / 6.0  # from graphviz to LTspice, for 36 gv -> 6 units dr
scaling_from_dr_to_ltspice = 1.0  # from dr to LTspice,

# fig   : extracted from graphviz
# shape : used by LTspice
# geom  : used for shapely optimizations


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
        s = scaling_from_gv_to_dr
        x1, y1, x2, y2 = self.fig
        self.shape = (int(round(x1 * s)), int(round(y1 * s)), int(round(x2 * s)), int(round(y2 * s)))
        return self.shape

    def shape2fig(self):
        if self.shape is None:
            self.fig = None
            return self.fig
        s = scaling_from_gv_to_dr
        x1, y1, x2, y2 = self.shape
        self.fig = (round(x1 / s, 2), round(y1 / s, 2), round(x2 / s, 2), round(y2 / s, 2))
        return self.fig

    def shape2geom(self):
        if not self.shape:
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
        s = scaling_from_gv_to_dr
        x, y = self.fig
        self.shape = (int(round(x * s)), int(round(y * s)))
        return self.shape

    def shape2fig(self):
        if self.shape is None:
            self.fig = None
            return self.fig
        s = scaling_from_gv_to_dr
        x, y = self.shape
        self.fig = (round(x / s, 2), round(y / s, 2))
        return self.fig

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
    fixed: bool = False


# -----------------------------
# Pin
# -----------------------------
@dataclass
class PinPhysical(PointPhysical):
    fixed: bool = False
    direction: str = "C"


# -----------------------------
# Net
# -----------------------------
@dataclass
class NetPhysical:
    fig: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    shape: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    geom: Optional[MultiLineString] = None
    buffer_patch_points: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    topo: Optional["Topology"] = None
    total_cost: Optional[float] = float("inf")
    step_costs: Optional[Dict[[int, int], int]] = field(default_factory=dict)

    def fig2shape(self):
        if not self.fig:
            self.shape = []
            return self.shape

        def scale_point(point: tuple[float, float]) -> tuple[int, int]:
            """Scale a single points (x,y)"""
            s = scaling_from_gv_to_dr
            x, y = point
            return (int(round(x * s)), int(round(y * s)))

        segments = self.fig
        for seg_start, seg_end in segments:
            pt_start = scale_point(seg_start)
            pt_end = scale_point(seg_end)
            if pt_start != pt_end:
                self.shape.append((pt_start, pt_end))
        return self.shape

    def shape2fig(self):
        pass  # don't bother with routing back into fig

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

        # Normalize geometry into a list of LineStrings
        if isinstance(self.geom, LineString):
            lines = [self.geom]
        elif isinstance(self.geom, MultiLineString):
            lines = list(self.geom.geoms)
        elif isinstance(self.geom, list):  # Already list of geometries
            lines = self.geom
        else:
            log.error(f"Unsupported geometry type: {type(self.geom)}")
            return None

        # Convert consecutive coordinate pairs → segment list
        segments = []
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                x1, y1 = map(int, coords[i])
                x2, y2 = map(int, coords[i + 1])
                segments.append(((x1, y1), (x2, y2)))
        self.shape = segments
        return self.shape


# -----------------------------
# Instance
# -----------------------------
@dataclass
class InstancePhysical(RectanglePhysical):
    rank: int = -1
    orient: str = "R0"
    fixed_size: bool = False


# -----------------------------
# Module
# -----------------------------
@dataclass
class ModulePhysical(PointPhysical):
    pass


# -----------------------------
# Design
# -----------------------------
@dataclass
class DesignPhysical(RectanglePhysical):
    design: Optional["Design"] = None

    def clear_all_shapes(self):
        self.shape = None
        for module in self.design.modules.values():
            module.draw.shape = None
            for inst in module.instances.values():
                inst.draw.shape = None
            for net in module.nets.values():
                net.draw.shape = []
            for pin in module.pins.values():
                pin.draw.shape = None

    def _for_each_draw_obj(self, fn: Callable):
        modules = list(self.design.modules.values())
        if self.design.flat_module:
            modules.append(self.design.flat_module)
        for module in modules:
            for collection in (
                [module],
                module.ports.values(),
                module.instances.values(),
                module.nets.values(),
                module.pins.values(),
            ):
                for obj in collection:
                    if hasattr(obj, "draw"):
                        fn(obj.draw)

    def fig2shape(self):
        self.clear_all_shapes()
        self._for_each_draw_obj(lambda d: d.fig2shape())

    def shape2fig(self):
        self._for_each_draw_obj(lambda d: d.shape2fig())

    def geom2shape(self):
        self.clear_all_shapes()
        self._for_each_draw_obj(lambda d: d.geom2shape())

    def shape2geom(self):
        self._for_each_draw_obj(lambda d: d.shape2geom())

    def fig2geom(self):
        self.fig2shape()
        self.shape2geom()

    def geom2fig(self):
        self.geom2shape()
        self.shape2fig()
