import logging as log
import math

import shapely
from shapely.affinity import rotate, translate
from shapely.geometry import Point, box, LineString, MultiLineString


class LayoutOptimizer:
    def __init__(self, db):
        self.db = db

    def _calculate_best_orientation(self, old_pins, macro_pins_local, centroid):
        orientations = {"R0": 0, "R90": 90, "R180": 180, "R270": 270}
        min_dist = float("inf")
        best_orient = None
        best_pin_map = None
        best_rotated_pins_local = None

        for orient_name, angle in orientations.items():
            rotated_new_pins_local = [rotate(p, angle, origin="center") for p in macro_pins_local]
            new_pins_global_centered = [Point(centroid.x + p.x, centroid.y + p.y) for p in rotated_new_pins_local]

            dist1 = old_pins[0].distance(new_pins_global_centered[0]) + old_pins[1].distance(new_pins_global_centered[1])
            dist2 = old_pins[0].distance(new_pins_global_centered[1]) + old_pins[1].distance(new_pins_global_centered[0])

            current_dist, current_pin_map = (dist1, {0: 0, 1: 1}) if dist1 < dist2 else (dist2, {0: 1, 1: 0})

            if current_dist < min_dist:
                min_dist = current_dist
                best_orient = orient_name
                best_pin_map = current_pin_map
                best_rotated_pins_local = rotated_new_pins_local
        
        return best_orient, best_pin_map, best_rotated_pins_local

    def adjust_location(self, inst):
        """
        Place and orient a new macro shape to minimize distance to existing pins.
        """
        log.info(f"Optimizing instance {inst.name}")
        log.debug(f"  Initial geom: {inst.geom}, orient: {inst.orient}")
        for name, pin in inst.pins.items():
            log.debug(f"  Pin {name}: {pin.geom}")

        old_geom = inst.geom
        old_pins = [p.geom for p in inst.pins.values()]
        pin_names = list(inst.pins.keys())
        centroid = old_geom.centroid

        macro_box = box(-1, -3, 1, 3)
        macro_pins_local = [Point(0, 3), Point(0, -3)]

        best_orient, best_pin_map, best_rotated_pins_local = self._calculate_best_orientation(old_pins, macro_pins_local, centroid)

        p1_geom, p2_geom = old_pins[0], old_pins[1]
        np_local1 = best_rotated_pins_local[best_pin_map[0]]
        np_local2 = best_rotated_pins_local[best_pin_map[1]]

        npgc1 = Point(centroid.x + np_local1.x, centroid.y + np_local1.y)
        npgc2 = Point(centroid.x + np_local2.x, centroid.y + np_local2.y)

        dx, dy = 0, 0
        if best_orient in ["R0", "R180"]:
            dy = (p1_geom.y + p2_geom.y - (npgc1.y + npgc2.y)) / 2
            dy = max(-1.5, min(1.5, dy))
        else:  # R90, R270
            dx = (p1_geom.x + p2_geom.x - (npgc1.x + npgc2.x)) / 2
            dx = max(-1.5, min(1.5, dx))

        final_translation_x = centroid.x + dx
        final_translation_y = centroid.y + dy
        
        orientations = {"R0": 0, "R90": 90, "R180": 180, "R270": 270}
        angle = orientations[best_orient]
        new_geom = translate(rotate(macro_box, angle, origin="center"), xoff=final_translation_x, yoff=final_translation_y)
        
        inst.geom = new_geom
        inst.orient = best_orient

        new_pin_geoms = [
            Point(final_translation_x + best_rotated_pins_local[best_pin_map[0]].x, final_translation_y + best_rotated_pins_local[best_pin_map[0]].y),
            Point(final_translation_x + best_rotated_pins_local[best_pin_map[1]].x, final_translation_y + best_rotated_pins_local[best_pin_map[1]].y)
        ]

        for i, pin_name in enumerate(pin_names):
            pin = inst.pins[pin_name]
            old_pin_geom = old_pins[i]
            new_pin_geom = new_pin_geoms[i]
            pin.geom = new_pin_geom
            
            if pin.net and pin.net.geom:
                new_segment = LineString([old_pin_geom, new_pin_geom])
                existing_lines = list(pin.net.geom.geoms)
                pin.net.geom = MultiLineString(existing_lines + [new_segment])

        log.debug(f"  Final geom: {inst.geom}, orient: {inst.orient}")
        for name, pin in inst.pins.items():
            log.debug(f"  Pin {name}: {pin.geom}")


    def optimize_layout(self):
        for inst in self.db.top_module.get_all_instances().values():
            if len(inst.pins) == 2:
                self.adjust_location(inst)
        self.db.geom2shape()
