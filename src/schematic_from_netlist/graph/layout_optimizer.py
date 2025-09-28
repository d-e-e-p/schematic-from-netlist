import shapely


class LayoutOptimizer:
    def __init__(self, db):
        self.db = db

    def optimize_layout(self):
        for name, inst in self.db.top_module.get_all_instances().items():
            if len(inst.pins) == 2:
                pin_geom = [pin.geom for pin in inst.pins.values()]
                print(f"optimizing inst {name} with current {inst.geom=} {inst.orient=}")
                for name, pin in inst.pins.items():
                    print(f" pin {name} : {pin.geom=}")
                # replace with new geom with macro with shape 6 h x 2 w, with center at 0,0
                # so shape is (-1, -3) to (1, 3)
                # pins call "1" and "2" at (0, 3) and (0, -3)
                # need to place and orient new macro so the 2 pins are closest to existing pins
