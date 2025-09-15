import glob
import json  # noqa: F401
import math
import os

import sexpdata


class KiCadSymbolLibParser:
    def __init__(self, file_path):
        """Initializes the parser with the path to the .lib file."""
        self.file_path = file_path
        self.raw_symbols_data = {}
        self.flat_symbols_data = {}

    def parse_properties_list(self, prop_list):
        """
        Parses a list of s-expression properties into a dictionary.
        e.g. [(Symbol('start'), -1.27, 1.27), (Symbol('end'), 1.27, -1.27)]
        becomes {'start': '-1.27 1.27', 'end': '1.27 -1.27'}
        """
        properties = {}
        for item in prop_list:
            if isinstance(item, list) and item and isinstance(item[0], sexpdata.Symbol):
                key = item[0].value()
                if len(item) > 2:
                    value = " ".join(str(v) for v in item[1:])
                elif len(item) == 2:
                    value = item[1]
                else:
                    value = ""

                if isinstance(value, list):
                    properties[key] = self.parse_properties_list(value)
                else:
                    properties[key] = value
        return properties

    def _parse_symbol_def(self, symbol_def):
        symbol_name = symbol_def[1]
        symbol_data = {"properties": {}, "pins": [], "rects": [], "children": [], "extends": None, "inline_children": {}}

        for item in symbol_def[2:]:
            if not isinstance(item, list) or not item or not isinstance(item[0], sexpdata.Symbol):
                continue

            item_type = item[0].value()

            if item_type == "property":
                symbol_data["properties"][item[1]] = item[2]
            elif item_type == "pin":
                symbol_data["pins"].append(self.parse_properties_list(item[1:]))
            elif item_type == "rectangle":
                symbol_data["rects"].append(self.parse_properties_list(item[1:]))
            elif item_type == "symbol":
                child_name = item[1]
                child_content = item[2:]

                has_graphics = any(isinstance(sub, list) and sub and isinstance(sub[0], sexpdata.Symbol) and sub[0].value() in ["pin", "rectangle", "circle", "arc", "polyline", "text"] for sub in child_content)

                if has_graphics:  # Inline definition
                    inline_child_data = {"pins": [], "rects": []}
                    for sub_item in child_content:
                        if isinstance(sub_item, list) and sub_item and isinstance(sub_item[0], sexpdata.Symbol):
                            sub_item_type = sub_item[0].value()
                            if sub_item_type == "pin":
                                inline_child_data["pins"].append(self.parse_properties_list(sub_item[1:]))
                            elif sub_item_type == "rectangle":
                                inline_child_data["rects"].append(self.parse_properties_list(sub_item[1:]))
                    symbol_data["inline_children"][child_name] = inline_child_data
                else:  # Instance
                    child_info = {"lib_id": child_name}
                    child_info.update(self.parse_properties_list(child_content))
                    symbol_data["children"].append(child_info)

            elif item_type == "extends":
                symbol_data["extends"] = item[1]

        return symbol_name, symbol_data

    def parse_initial(self):
        """
        First pass: Parses the file and stores raw, hierarchical data.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            sexp_data = sexpdata.load(f)

        if not isinstance(sexp_data, list) or not sexp_data or not isinstance(sexp_data[0], sexpdata.Symbol) or sexp_data[0].value() != "kicad_symbol_lib":
            raise ValueError("Invalid KiCad symbol library format.")

        for item in sexp_data[1:]:
            if isinstance(item, list) and item and isinstance(item[0], sexpdata.Symbol) and item[0].value() == "symbol":
                symbol_name, symbol_data = self._parse_symbol_def(item)
                self.raw_symbols_data[symbol_name] = symbol_data

    def _transform_point(self, x, y, angle_deg, tx, ty):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Rotate
        rot_x = x * cos_a - y * sin_a
        rot_y = x * sin_a + y * cos_a

        # Translate
        new_x = rot_x + tx
        new_y = rot_y + ty

        return new_x, new_y

    def _transform_pin(self, pin, tx, ty, angle):
        pin_at_str = pin.get("at", "0 0 0")
        pin_at = [float(v) for v in pin_at_str.split()]
        px, py, p_angle = pin_at[0], pin_at[1], pin_at[2]

        new_px, new_py = self._transform_point(px, py, angle, tx, ty)
        new_p_angle = (p_angle + angle) % 360

        transformed_pin = pin.copy()
        transformed_pin["at"] = f"{new_px:.3f} {new_py:.3f} {new_p_angle:.0f}"
        return transformed_pin

    def _transform_rect(self, rect, tx, ty, angle):
        transformed_rect = rect.copy()
        if "start" in rect and "end" in rect:
            start_xy = [float(v) for v in rect["start"].split()]
            end_xy = [float(v) for v in rect["end"].split()]

            new_start_x, new_start_y = self._transform_point(start_xy[0], start_xy[1], angle, tx, ty)
            new_end_x, new_end_y = self._transform_point(end_xy[0], end_xy[1], angle, tx, ty)

            transformed_rect["start"] = f"{new_start_x:.3f} {new_start_y:.3f}"
            transformed_rect["end"] = f"{new_end_x:.3f} {new_end_y:.3f}"

        return transformed_rect

    def _get_flattened_symbol(self, symbol_name, visited=None):
        if visited is None:
            visited = set()

        if symbol_name in visited:
            raise Exception(f"Circular dependency detected in symbol hierarchy for {symbol_name}")
        visited.add(symbol_name)

        if symbol_name in self.flat_symbols_data:
            visited.remove(symbol_name)
            return self.flat_symbols_data[symbol_name]

        raw_data = self.raw_symbols_data.get(symbol_name)
        if not raw_data:
            # This can happen for symbols from other libs, e.g. "power:GND"
            # We can't resolve them, so we return an empty symbol.
            print(f"Warning: Symbol '{symbol_name}' not found in the library.")
            visited.remove(symbol_name)
            return {"properties": {}, "pins": [], "rects": []}

        flat_data = {"properties": raw_data["properties"].copy(), "pins": raw_data["pins"].copy(), "rects": raw_data["rects"].copy()}

        # Merge inline children from the raw data
        for child_data in raw_data.get("inline_children", {}).values():
            flat_data["pins"].extend(child_data["pins"])
            flat_data["rects"].extend(child_data["rects"])

        # 1. Handle 'extends' inheritance
        parent_name = raw_data.get("extends")
        if parent_name:
            parent_flat_data = self._get_flattened_symbol(parent_name, visited)

            merged_properties = parent_flat_data["properties"].copy()
            merged_properties.update(flat_data["properties"])
            flat_data["properties"] = merged_properties

            flat_data["pins"] = parent_flat_data["pins"] + flat_data["pins"]
            flat_data["rects"] = parent_flat_data["rects"] + flat_data["rects"]

        # 2. Handle children (composition)
        for child_instance in raw_data.get("children", []):
            child_lib_id = child_instance["lib_id"]
            child_flat_data = self._get_flattened_symbol(child_lib_id, visited)

            at_str = child_instance.get("at", "0 0 0")
            at = [float(v) for v in at_str.split()]
            tx, ty, angle = at[0], at[1], at[2]

            for pin in child_flat_data["pins"]:
                transformed_pin = self._transform_pin(pin, tx, ty, angle)
                flat_data["pins"].append(transformed_pin)

            for rect in child_flat_data["rects"]:
                transformed_rect = self._transform_rect(rect, tx, ty, angle)
                flat_data["rects"].append(transformed_rect)

        # De-duplicate pins by number
        unique_pins = {}
        for pin in flat_data["pins"]:
            pin_number = pin.get("number")
            if pin_number:
                unique_pins[pin_number] = pin
        flat_data["pins"] = list(unique_pins.values())

        visited.remove(symbol_name)
        self.flat_symbols_data[symbol_name] = flat_data
        return flat_data

    def flatten_and_inherit(self):
        """
        Second pass: Flattens the data and applies inheritance.
        """
        for symbol_name in self.raw_symbols_data:
            self._get_flattened_symbol(symbol_name)

    def get_flat_symbols(self):
        """
        Returns the final flattened and inherited symbol data.
        """
        return self.flat_symbols_data

    def pin_rect(self, at, length, thickness=2):
        x, y, angle = at
        rad = math.radians(angle)
        dx = math.cos(rad)
        dy = math.sin(rad)
        x1, y1 = x, y
        x2, y2 = x + dx * length, y + dy * length
        xmin = min(x1, x2) - thickness / 2
        xmax = max(x1, x2) + thickness / 2
        ymin = min(y1, y2) - thickness / 2
        ymax = max(y1, y2) + thickness / 2
        return f"RECT {xmin:.3f} {ymin:.3f} {xmax:.3f} {ymax:.3f} ;"

    def get_pin_bbox(self, pin):
        try:
            at = [float(x) for x in str(pin["at"]).split()]
            length = float(pin["length"])
            thickness = 0.14  # default from pin_rect
        except (KeyError, ValueError):
            return None

        x, y, angle = at
        rad = math.radians(angle)
        dx = math.cos(rad)
        dy = math.sin(rad)
        x1, y1 = x, y
        x2, y2 = x + dx * length, y + dy * length
        xmin = min(x1, x2) - thickness / 2
        xmax = max(x1, x2) + thickness / 2
        ymin = min(y1, y2) - thickness / 2
        ymax = max(y1, y2) + thickness / 2
        return xmin, ymin, xmax, ymax

    def get_rect_bbox(self, rect):
        try:
            start = [float(x) for x in rect["start"].split()]
            end = [float(x) for x in rect["end"].split()]
            # KiCad rectangles can have start/end in any order.
            xmin = min(start[0], end[0])
            ymin = min(start[1], end[1])
            xmax = max(start[0], end[0])
            ymax = max(start[1], end[1])
            return xmin, ymin, xmax, ymax
        except (KeyError, ValueError):
            return None

    def convert_to_lef(self, root):
        """Convert entire symbol library dict into one LEF string."""

        lef = ["VERSION 5.6 ;", 'BUSBITCHARS "[]" ;', 'DIVIDERCHAR "/" ;', ""]

        for name, node in root.items():
            # --- Bounding box calculation ---
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")

            for pin in node.get("pins", []):
                bbox = self.get_pin_bbox(pin)
                if bbox:
                    min_x = min(min_x, bbox[0])
                    min_y = min(min_y, bbox[1])
                    max_x = max(max_x, bbox[2])
                    max_y = max(max_y, bbox[3])

            for rect in node.get("rects", []):
                bbox = self.get_rect_bbox(rect)
                if bbox:
                    min_x = min(min_x, bbox[0])
                    min_y = min(min_y, bbox[1])
                    max_x = max(max_x, bbox[2])
                    max_y = max(max_y, bbox[3])

            origin_x, origin_y = 0.0, 0.0
            if any(math.isinf(v) for v in [min_x, min_y, max_x, max_y]):
                # No geometry found, use default size
                width, height = 999, 999
                obs_rect_str = ""
            else:
                origin_x, origin_y = min_x, min_y
                width = max_x - origin_x
                height = max_y - origin_y
                obs_rect_str = f"    RECT {min_x:.3f} {min_y:.3f} {max_x:.3f} {max_y:.3f} ;"

            props = node.get("properties", {})
            value = props.get("Value", name)
            if value and value[0].isdigit():
                value = "NUM_" + value

            lef.extend(
                [
                    f"MACRO {value}",
                    "  CLASS BLOCK ;",
                    f"  FOREIGN {value} 0 0 ;",
                    # TODO: explain negative origin
                    f"  ORIGIN {-origin_x:.3f} {-origin_y:.3f} ;",
                    "  SYMMETRY X Y R90 ;",
                    f"  SIZE {width:.3f} BY {height:.3f} ;",
                ]
            )

            if obs_rect_str:
                lef.extend(
                    [
                        "  OBS",
                        "    LAYER metal1 ;",
                        obs_rect_str,
                        "    LAYER metal2 ;",
                        obs_rect_str,
                        "    LAYER metal3 ;",
                        obs_rect_str,
                        "    LAYER metal4 ;",
                        obs_rect_str,
                        "    LAYER metal5 ;",
                        obs_rect_str,
                        "  END",
                    ]
                )

            for pin in node.get("pins", []):
                pname = str(pin.get("name", "PIN")).split()[0]
                pnum = str(pin.get("number", "PIN")).split()[0]
                try:
                    at = [float(x) for x in str(pin["at"]).split()]
                    length = float(pin["length"])
                except Exception:
                    continue

                rect = self.pin_rect(at, length)

                lef.extend(
                    [
                        f"  # {pname}",
                        f"  PIN PIN{pnum}",
                        "    PORT",
                        "      LAYER metal3 ;",
                        f"        {rect}",
                        "      LAYER metal4 ;",
                        f"        {rect}",
                        "      LAYER metal5 ;",
                        f"        {rect}",
                        "    END",
                        f"  END PIN{pnum}",
                    ]
                )

            lef.append(f"END {value}")
            lef.append("")

        return "\n".join(lef)


def main():
    input_dir = "symbols"
    for input_file in glob.glob(os.path.join(input_dir, "*.kicad_sym")):
        basename = os.path.splitext(os.path.basename(input_file))[0]

        parser = KiCadSymbolLibParser(input_file)
        parser.parse_initial()
        parser.flatten_and_inherit()
        symbols = parser.get_flat_symbols()

        # save json to dir for debug
        output_file = f"json/{basename}.json"
        with open(output_file, "w") as f:
            json.dump(symbols, f, indent=2)

        lef = parser.convert_to_lef(symbols)
        output_file = f"lef/{basename}.lef"
        with open(output_file, "w") as f:
            f.write(lef)


if __name__ == "__main__":
    main()
