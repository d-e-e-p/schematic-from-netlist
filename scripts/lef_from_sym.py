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

    def _parse_pins(self, pin_list):
        """Recursively extracts pin information from a nested list."""
        pins = []
        if isinstance(pin_list, list):
            for item in pin_list:
                if isinstance(item, list) and isinstance(item[0], sexpdata.Symbol) and item[0].value() == "pin":
                    pin_info = {}
                    for sub_item in item[1:]:
                        if isinstance(sub_item, list):
                            key = sub_item[0].value()
                            value = " ".join(str(val) for val in sub_item[1:])
                            pin_info[key] = value
                    pins.append(pin_info)
                elif isinstance(item, list) and isinstance(item[0], sexpdata.Symbol) and item[0].value() == "symbol":
                    # Recurse into child symbols to find pins
                    pins.extend(self._parse_pins(item))
        return pins

    def parse_sexp_list(self, s_expr_list):
        """
        Recursively parses a list from an s-expression into a dictionary.
        """
        if not isinstance(s_expr_list, list):
            return s_expr_list

        print(f"{s_expr_list=}")
        result = {}
        key = None
        for item in s_expr_list:
            print(f"{item=}")
            if isinstance(item, sexpdata.Symbol):
                key = item.value()
                print(f"{key=}")
            elif key:
                if isinstance(item, list):
                    result[key] = self.parse_sexp_list(item)
                else:
                    result[key] = item
                key = None
        print(f"{result=}")
        return result

    def _parse_rects(self, rect_list):
        rects = []
        print(f"rect_list: {rect_list}")
        if isinstance(rect_list, list):
            for item in rect_list:
                if isinstance(item, list) and isinstance(item[0], sexpdata.Symbol) and item[0].value() == "rectangle":
                    rect_info = self.parse_sexp_list(item[1:])
                elif isinstance(item, list) and isinstance(item[0], sexpdata.Symbol) and item[0].value() == "symbol":
                    # Recurse into child symbols to find pins
                    rect_list.extend(self._parse_rects(item))
        return rects

    def _parse_properties(self, prop_list):
        """Extracts properties from a nested list."""
        properties = {}
        for item in prop_list:
            if isinstance(item, list) and isinstance(item[0], sexpdata.Symbol) and item[0].value() == "property":
                key = item[1]
                value = item[2]
                properties[key] = value
        return properties

    def parse_initial(self):
        """
        First pass: Parses the file and stores raw, hierarchical data.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            sexp_data = sexpdata.load(f)

        if not isinstance(sexp_data, list) or not isinstance(sexp_data[0], sexpdata.Symbol) or sexp_data[0].value() != "kicad_symbol_lib":
            raise ValueError("Invalid KiCad symbol library format.")

        for item in sexp_data[1:]:
            if isinstance(item, list) and isinstance(item[0], sexpdata.Symbol) and item[0].value() == "symbol":
                symbol_name = item[1]

                extends_parent = None
                for sub_item in item:
                    if isinstance(sub_item, list) and isinstance(sub_item[0], sexpdata.Symbol) and sub_item[0].value() == "extends":
                        extends_parent = sub_item[1]
                        break

                properties = self._parse_properties(item)
                pins = self._parse_pins(item)
                rects = self._parse_rects(item)

                self.raw_symbols_data[symbol_name] = {"properties": properties, "pins": pins, "extends": extends_parent}

    def flatten_and_inherit(self):
        """
        Second pass: Flattens the data and applies inheritance.
        """
        for symbol_name, raw_data in self.raw_symbols_data.items():
            parent_name = raw_data.get("extends")

            flat_data = raw_data.copy()

            if parent_name and parent_name in self.raw_symbols_data:
                parent_data = self.raw_symbols_data[parent_name]

                new_properties = parent_data["properties"].copy()
                new_properties.update(raw_data["properties"])
                flat_data["properties"] = new_properties

                flat_data["pins"] = parent_data["pins"]

            flat_data.pop("extends", None)

            self.flat_symbols_data[symbol_name] = flat_data

    def get_flat_symbols(self):
        """
        Returns the final flattened and inherited symbol data.
        """
        return self.flat_symbols_data

    def pin_rect(self, at, length, thickness=0.14):
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

    def convert_to_lef(self, root):
        """Convert entire symbol library dict into one LEF string."""

        lef = ["VERSION 5.6 ;", 'BUSBITCHARS "[]" ;', 'DIVIDERCHAR "/" ;', ""]

        for name, node in root.items():
            props = node.get("properties", {})
            value = props.get("Value", name)
            # TODO: derive real size from footprint?
            width, height = 100, 430

            lef.extend(
                [
                    f"MACRO {value}",
                    "  CLASS BLOCK ;",
                    f"  FOREIGN {value} 0 0 ;",
                    "  ORIGIN 0 0 ;",
                    "  SYMMETRY X Y ;",
                    f"  SIZE {width} BY {height} ;",
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
                        "    DIRECTION INOUT ;",  # TODO: map to actual dir
                        "    PORT",
                        "      LAYER metal3 ;",
                        f"        {rect}",
                        "    END",
                        f"  END PIN{pnum}",
                    ]
                )

            lef.append(f"END {value}")
            lef.append("")

        return "\n".join(lef)


def main():
    input_dir = "test_symbols"
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
        output_file = f"test_lef/{basename}.lef"
        with open(output_file, "w") as f:
            f.write(lef)


if __name__ == "__main__":
    main()
