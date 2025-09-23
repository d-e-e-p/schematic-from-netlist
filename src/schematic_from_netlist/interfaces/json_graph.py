import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GeomDB:
    """A database for geometric primitives extracted from the graph layout."""

    ports: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    nets: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    def write_geom_db_report(self, filepath: str = "data/json/read_json.rpt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        out = "PORTS\n"
        for name in sorted(self.ports.keys()):
            port = self.ports[name]
            out += f" {name:20}: {port}\n"

        out += "NETS\n"
        for name in sorted(self.nets.keys()):
            pts = self.nets[name]
            out += f" {name:20}: {pts}\n"

        with open(filepath, "w") as f:
            f.write(out)


class ParseJson:
    def __init__(self, json_file):
        self.json_file = json_file
        with open(self.json_file, "r") as f:
            self.json_data = json.load(f)

    def parse(self) -> GeomDB:
        """Parses the JSON file and populates geomdb."""
        geom_db = GeomDB()

        for edge in self.json_data.get("edges", []):
            try:
                # Extract port rectangle from the head of the edge
                rect_data = edge["_hdraw_"][-1]["rect"]
                text_data = edge["headlabel"]
                x, y, _, _ = rect_data
                geom_db.ports[text_data] = (x, y)

                # Extract port rectangle from the tail of the edge
                rect_data = edge["_tdraw_"][-1]["rect"]
                text_data = edge["taillabel"]
                x, y, _, _ = rect_data
                geom_db.ports[text_data] = (x, y)

                # Extract wire points
                text_data = edge["label"]
                points_data = edge["_draw_"][-1]["points"]
                geom_db.nets[text_data] = points_data

            except (KeyError, IndexError, AttributeError, ValueError, TypeError):
                continue

        print(f"Parsed {len(geom_db.ports)} ports and {len(geom_db.nets)} nets from graph.")
        geom_db.write_geom_db_report()
        return geom_db
