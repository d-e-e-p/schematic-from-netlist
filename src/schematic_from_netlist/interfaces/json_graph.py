import json
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class GeomPort:
    """Represents the geometric location of a port on an instance."""
    name: str  # Head instance name
    conn: str  # Tail instance name
    rect: Tuple[float, float, float, float]

@dataclass
class GeomNet:
    """Represents the geometric path of a wire."""
    conn: List[str]
    points: List[Tuple[float, float]]

@dataclass
class GeomDB:
    """A database for geometric primitives extracted from the graph layout."""
    ports: List[GeomPort] = field(default_factory=list)
    nets: List[GeomNet] = field(default_factory=list)

class ParseJson:
    def __init__(self, json_file):
        self.json_file = json_file
        with open(self.json_file, "r") as f:
            self.json_data = json.load(f)

    def parse(self) -> GeomDB:
        """Parses the JSON file and populates the geometric database."""
        geom_db = GeomDB()
        objects = self.json_data.get("objects", [])

        # Create a mapping from gvid to object name for quick lookup
        gvid_to_name = {i: obj.get("name") for i, obj in enumerate(objects)}

        for edge in self.json_data.get("edges", []):
            try:
                head_gvid = edge.get("head")
                tail_gvid = edge.get("tail")
                
                head_name = gvid_to_name.get(head_gvid)
                tail_name = gvid_to_name.get(tail_gvid)

                if not head_name or not tail_name:
                    continue

                # Extract port rectangle from the head of the edge
                head_rect_data = edge["_hdraw_"][-1]["rect"]
                x_h, y_h, w_h, h_h = head_rect_data
                head_port_rect = (x_h - w_h, y_h - h_h, x_h + w_h, y_h + h_h)
                geom_db.ports.append(GeomPort(name=head_name, conn=tail_name, rect=head_port_rect))

                # Extract port rectangle from the tail of the edge
                tail_rect_data = edge["_tdraw_"][-1]["rect"]
                x_t, y_t, w_t, h_t = tail_rect_data
                tail_port_rect = (x_t - w_t, y_t - h_t, x_t + w_t, y_t + h_t)
                geom_db.ports.append(GeomPort(name=tail_name, conn=head_name, rect=tail_port_rect))

                # Extract wire points
                points_data = edge["_draw_"][-1]["points"]
                wire_points = [(p[0], p[1]) for p in points_data]
                geom_db.nets.append(GeomNet(conn=[head_name, tail_name], points=wire_points))

            except (KeyError, IndexError, AttributeError, ValueError, TypeError):
                continue
        
        print(f"Parsed {len(geom_db.ports)} ports and {len(geom_db.nets)} nets from JSON.")
        return geom_db
