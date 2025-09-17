from dataclasses import dataclass, field
from typing import List, Tuple

from schematic_from_netlist.interfaces.netlist_database import Instance, Net, Pin


@dataclass
class NetShape:
    net: Net
    points: List[Tuple[float, float]]


@dataclass
class PortShape:
    pin: Pin
    rect: Tuple[float, float, float, float]


@dataclass
class InstShape:
    inst: Instance
    rect: Tuple[float, float, float, float]
    port_shapes: List[PortShape]


class CreateSchematic:
    def __init__(self, geom_db, netlist_db):
        self.geom_db = geom_db
        self.netlist_db = netlist_db

        self.port_shapes = []
        self.inst_shapes = []
        self.net_shapes = []

    def find_net_between_inst(self, list_of_inst):
        """
        Find nets that are common to all instances in list_of_inst.
        Each inst must have .get_connected_nets() -> List[Net].
        """
        if not list_of_inst:
            return []

        # Get nets of the first inst
        common_nets = {net.name: net for net in list_of_inst[0].get_connected_nets()}

        # Intersect with the rest
        for inst in list_of_inst[1:]:
            inst_nets = {net.name: net for net in inst.get_connected_nets()}
            # keep only common nets
            common_nets = {n: net for n, net in common_nets.items() if n in inst_nets}

        # Return list of Net objects
        return list(common_nets.values())

    def associate_ports_to_blocks(self):
        # look over all port pairs and compare them to

        for port in self.geom_db.ports:
            inst1 = self.netlist_db.inst_by_name[port.name]
            inst2 = self.netlist_db.inst_by_name[port.conn]

            # find a net connecting these 2 inst
            match_nets = self.find_net_between_inst([inst1, inst2])

            # if multiple net connections between the same 2 inst, we have to assign one port pair
            # to each connection
            for net in match_nets:
                for pin in net.pins:
                    if pin.instance.name == inst1.name:
                        ps = PortShape(pin, port.rect)
                        self.port_shapes.append(ps)

    def find_outline_of_block(self):
        instname2shapes = {}
        for shape in self.port_shapes:
            pin = shape.pin
            if pin.instance.name not in instname2shapes:
                instname2shapes[pin.instance.name] = []
            instname2shapes[pin.instance.name].append(shape)

        # ok now for each inst look at all the geom
        rects = {}
        for instname, shapes in instname2shapes.items():
            if instname not in rects:
                rects[instname] = []
            for shape in shapes:
                rect = shape.rect
                rects[instname].append(rect)

        for base_instname, rect_list in rects.items():
            x_min = min(rect[0] for rect in rect_list)
            y_min = min(rect[1] for rect in rect_list)
            x_max = max(rect[0] for rect in rect_list)
            y_max = max(rect[1] for rect in rect_list)
            rect = (x_min, y_min, x_max, y_max)

            full_instname = self.netlist_db.top_module.name + "/" + base_instname
            port_shapes = instname2shapes[instname]
            inst = self.netlist_db.inst_by_name[full_instname]
            inst_shape = InstShape(inst=inst, rect=rect, port_shapes=port_shapes)
            self.inst_shapes.append(inst_shape)

    def find_net_shapes(self):
        for net_geom in self.geom_db.nets:
            list_of_inst = []
            for instname in net_geom.conn:
                list_of_inst.append(self.netlist_db.inst_by_name[instname])

            nets = self.find_net_between_inst(list_of_inst)
            if nets:
                net_shape = NetShape(net=nets[0], points=net_geom.points)
                self.net_shapes.append(net_shape)

    def generate_schematic_info(self):
        self.associate_ports_to_blocks()
        self.find_outline_of_block()
        self.find_net_shapes()
        return self

    def patch_and_remove_buffers(self):
        """
        Finds buffer instances, replaces them with a wire in the geometric data,
        and then removes them from the logical netlist.
        """
        buffers_to_remove = []
        new_wires = []

        # Find buffer instance shapes
        for inst_shape in self.inst_shapes:
            if inst_shape.inst.name.startswith("buf_"):
                buffers_to_remove.append(inst_shape)
                
                input_port = None
                output_port = None
                for port_shape in inst_shape.port_shapes:
                    if port_shape.pin.name == 'I':
                        input_port = port_shape
                    elif port_shape.pin.name == 'O':
                        output_port = port_shape

                if input_port and output_port:
                    # Create a new wire to "patch" the connection
                    in_rect = input_port.rect
                    out_rect = output_port.rect
                    
                    # Get center points of the port rectangles
                    p1 = ((in_rect[0] + in_rect[2]) / 2, (in_rect[1] + in_rect[3]) / 2)
                    p2 = ((out_rect[0] + out_rect[2]) / 2, (out_rect[1] + out_rect[3]) / 2)
                    
                    # The net can be None as it's just for drawing
                    new_wires.append(NetShape(net=None, points=[p1, p2]))

        # Remove buffer shapes from the list
        self.inst_shapes = [s for s in self.inst_shapes if s not in buffers_to_remove]
        
        # Add the new patch wires
        self.net_shapes.extend(new_wires)

        # Remove buffers from the logical netlist
        self.netlist_db.remove_buffers()
        print(f"Patched and removed {len(buffers_to_remove)} buffer instances.")
