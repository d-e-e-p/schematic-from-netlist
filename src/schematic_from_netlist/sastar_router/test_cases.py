from models import Net
from shapely.geometry import MultiLineString, box


def create_hard_test_case(difficulty="hard"):
    """
    Create test cases of varying difficulty

    difficulty: "easy", "hard", "extreme", "maze", "precision"
    """
    if difficulty == "easy":
        nets = [
            Net("net1", [(-10, -2), (15, 10)]),
            Net("net2", [(0, 0), (20, 5), (25, 25)]),
        ]
        obstacles = [
            box(5, 2, 15, 5),
            box(10, 5, 20, 10),
        ]

    elif difficulty == "hard":
        nets = [
            Net("net1", [(5, 2), (15, 5)]),
            Net("net2", [(10, 5), (20, 10), (20, 15)]),
            Net("net3", [(15, 2), (10, 10)]),
            Net("net4", [(5, 5), (10, 10), (25, 15)]),
        ]
        obstacles = [
            box(5, 2, 15, 5),
            box(10, 5, 20, 10),
            box(18, 8, 25, 15),
            box(12, 12, 18, 18),
        ]

    elif difficulty == "extreme":
        nets = [
            Net("net1", [(5, 2), (15, 5)]),
            Net("net2", [(15, 2), (10, 10), (25, 15)]),
            Net("net3", [(15, 5), (18, 8)]),
            Net("net4", [(5, 5), (20, 8), (12, 18)]),
            Net("net5", [(10, 5), (10, 10)]),
            Net("net6", [(5, 2), (25, 8), (18, 18), (5, 12)]),
        ]
        obstacles = [
            box(5, 2, 15, 5),
            box(10, 5, 20, 10),
            box(18, 8, 25, 15),
            box(12, 12, 18, 18),
            box(6, 8, 9, 11),
            box(22, 2, 28, 5),
        ]

    elif difficulty == "maze":
        nets = [
            Net("net1", [(2, 2), (28, 18)]),
            Net("net2", [(2, 8), (15, 2), (28, 12)]),
            Net("net3", [(10, 5), (20, 5)]),
        ]
        obstacles = [
            box(5, 0, 7, 10),
            box(10, 5, 12, 20),
            box(15, 0, 17, 8),
            box(15, 12, 17, 20),
            box(20, 3, 22, 15),
            box(25, 0, 27, 12),
            box(0, 5, 10, 7),
            box(12, 10, 22, 12),
            box(7, 15, 20, 17),
        ]

    elif difficulty == "precision":
        obstacles = [
            box(5, 2, 15, 5),
            box(10, 5, 20, 10),
            box(15, 2, 25, 8),
        ]

        nets = [
            Net("net1", [(5, 2), (15, 5)], MultiLineString([[(5, 2), (5, -2)], [(5, -2), (15, -2)], [(15, -2), (15, 5)]])),
            Net(
                "net2",
                [(10, 5), (20, 10)],
                MultiLineString([[(10, 5), (9, 5)], [(9, 5), (9, 14)], [(9, 14), (20, 14)], [(20, 14), (20, 10)]]),
            ),
            Net(
                "net3",
                [(15, 5), (20, 10)],
                MultiLineString(
                    [
                        [(15, 5), (15, 1)],
                        [(15, 1), (29, 1)],
                        [(29, 1), (29, 11)],
                        [(29, 11), (21, 11)],
                        [(21, 11), (21, 10)],
                        [(21, 10), (20, 10)],
                    ]
                ),
            ),
        ]
    elif difficulty == "macro_grid":
        """
        Regular grid of macros (5x5um blocks) in 1000x1000 space
        Each macro has 2 pins on opposite faces (middle of edges)
        Nets connect adjacent macros in a mesh pattern
        """
        macro_size = 5
        macro_spacing = 8  # Total space per macro (5 + 3 gap)
        num_macros_per_side = 10  # 10x10 grid fits in ~100x100, scale up

        # For 1000x1000 span, use larger macros/spacing
        macro_size = 50
        macro_spacing = 100  # 50um macro + 50um routing space
        num_macros_per_side = 9  # 9x9 grid = 900x900 with 50 margin

        obstacles = []
        nets = []
        net_id = 1

        # Create grid of macros
        macro_pins = {}  # (row, col) -> {left, right, top, bottom} pin coordinates

        for row in range(num_macros_per_side):
            for col in range(num_macros_per_side):
                # Macro position (bottom-left corner)
                x = 50 + col * macro_spacing
                y = 50 + row * macro_spacing

                # Create macro obstacle
                obstacles.append(box(x, y, x + macro_size, y + macro_size))

                # Pins on middle of each face
                pins = {
                    "left": (x, y + macro_size / 2),
                    "right": (x + macro_size, y + macro_size / 2),
                    "top": (x + macro_size / 2, y + macro_size),
                    "bottom": (x + macro_size / 2, y),
                }
                macro_pins[(row, col)] = pins

        # Create interconnect nets
        # Horizontal connections (left-right pins)
        for row in range(num_macros_per_side):
            for col in range(num_macros_per_side - 1):
                pin1 = macro_pins[(row, col)]["right"]
                pin2 = macro_pins[(row, col + 1)]["left"]
                nets.append(Net(f"net_h_{row}_{col}", [pin1, pin2]))

        # Vertical connections (bottom-top pins)
        for row in range(num_macros_per_side - 1):
            for col in range(num_macros_per_side):
                pin1 = macro_pins[(row, col)]["top"]
                pin2 = macro_pins[(row + 1, col)]["bottom"]
                nets.append(
                    Net(
                        f"net_v_{row}_{col}",
                        [pin1, pin2],
                    )
                )

        # Diagonal long-distance nets (stress test)
        for i in range(0, num_macros_per_side, 3):
            for j in range(0, num_macros_per_side, 3):
                if i + 2 < num_macros_per_side and j + 2 < num_macros_per_side:
                    pin1 = macro_pins[(i, j)]["right"]
                    pin2 = macro_pins[(i + 2, j + 2)]["left"]
                    nets.append(
                        Net(
                            f"net_diag_{i}_{j}",
                            [pin1, pin2],
                        )
                    )

    return nets, obstacles
