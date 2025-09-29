import os


class SymbolLibrary:
    """Generate LTspice symbol definitions for standard components (res, cap, ind, dio, led)."""

    def __init__(self):
        self.seen = {}
        self.symbols = {
            "res": """Version 4
SymbolType CELL
LINE Normal 0 -48 0 -32
LINE Normal 0 -32 -8 -24
LINE Normal -8 -24 8 -8
LINE Normal 8 -8 -8 8
LINE Normal -8 8 8 24
LINE Normal 8 24 0 32
LINE Normal 0 32 0 48
WINDOW 0 12 -40 Left 2
WINDOW 3 12 40 Left 2
SYMATTR Value R
SYMATTR Prefix R
SYMATTR Description A resistor
PIN 0 -48 BOTTOM 0
PINATTR PinName 1
PINATTR SpiceOrder 1
PIN 0 48 TOP 0
PINATTR PinName 2
PINATTR SpiceOrder 2
""",
            "cap": """Version 4
SymbolType CELL
LINE Normal 0 -48 0 -16
LINE Normal 0 16 0 48
LINE Normal -12 -8 12 -8
LINE Normal -12 8 12 8
WINDOW 0 12 -40 Left 2
WINDOW 3 12 40 Left 2
SYMATTR Value C
SYMATTR Prefix C
SYMATTR Description A capacitor
PIN 0 -48 BOTTOM 0
PINATTR PinName 1
PINATTR SpiceOrder 1
PIN 0 48 TOP 0
PINATTR PinName 2
PINATTR SpiceOrder 2
""",
            "ind": """Version 4
SymbolType CELL
LINE Normal 0 -48 0 -24 1
ARC Normal -8 -24 8 -8 8 -24 8 -8 1
ARC Normal -8 -8 8 8 8 -8 8 8 1
ARC Normal -8 8 8 24 8 8 8 24 1
LINE Normal 0 24 0 48 1
WINDOW 0 12 -40 Left 2
WINDOW 3 12 40 Left 2
SYMATTR Value L
SYMATTR Prefix L
SYMATTR Description An inductor
PIN 0 -48 BOTTOM 0
PINATTR PinName 1
PINATTR SpiceOrder 1
PIN 0 48 TOP 0
PINATTR PinName 2
PINATTR SpiceOrder 2
""",
            "dio": """Version 4
SymbolType CELL
LINE Normal 0 -48 0 -24
LINE Normal 0 -24 0 24
LINE Normal 0 24 0 48
LINE Normal -12 0 12 0
LINE Normal -12 -16 0 0
LINE Normal -12 16 0 0
WINDOW 0 12 -40 Left 2
WINDOW 3 12 40 Left 2
SYMATTR Value D
SYMATTR Prefix D
SYMATTR Description A diode
PIN 0 -48 BOTTOM 0
PINATTR PinName 1
PINATTR SpiceOrder 1
PIN 0 48 TOP 0
PINATTR PinName 2
PINATTR SpiceOrder 2
""",
            "led": """Version 4
SymbolType CELL
LINE Normal 0 -48 0 -24
LINE Normal 0 -24 0 24
LINE Normal 0 24 0 48
LINE Normal -12 0 12 0
LINE Normal -12 -16 0 0
LINE Normal -12 16 0 0
LINE Normal 4 16 10 22
LINE Normal 4 8 10 14
LINE Normal 10 22 6 16
LINE Normal 10 14 6 8
WINDOW 0 12 -40 Left 2
WINDOW 3 12 40 Left 2
SYMATTR Value LED
SYMATTR Prefix D
SYMATTR Description A light-emitting diode
PIN 0 -48 BOTTOM 0
PINATTR PinName 1
PINATTR SpiceOrder 1
PIN 0 48 TOP 0
PINATTR PinName 2
PINATTR SpiceOrder 2
""",
        }

    def get_symbol(self, key: str) -> str:
        """Return full .asy text for symbol key ('res', 'cap', 'ind', 'dio', 'led')."""
        key = key.lower()
        if key not in self.symbols:
            raise ValueError(f"Unknown symbol type: {key}")
        return self.symbols[key]

    def generate_symbol_asy(self, key: str, output_dir="data/ltspice"):
        """Write symbol definition to a .asy file."""
        if key in self.seen:
            return
        self.seen[key] = True
        filepath = os.path.join(output_dir, f"{key}.asy")
        with open(filepath, "w") as f:
            f.write(self.get_symbol(key))
