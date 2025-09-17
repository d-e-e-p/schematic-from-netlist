"""
Allows the package to be run as a script.
Example: python -m schematic_from_netlist data/verilog/primitive.v
"""
from .cli import main

if __name__ == "__main__":
    main()
