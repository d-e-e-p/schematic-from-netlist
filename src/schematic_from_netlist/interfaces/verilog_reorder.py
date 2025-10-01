import logging
import os
import re
from collections import defaultdict, deque
from pathlib import Path

import pyslang

log = logging.getLogger(__name__)


class ModuleInfo:
    """Store module information from AST"""

    def __init__(self, name, syntax_node):
        self.name = name
        self.syntax_node = syntax_node
        self.dependencies = set()
        self.instances = []  # (instance_name, module_type, port_connections)


class VerilogReorder:
    """Reorder modules using pyslang syntax tree (AST) before compilation"""

    def __init__(self):
        self.modules = {}  # module_name -> ModuleInfo
        self.syntax_tree = None

    def parse_syntax_tree(self, filename):
        """Parse file to get syntax tree (no compilation yet)"""
        log.info(f"Parsing syntax tree from: {filename}")

        # Parse to syntax tree only (no semantic analysis)
        self.syntax_tree = pyslang.SyntaxTree.fromFile(filename)

        # Check for syntax errors
        if self.syntax_tree.diagnostics:
            for diag in self.syntax_tree.diagnostics:
                if diag.isError():
                    log.error(f"Syntax error: {diag.code} {diag.args} at {diag.location}")
                    raise RuntimeError("Syntax errors found")

        # Walk the syntax tree to find modules
        self._walk_syntax_tree(self.syntax_tree.root)

        log.info(f"Found {len(self.modules)} modules")

        # Find instantiations in each module
        for module_name, module_info in self.modules.items():
            self._find_instantiations(module_info)

    def _walk_syntax_tree(self, node):
        """Recursively walk syntax tree to find module declarations"""
        if node is None:
            return

        # Check if this is a module declaration
        if node.kind == pyslang.SyntaxKind.ModuleDeclaration:
            module_name = self._get_module_name(node)
            log.info(f"found module with {module_name=}")
            if module_name:
                self.modules[module_name] = ModuleInfo(module_name, node)
                log.debug(f"Found module: {module_name}")

        # Recurse through children - handle different node types
        children = self._get_children(node)
        for child in children:
            self._walk_syntax_tree(child)

    def _get_children(self, node):
        """Get child nodes handling different syntax node types"""
        # CompilationUnit has 'members' not 'childNodes'
        if hasattr(node, "members"):
            return node.members
        # Most other nodes have childNodes()
        elif hasattr(node, "childNodes"):
            return node.childNodes()
        # Some nodes are iterable directly
        elif hasattr(node, "__iter__") and not isinstance(node, str):
            try:
                return list(node)
            except:
                return []
        return []

    def _get_module_name(self, module_node):
        """Extract module name from module declaration node"""
        # Look for ModuleHeader -> NamedBlockClause -> Identifier
        return module_node.header.name.value

    def _find_instantiations(self, module_info):
        """Find all instantiations within a module using AST"""
        log.debug(f"Finding instantiations in module: {module_info.name}")
        self._walk_for_instances(module_info.syntax_node, module_info)

    def _walk_for_instances(self, node, module_info):
        """Walk module AST to find HierarchyInstantiation nodes"""
        if node is None:
            return

        # Found an instantiation
        if node.kind == pyslang.SyntaxKind.HierarchyInstantiation:
            module_type = self._get_instantiated_type(node)
            instances = self._get_instance_list(node)
            log.info(f"Found instantiation: {module_type=} {instances=}")

            for inst_name, port_connections in instances:
                module_info.instances.append(
                    {"instance_name": inst_name, "module_type": module_type, "connections": port_connections}
                )
                module_info.dependencies.add(module_type)
                log.debug(f"  Found: {module_type} {inst_name}")
                log.debug(f"    Ports: {list(port_connections.keys())}")

        # Recurse
        children = self._get_children(node)
        for child in children:
            self._walk_for_instances(child, module_info)

    def _get_instantiated_type(self, hier_inst_node):
        """
        Get the module type being instantiated.
        HierarchyInstantiation nodes often have a `.type` token with .valueText;
        fallback: search children for NamedType / Identifier.
        """
        # Common case: direct token
        try:
            t = getattr(hier_inst_node, "type", None)
            if t:
                # valueText works on token-like objects
                return getattr(t, "valueText", getattr(t, "value", None))
        except Exception:
            pass

        # Fallback: search children for NamedType / Identifier
        for child in self._get_children(hier_inst_node):
            if child.kind == pyslang.SyntaxKind.NamedType:
                # NamedType -> IdentifierName or QualifiedName etc.
                for g in self._get_children(child):
                    if hasattr(g, "valueText"):
                        return g.valueText
            # direct identifier token
            if hasattr(child, "valueText"):
                return child.valueText

        return None

    def _get_instance_name(self, inst_node):
        """Get instance name from HierarchicalInstance"""
        if hasattr(inst_node, "decl") and inst_node.decl:
            if hasattr(inst_node.decl, "name") and inst_node.decl.name:
                return inst_node.decl.name.value
        return None

    def _get_port_connections(self, inst_node):
        """Extract port connections from instance"""
        connections = {}

        # Look for SeparatedList
        children = self._get_children(inst_node)
        for child in children:
            if child.kind == pyslang.SyntaxKind.SeparatedList:
                conn_children = self._get_children(child)
                for conn_child in conn_children:
                    if conn_child.kind == pyslang.SyntaxKind.NamedPortConnection:
                        port_name, signal = self._parse_named_port_connection(conn_child)
                        if port_name:
                            connections[port_name] = signal

        return connections

    def _parse_named_port_connection(self, conn_node):
        """Parse .port_name(signal) connection"""
        port_name = None
        signal = None

        # Get port name from 'name' token
        if hasattr(conn_node, "name") and conn_node.name:
            port_name = conn_node.name.valueText

        # Get connected signal - simplified, just get text representation
        children = self._get_children(conn_node)
        for child in children:
            # Skip the dot and port name, get the expression
            if child.kind != pyslang.TokenKind.Dot:
                # Get source text for the expression
                signal = self._get_node_text(child)
                break

        return port_name, signal

    def _get_node_text(self, node):
        """Get source text from a syntax node"""
        if node is None:
            return ""
        # Get the text span and extract from source
        return str(node).strip()

    def topological_sort(self):
        """Sort modules in dependency order (bottom-up)"""
        in_degree = defaultdict(int)
        all_modules = set(self.modules.keys())

        # Build reverse dependency graph
        dependency_graph = defaultdict(set)
        for module_name, module_info in self.modules.items():
            for dep in module_info.dependencies:
                if dep in all_modules:
                    dependency_graph[dep].add(module_name)
                    in_degree[module_name] += 1

        # Initialize leaf modules
        for module in all_modules:
            if module not in in_degree:
                in_degree[module] = 0

        # Topological sort (Kahn's algorithm)
        queue = deque([m for m in all_modules if in_degree[m] == 0])
        sorted_modules = []

        while queue:
            module = queue.popleft()
            sorted_modules.append(module)

            for dependent in dependency_graph[module]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(sorted_modules) != len(all_modules):
            remaining = all_modules - set(sorted_modules)
            log.warning(f"Circular dependencies detected: {remaining}")
            sorted_modules.extend(remaining)

        return sorted_modules

    def find_missing_modules(self):
        """Find modules that are instantiated but not defined"""
        defined = set(self.modules.keys())
        instantiated = set()

        for module_info in self.modules.values():
            instantiated.update(module_info.dependencies)

        missing = instantiated - defined
        return missing

    def _sanitize_identifier(self, name):
        """Sanitize identifier to be valid SystemVerilog using escaped identifiers"""
        import re

        if not name:
            return "port_unnamed"

        # Check if name needs escaping (starts with digit or has special chars)
        needs_escape = (
            name[0].isdigit()  # Starts with digit
            or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_$]*", name)  # Contains special chars
        )

        if needs_escape:
            return f"\\{name}"
        else:
            return name

    def generate_stub_module(self, module_name):
        """Generate stub module text based on instantiations"""
        log.info(f"Generating stub for: {module_name}")

        # Collect all unique port names
        all_ports = set()

        for module_info in self.modules.values():
            for inst in module_info.instances:
                if inst["module_type"] == module_name:
                    all_ports.update(inst["connections"].keys())

        # Generate stub
        lines = [f"module {module_name} ("]

        port_list = sorted(all_ports)
        for i, port in enumerate(port_list):
            comma = "," if i < len(port_list) - 1 else ""
            safe_port = self._sanitize_identifier(port)
            lines.append(f"    input logic {safe_port} {comma}  // Auto-generated stub")

        lines.append(");")
        lines.append(f"    // TODO: Implement {module_name}")
        lines.append("endmodule")

        return "\n".join(lines)

    def _get_node_text_from_tree(self, node):
        """Return the source text for any syntax node using the stored SyntaxTree."""
        if node is None:
            return ""
        # Some nodes may not have a sourceRange (be defensive)
        sr = getattr(node, "sourceRange", None)
        if not sr:
            return ""
        sm = self.syntax_tree.sourceManager
        buf = sr.start.buffer
        full_text = sm.getSourceText(buf)
        start = sr.start.offset
        end = sr.end.offset
        return full_text[start:end]

    def _get_instance_list(self, hier_inst_node):
        """Get list of instances from HierarchyInstantiation"""
        instances = []
        for child in self._get_children(hier_inst_node):
            # Usually the list of instances is stored in a SeparatedList
            if child.kind == pyslang.SyntaxKind.SeparatedList:
                for item in self._get_children(child):
                    if item.kind == pyslang.SyntaxKind.HierarchicalInstance:
                        inst_name = self._get_instance_name(item)
                        log.info(f"  Found instance: {inst_name}")
                        port_connections = self._get_port_connections(item)
                        if inst_name:
                            instances.append((inst_name, port_connections))
        return instances

    def generate_reordered_file(self, output_file):
        """Generate reordered file with stubs for missing modules"""
        # Find missing modules
        missing = self.find_missing_modules()

        # Generate stubs
        stubs = {}
        for module_name in missing:
            stubs[module_name] = self.generate_stub_module(module_name)

        # Sort modules
        sorted_names = self.topological_sort()

        log.info(f"Module order: {' -> '.join(sorted_names)}")

        # Write output file
        with open(output_file, "w") as f:
            f.write("// Auto-reordered SystemVerilog using AST\n")
            f.write("// Module order: bottom-up (leaves first, top last)\n\n")

            # Write stubs first (they're leaves)
            if stubs:
                f.write("// ===== Auto-generated stub modules =====\n\n")
                for module_name in sorted(stubs.keys()):
                    f.write(stubs[module_name])
                    f.write("\n\n")

            # Write existing modules in sorted order
            f.write("// ===== Reordered existing modules =====\n\n")
            for module_name in sorted_names:
                if module_name in self.modules:
                    module_info = self.modules[module_name]
                    node = module_info.syntax_node

                    # Extract original module text
                    sm = self.syntax_tree.sourceManager
                    sr = node.sourceRange
                    buf = sr.start.buffer  # get the buffer where the node lives

                    full_text = sm.getSourceText(buf)
                    start = sr.start.offset
                    end = sr.end.offset

                    module_text = full_text[start:end]

                    f.write(f"// Module: {module_name}\n")
                    f.write(module_text)
                    f.write("\n\n")

        log.info(f"Reordered file written to: {output_file}")
        return output_file


class SystemVerilogParser:
    """Parser with AST-based reordering"""

    def __init__(self):
        self.compilation = None
        self.reorderer = VerilogReorder()

    def reorder_verilog_modules(self, filename):
        """Parse with automatic AST-based reordering"""
        output_dir = "data/verilog"
        os.makedirs(output_dir, exist_ok=True)
        basename = Path(filename).name
        output_file = os.path.join(output_dir, f"reordered_{basename}")
        log.info(f"Reordered verilog: {output_file}")

        # Parse syntax tree and reorder
        log.info("Step 1: Parsing syntax tree...")
        self.reorderer.parse_syntax_tree(filename)

        log.info("Step 2: Generating reordered file...")
        reordered_file = self.reorderer.generate_reordered_file(output_file)

        # Now compile the reordered file
        log.info("Step 3: Compiling with pyslang...")
        tree = pyslang.SyntaxTree.fromFile(reordered_file)
        self.compilation = pyslang.Compilation()
        self.compilation.addSyntaxTree(tree)

        # Check diagnostics
        all_diags = list(self.compilation.getAllDiagnostics())
        has_errors = False
        sm = self.compilation.sourceManager

        for diag in all_diags:
            location = diag.location
            line_num = sm.getLineNumber(location)
            col_num = sm.getColumnNumber(location)

            allowed_errors = [
                "ParameterDoesNotExist",
                "DuplicatePortConnection",
            ]

            code_name = str(diag.code).replace("DiagCode(", "").replace(")", "")
            if code_name in allowed_errors:
                continue

            if diag.isError():
                log.error(f"Error at line {line_num}, col {col_num}: {diag.code}")
            else:
                log.warning(f"Warning at line {line_num}, col {col_num}: {diag.code}")

        return self.compilation


# Usage Example
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    # Create test file with top-first order and a missing module
    with open("design.sv", "w") as f:
        f.write("""
module top(input clk, output [7:0] out);
    wire [7:0] cpu_out;
    wire [7:0] mem_out;

    cpu u_cpu(
        .clk(clk),
        .data_in(mem_out),
        .data_out(cpu_out)
    );

    memory u_mem(
        .clk(clk),
        .addr(cpu_out),
        .data_out(mem_out)
    );

    assign out = cpu_out;
endmodule

module cpu(input clk, input [7:0] data_in, output [7:0] data_out);
    wire [7:0] alu_result;
    wire [7:0] reg_result;

    alu u_alu(
        .a(data_in),
        .b(reg_result),
        .out(alu_result)
    );

    reg_file u_regs(
        .clk(clk),
        .d_in(alu_result),
        .d_out(reg_result)
    );

    assign data_out = alu_result;
endmodule

module alu(input [7:0] a, input [7:0] b, output [7:0] out);
    adder u_adder(.a(a), .b(b), .sum(out));
endmodule

module memory(input clk, input [7:0] addr, output [7:0] data_out);
    // This module will be a stub
    ram_block u_ram(
        .clk(clk),
        .addr(addr),
        .data(data_out)
    );
endmodule

module adder(input [7:0] a, input [7:0] b, output [7:0] sum);
    assign sum = a + b;
endmodule

module reg_file(input clk, input [7:0] d_in, output reg [7:0] d_out);
    always @(posedge clk) begin
        d_out <= d_in;
    end
endmodule
""")

    print("=== Using AST-Based Reordering ===\n")

    parser = SystemVerilogParser()
    try:
        compilation = parser.reorder_verilog_modules("design.sv")
        print("\n Success! Check design_reordered.sv")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
