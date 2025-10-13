import logging as log  # Changed import to use 'log' alias
import os  # Added for file path manipulation/cleanup in main block
import re  # Added for JSON string cleanup

import jpype
import jpype.imports
import pygraphviz as pgv  # Added for DOT file manipulation and layout customization
from jpype import shutdownJVM, startJVM
from jpype.types import JFloat, JString

from schematic_from_netlist.utils.config import setup_elk

setup_elk()

# ruff: noqa: E402, E501
# pyright: reportMissingImports=false
# --- Import Xtext/EMF/ELK Libraries ---
from org.eclipse.elk.alg.graphviz.dot import GraphvizDotStandaloneSetup
from org.eclipse.elk.alg.graphviz.dot.transform import DotExporter, DotTransformationData
from org.eclipse.elk.graph.json import ElkGraphJson
from org.eclipse.elk.graph.properties import Property
from org.eclipse.elk.graph.text import ElkGraphStandaloneSetup
from org.eclipse.elk.graph.util import ElkGraphUtil
from org.eclipse.emf.common.util import URI
from org.eclipse.emf.ecore.resource.impl import ResourceSetImpl
from org.eclipse.emf.ecore.util import EcoreUtil  # Added for deep copying
from org.eclipse.xtext.resource import IResourceFactory
from org.eclipse.xtext.serializer import ISerializer


class ElkUtils:
    """
    Manages the setup and serialization of ELK Graph objects into DOT, ELKT, and JSON formats
    using the respective Xtext Standalone Setups and internal exporters.
    """

    def __init__(self):
        """Initializes Xtext injectors, serializers, and resource factories."""

        # --- DOT Setup ---
        dot_setup = GraphvizDotStandaloneSetup()
        self.dot_injector = dot_setup.createInjectorAndDoEMFRegistration()
        self.dot_serializer = self.dot_injector.getInstance(ISerializer)
        self.dot_resource_factory = self.dot_injector.getInstance(IResourceFactory)

        # --- ELKT Setup ---
        elkt_setup = ElkGraphStandaloneSetup()
        self.elkt_injector = elkt_setup.createInjectorAndDoEMFRegistration()
        self.elkt_serializer = self.elkt_injector.getInstance(ISerializer)
        self.elkt_resource_factory = self.elkt_injector.getInstance(IResourceFactory)

        self.resource_set = ResourceSetImpl()

    @staticmethod
    def create_test_graph():
        """Creates and returns a minimal ELK graph for testing."""
        graph = ElkGraphUtil.createGraph()
        graph.setIdentifier("root")

        # --- DPI Setting on Root Graph ---
        # Set a DPI property on the graph. This is often necessary for Graphviz
        # to correctly scale things like fonts and shapes.
        # Use JFloat for type safety in Java properties.
        GRAPHVIZ_DPI = jpype.JClass("org.eclipse.elk.core.options.CoreOptions").GRAPH_PROPERTY_PREFIX + ".graphviz.dot.dpi"
        graph.setProperty(GRAPHVIZ_DPI, JFloat(96.0))  # Setting DPI to 96.0

        # Nodes
        node_foo = ElkGraphUtil.createNode(graph)
        node_foo.setIdentifier("foo")
        node_bar = ElkGraphUtil.createNode(graph)
        node_bar.setIdentifier("bar")

        # Edges (creating a cycle)
        edge_foo_bar_1 = ElkGraphUtil.createEdge(graph)
        edge_foo_bar_1.setIdentifier("edge1")
        edge_foo_bar_1.getSources().add(node_foo)
        edge_foo_bar_1.getTargets().add(node_bar)

        # --- Setting Edge Label ---
        label1 = ElkGraphUtil.createLabel(edge_foo_bar_1)
        label1.setText("Signal_A")

        edge_foo_bar_2 = ElkGraphUtil.createEdge(graph)
        edge_foo_bar_2.setIdentifier("edge2")
        edge_foo_bar_2.getSources().add(node_bar)
        edge_foo_bar_2.getTargets().add(node_foo)

        # Setting a second edge label
        label2 = ElkGraphUtil.createLabel(edge_foo_bar_2)
        label2.setText("Signal_B")

        return graph

    @staticmethod
    def _create_deep_copy_and_rename(original_graph):
        """
        Creates a deep copy of the graph and renames all nodes, ports, and edges
        to compliant identifiers (n_X, p_X, e_X) for serialization.
        This is required for ELKT and JSON which have stricter naming requirements.
        """
        # Use EcoreUtil.copy for a robust EMF deep copy
        copied_graph = EcoreUtil.copy(original_graph)

        counter = {"node": 0, "port": 0, "edge": 0}

        def rename_elements(container):
            # Rename child nodes
            for node in container.getChildren():
                counter["node"] += 1
                node.setIdentifier(f"n_{counter['node']}")

                # Rename ports
                for port in node.getPorts():
                    counter["port"] += 1
                    port.setIdentifier(f"p_{counter['port']}")

                # Recurse into child nodes
                rename_elements(node)

            # Rename contained edges
            for edge in container.getContainedEdges():
                counter["edge"] += 1
                edge.setIdentifier(f"e_{counter['edge']}")

        # Start renaming from the root of the copied graph
        rename_elements(copied_graph)
        return copied_graph

    def _associate_model_with_resource(self, model, factory, extension="gv"):
        """
        Helper to create an in-memory resource and attach the model,
        which is necessary for Xtext serialization.
        """
        try:
            uri = URI.createURI(JString(f"inmemory://model.{extension}"))
            resource = factory.createResource(uri)

            if resource is not None:
                # Add the created resource to the resource set and attach the model
                self.resource_set.getResources().add(resource)
                resource.getContents().add(model)
                return resource
            else:
                log.error(f"Resource creation failed for extension .{extension}.")
                return None
        except Exception as e:
            log.error(f"FATAL ERROR during resource association for .{extension}: {e}")
            return None

    def get_dot_string(self, elk_graph):
        """
        Transforms the ELK graph to a Graphviz model and serializes it to a DOT string.
        Uses the original graph names as requested.
        """

        # 1. Transform ELK Graph to Graphviz Model (Abstract Syntax Tree)
        exporter = DotExporter()
        trans_data = DotTransformationData()

        # Set the USE_EDGE_IDS property on the transformation data to ensure edge identifiers (and labels)
        DOT_USE_EDGE_IDS = Property("dotExporter.useEdgeIds", jpype.JBoolean(False))  # Assuming JBoolean is needed for type safety
        trans_data.setProperty(DOT_USE_EDGE_IDS, jpype.JBoolean(True))

        # Use the original graph directly (without deep copy/rename)
        trans_data.setSourceGraph(elk_graph)
        exporter.transform(trans_data)

        # Get GraphvizModel
        gv_model = trans_data.getTargetGraphs().get(0)

        # 2. Associate Graphviz Model with Xtext Resource
        resource = self._associate_model_with_resource(gv_model, self.dot_resource_factory, "gv")

        if resource is not None:
            # 3. Serialize
            try:
                dot_str = self.dot_serializer.serialize(gv_model)
                return f"--- Generated DOT String ---\n{dot_str}"
            except Exception as e:
                return f"FATAL ERROR during DOT serialization: {e}"

        return "DOT Serialization skipped due to resource error."

    def write_dot_file(self, elk_graph, filepath):
        """
        Generates the DOT string, loads it into pygraphviz for layout customization,
        and writes the modified DOT string to the specified file.
        """
        dot_str_with_header = self.get_dot_string(elk_graph)

        if dot_str_with_header.startswith("FATAL ERROR") or dot_str_with_header.endswith("skipped due to resource error."):
            log.error(f"Cannot write DOT file to {filepath}. Serialization failed.")
            return

        # Extract the content by splitting the header
        raw_dot_content = dot_str_with_header.split("--- Generated DOT String ---\n", 1)[-1]

        try:
            # 1. Load the DOT content into pygraphviz
            A = pgv.AGraph(raw_dot_content)

            # 2. Change shape of all nodes to 'box'
            for node in A.nodes():
                node.attr["shape"] = "box"

            # 3. Update graph attributes for better layout
            A.graph_attr.update(
                K="0.2",
                maxiter="50000",
                mclimit="9999",  # allow more time for mincross optimization
                nslimit="9999",  # allow more time for network simplex
                nslimit1="9999",  # same for phase 3 placement
                overlap="vpsc",
                pack="true",
                packmode="graph",
                sep="+20,20",
                esep="+2,2",
                epsilon="1e-7",
                rankdir="LR",
                start="rand",  # better escape from local minima
                mode="hier",  # hierarchical bias
                ratio="auto",
                splines="ortho",
            )

            # 4. Generate the final modified DOT string
            final_dot_content = A.string()

            # 5. Write the modified content to file
            with open(filepath, "w") as f:
                f.write(final_dot_content)
            log.debug(f" wrote DOT content to {filepath}")

        except Exception as e:
            log.error(f"Failed to process or write DOT file {filepath} using pygraphviz: {e}")

    def get_elkt_string(self, elk_graph):
        """Serializes the ELK graph directly to an ELKT string."""

        # Create compliant copy for serialization
        graph_copy = self._create_deep_copy_and_rename(elk_graph)

        # 1. Associate ELK Graph Model with Xtext Resource
        resource = self._associate_model_with_resource(graph_copy, self.elkt_resource_factory, "elkt")

        if resource is not None:
            # 2. Serialize
            try:
                # The ELKT serializer uses the copied, renamed ELK graph directly
                elkt_str = self.elkt_serializer.serialize(graph_copy)
                elkt_str = re.sub(r'^.*"resolvedAlgorithm".*\n?', "", elkt_str, flags=re.MULTILINE)
                return f"--- Generated ELKT String ---\n{elkt_str}"
            except Exception as e:
                return f"FATAL ERROR during ELKT serialization: {e}"

        return "ELKT Serialization skipped due to resource error."

    def write_elkt_file(self, elk_graph, filepath):
        """Generates the ELKT string and writes it to the specified file."""
        elkt_str_with_header = self.get_elkt_string(elk_graph)

        if elkt_str_with_header.startswith("FATAL ERROR") or elkt_str_with_header.endswith("skipped due to resource error."):
            log.error(f"Cannot write ELKT file to {filepath}. Serialization failed.")
            return

        # Extract the content by splitting the header
        content = elkt_str_with_header.split("--- Generated ELKT String ---\n", 1)[-1]

        try:
            with open(filepath, "w") as f:
                f.write(content)
            log.debug(f" wrote ELKT content to {filepath}")
        except Exception as e:
            log.error(f"Failed to write file {filepath}: {e}")

    def get_json_string(self, elk_graph):
        """Serializes the ELK graph directly to a JSON string using ElkGraphJson exporter."""

        try:
            # 1. Initialize JSON Export Builder
            graph_copy = self._create_deep_copy_and_rename(elk_graph)
            export_builder = ElkGraphJson.forGraph(graph_copy)

            # 2. Configure options:
            export_builder.prettyPrint(True)  #
            export_builder.omitLayout(False)  # -> removes "layoutOptions": {...}
            export_builder.omitZeroPositions(False)  # -> omit nodes with (0,0)
            export_builder.omitZeroDimension(True)  # -> omit width=0 or height=0
            export_builder.shortLayoutOptionKeys(True)  # -> use short keys
            export_builder.omitUnknownLayoutOptions(True)  # -> hide unknown options

            # 3. Finally generate JSON
            json_str = str(export_builder.toJson())

            # 4. Apply regex substitution
            json_str = re.sub(r'^.*"resolvedAlgorithm".*\n?', "", json_str, flags=re.MULTILINE)

            return f"--- Generated JSON String ---\n{json_str}"
        except Exception as e:
            return f"FATAL ERROR during JSON serialization: {e}"

    def write_json_file(self, elk_graph, filepath):
        """Generates the JSON string and writes it to the specified file."""
        json_str_with_header = self.get_json_string(elk_graph)

        if json_str_with_header.startswith("FATAL ERROR"):
            log.error(f"Cannot write JSON file to {filepath}. Serialization failed.")
            return

        # Extract the content by splitting the header
        content = json_str_with_header.split("--- Generated JSON String ---\n", 1)[-1]

        try:
            with open(filepath, "w") as f:
                f.write(content)
            log.debug(f" wrote JSON content to {filepath}")
        except Exception as e:
            log.error(f"Failed to write file {filepath}: {e}")


if __name__ == "__main__":
    # --- Java Configuration ---

    # Configure logging for main execution
    log.basicConfig(level=log.debug, format="%(levelname)s: %(message)s")

    # Define temporary file paths
    dot_path = "output_test_graph.gv"
    elkt_path = "output_test_graph.elkt"
    json_path = "output_test_graph.json"

    # 1. Initialize Serializer Class (runs Xtext setups)
    elk_utils = ElkUtils()

    # 2. Create Test Graph
    test_graph = ElkUtils.create_test_graph()

    # 3. Test Serialization (Print Strings)
    print("\n--- String Outputs ---")
    print(elk_utils.get_dot_string(test_graph))
    print(elk_utils.get_elkt_string(test_graph))
    print(elk_utils.get_json_string(test_graph))
    print("----------------------\n")

    # 4. Test File Writing
    log.debug("--- Testing File Writes ---")
    elk_utils.write_dot_file(test_graph, dot_path)
    elk_utils.write_elkt_file(test_graph, elkt_path)
    elk_utils.write_json_file(test_graph, json_path)
    log.debug("---------------------------")

    # Optional: Cleanup temporary files
    # if os.path.exists(dot_path): os.remove(dot_path)
    # if os.path.exists(elkt_path): os.remove(elkt_path)
    # if os.path.exists(json_path): os.remove(json_path)

    if jpype.isJVMStarted():
        log.debug("Shutting down JVM...")
        shutdownJVM()
