import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List

import kahypar
import networkx as nx
import pygraphviz as pgv


@dataclass
class HypergraphData:
    """
    Data structure to hold hypergraph information for KaHyPar.
    """

    num_nodes: int
    num_edges: int
    index_vector: List[int]
    edge_vector: List[int]


class HypergraphPartitioner:
    def __init__(self, hypergraph_data: HypergraphData, id_to_name: Dict[int, str]):
        self.hypergraph_data = hypergraph_data
        self.id_to_name = id_to_name
        self.context = None
        self.g = None
        self.graph_json_data = None

    def hypergraph_to_graph(self):
        """
        Convert KaHyPar hypergraph to NetworkX graph using clique expansion.
        Each hyperedge connects all pairs of its nodes.
        """
        G = nx.Graph()
        for e in range(self.g.numEdges()):
            pins = list(self.g.pins(e))
            for u, v in combinations(pins, 2):
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1  # accumulate weight
                else:
                    G.add_edge(u, v, weight=1)
        return G

    def setup_run(self, k, ini_file):
        """Configures KaHyPar context with internal settings."""
        self.context = kahypar.Context()
        self.context.loadINIconfiguration(ini_file)

        # General settings
        self.context.setK(k)
        self.context.setEpsilon(0.03)  # 3% imbalance
        self.context.setSeed(42)

        # Create hypergraph for this run
        self.g = kahypar.Hypergraph(
            self.hypergraph_data.num_nodes,
            self.hypergraph_data.num_edges,
            self.hypergraph_data.index_vector,
            self.hypergraph_data.edge_vector,
            k,
        )

    def run_partitioning(self, k, ini_file):
        """Sets up and runs the partitioning."""
        self.setup_run(k, ini_file)
        kahypar.partition(self.g, self.context)
        partition_dict = {v: self.g.blockID(v) for v in range(self.g.numNodes())}
        return partition_dict

    def dump_graph_to_json(self, k, partition, data_dir="data"):
        """
        Draws and saves the graph with a hierarchical layout, grouping partitions
        into labeled clusters, and saves it as a JSON file.
        """
        G = self.hypergraph_to_graph()
        A = pgv.AGraph(directed=True, strict=False, rankdir="TB", ratio="auto")

        groups = {}
        for node, block_id in partition.items():
            groups.setdefault(block_id, []).append(node)

        for block_id, nodes in groups.items():
            subgraph_name = f"cluster_{block_id}"
            A.add_subgraph(name=subgraph_name, label=f"Partition {block_id}", color="lightgrey", style="filled")
            subgraph = A.get_subgraph(subgraph_name)
            for node in nodes:
                subgraph.add_node(self.id_to_name[node], shape="box")

        for u, v in G.edges():
            A.add_edge(self.id_to_name[u], self.id_to_name[v])

        # Set graph attributes for better layout
        A.graph_attr["splines"] = "ortho"
        A.graph_attr["overlap"] = "false"
        A.node_attr["style"] = "filled"
        A.node_attr["fillcolor"] = "white"
        A.node_attr["fontsize"] = "10"
        A.edge_attr["fontsize"] = "8"
        # The arrow can be styled with: 'dot', 'inv', 'none', 'normal', etc.
        A.edge_attr["arrowhead"] = "dot"
        A.edge_attr["arrowtail"] = "dot"
        A.edge_attr["arrowsize"] = "0.0"
        A.edge_attr["dir"] = "both"

        # Ensure output directories exist
        file_types = ["dot", "png", "json"]
        filenames = {}
        basename = f"partition_{k}"
        for ft in file_types:
            output_dir = os.path.join(data_dir, ft)
            os.makedirs(output_dir, exist_ok=True)
            filenames[ft] = os.path.join(output_dir, f"{basename}.{ft}")

        # Layout and draw the graph
        A.layout(prog="dot")
        A.write(filenames["dot"])
        A.draw(filenames["png"], format="png")
        A.draw(filenames["json"], format="json")
        print(f"Graph partition saved to {filenames['json']}")
        self.graph_json_data = filenames["json"]

