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


from dataclasses import dataclass, field


@dataclass
class Edge:
    src: str
    dst: str
    name: str | None = None  # optional label
    color: str | None = None  # optional color
    fontsize: int | None = None
    weight: int | None = None

    @property
    def attrs(self) -> dict:
        """Return a dictionary of Graphviz attributes, skipping None values."""
        return {k: v for k, v in {"label": self.name, "color": self.color, "fontsize": self.fontsize, "weight": self.weight}.items() if v is not None}


class HypergraphPartitioner:
    def __init__(self, hypergraph_data: HypergraphData, db):
        self.hypergraph_data = hypergraph_data
        self.db = db
        self.id_to_name = db.instname_by_id
        self.context = None
        self.g = None
        self.k = 1
        self.graph_json_data = None

    def hypergraph_to_graph(self):
        """
        Convert KaHyPar hypergraph to NetworkX graph using clique expansion.
        Each hyperedge connects all pairs of its nodes.
        """
        G = nx.Graph()
        for e in range(self.g.numEdges()):
            pins = list(self.g.pins(e))
            # print(f"{e=} {len(pins)=}")
            # for pin in pins:
            #   print(f"  {pin=}  {self.id_to_name[pin]=}")
            for u, v in combinations(pins, 2):
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1  # accumulate weight
                else:
                    G.add_edge(u, v, weight=1)
        return G

    def compute_modularity_and_conductance(self, partition):
        """
        partition: dict node_id -> block_id
        Returns: modularity, average conductance
        """
        G = self.hypergraph_to_graph()
        # Prepare communities list for NetworkX modularity
        communities = {}
        for node, block in partition.items():
            communities.setdefault(block, set()).add(node)
        community_list = list(communities.values())

        # Compute modularity
        modularity = nx.algorithms.community.quality.modularity(G, community_list, weight="weight")

        # Compute conductance for each community
        conductances = []
        for comm in community_list:
            cond = nx.algorithms.cuts.conductance(G, comm)
            conductances.append(cond)
        avg_conductance = sum(conductances) / len(conductances)

        return modularity, avg_conductance

    def combined_score(self, modularity, conductance, w_mod=0.7, w_cond=0.3):
        """
        Combine modularity and conductance into a single score.
        Higher score = better.
        w_mod + w_cond should = 1
        """
        good_cond = 1.0 - conductance
        return w_mod * modularity + w_cond * good_cond

    def evaluate_run(self):
        self.extract_groups()
        print("Cut edges:", self.cut_metric())
        partition_dict = {v: self.g.blockID(v) for v in range(self.g.numNodes())}

        modularity, avg_cond = self.compute_modularity_and_conductance(partition_dict)
        combined = self.combined_score(modularity, avg_cond)

        print(f"QOR: Modularity: {modularity:.4f} Conductance: {avg_cond:.4f}  combined {combined:.4f}")
        return modularity, avg_cond, combined

    def extract_groups(self):
        groups = {}
        for v in range(self.g.numNodes()):
            part = self.g.blockID(v)
            groups.setdefault(part, []).append(v)

        for group, members in groups.items():
            nodes = [self.id_to_name[v] for v in members]
            print(f"Group {group}: {nodes}")

        return groups

    def cut_metric(self):
        cut_edges = 0
        for e in range(self.g.numEdges()):
            parts_touched = set(self.g.blockID(v) for v in self.g.pins(e))
            if len(parts_touched) > 1:
                cut_edges += 1
        return cut_edges

    def setup_run(self, ini_file):
        """Configures KaHyPar context with internal settings."""
        self.context = kahypar.Context()
        self.context.loadINIconfiguration(ini_file)

        # General settings
        self.context.setK(self.k)
        self.context.setEpsilon(0.90)  # higher => more imbalance
        self.context.setSeed(42)

        # Create hypergraph for this run
        self.g = kahypar.Hypergraph(
            self.hypergraph_data.num_nodes,
            self.hypergraph_data.num_edges,
            self.hypergraph_data.index_vector,
            self.hypergraph_data.edge_vector,
            self.k,
        )

    def run_partitioning(self, k, ini_file):
        """Sets up and runs the partitioning."""
        self.k = k
        self.setup_run(ini_file)
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
        print(f"Partition into {k} clusters complete")

        # --- Add text information ---
        if self.k > 1:
            modularity, conductance, combined = self.evaluate_run()
            info_text = f"Partitions: {k}\nModularity: {modularity:.4f}\nConductance: {conductance:.4f}\nCombined Score: {combined:.4f}"
            A.graph_attr["label"] = info_text
            A.graph_attr["labelloc"] = "t"
            A.graph_attr["fontsize"] = "20"

        # --- Node size scaling ---
        degrees = dict(G.degree())
        min_degree = min(degrees.values()) if degrees else 1
        max_degree = max(degrees.values()) if degrees else 1

        min_macro_size = 1.0
        max_macro_size = 10.0
        buffer_size = 0.2

        def scale_macro_size(degree):
            if max_degree == min_degree:
                return min_macro_size
            size = min_macro_size + (degree - min_degree) * (max_macro_size - min_macro_size) / (max_degree - min_degree)
            # round to one decimal place
            return round(size, 1)

        # Group nodes by partition
        groups = {}
        for node, block_id in partition.items():
            groups.setdefault(block_id, []).append(node)

        for block_id, nodes in groups.items():
            subgraph_name = f"cluster_{block_id}"
            A.add_subgraph(name=subgraph_name, label=f"Partition {block_id}", color="lightgrey", style="filled")
            subgraph = A.get_subgraph(subgraph_name)
            for node in nodes:
                node_name = self.id_to_name.get(node, str(node))
                if node_name.startswith("buf⊕_"):
                    subgraph.add_node(node_name, width=buffer_size, height=buffer_size, fixedsize=True, shape="circle")
                else:
                    degree = degrees.get(node, 0)
                    size = scale_macro_size(degree)
                    subgraph.add_node(node_name, width=size, height=size, fixedsize=True, shape="box")
                    # printf(f"Node {node_name} has degree {degree} and size {size}")

        for block_id, nodes in groups.items():
            edges = self.db.get_edges_between_nodes(nodes)
            for edge in edges:
                A.add_edge(edge.src, edge.dst, **edge.attrs)

        # Set graph attributes for better layout

        # Graph-level attributes
        A.graph_attr.update(
            {
                "fontsize": "20",
                "maxiter": "10000",  # Allow more iterations for force-based layouts
                "overlap": "false",  # Avoid node overlaps
                "pack": "false",  # Enable packing of disconnected components
                "sep": "+20",  # Extra separation between clusters
                "splines": "ortho",  # Orthogonal edge routing
            }
        )

        # Node-level attributes
        A.node_attr.update(
            {
                "style": "filled",
                "fillcolor": "white",
                "fontsize": "10",
            }
        )

        # Edge-level attributes
        A.edge_attr.update(
            {
                "fontsize": "8",
                "arrowhead": "dot",  # Options: 'dot', 'inv', 'normal', 'none', ...
                "arrowtail": "dot",
                "arrowsize": "0.0",  # Zero-size arrows (basically hidden)
                "dir": "both",  # Draw arrows at both ends
            }
        )

        # Ensure output directories exist
        file_types = ["dot", "png", "json"]
        filenames = {}
        basename = f"partition_{k}"
        for ft in file_types:
            output_dir = os.path.join(data_dir, ft)
            os.makedirs(output_dir, exist_ok=True)
            filenames[ft] = os.path.join(output_dir, f"{basename}.{ft}")

        # Layout and draw the graph
        A.write(filenames["dot"])
        # A.layout(prog="dot", args="-v")
        A.layout(prog="sfdp", args="-v")
        A.draw(filenames["png"], format="png")
        A.draw(filenames["json"], format="json")
        print(f"Graph saved to {filenames['json']}")
        self.graph_json_data = filenames["json"]
