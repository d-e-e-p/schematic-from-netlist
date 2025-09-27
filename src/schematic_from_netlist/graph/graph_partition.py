import json
import os
import re
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
    headlabel: str | None = None
    taillabel: str | None = None

    @property
    def attrs(self) -> dict:
        """Return a dictionary of Graphviz attributes, skipping None values."""
        return {
            k: v
            for k, v in {"label": self.name, "color": self.color, "fontsize": self.fontsize, "weight": self.weight}.items()
            if v is not None
        }


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
            # pinnames = [self.id_to_name[pin] for pin in pins]
            # print(f"{e=} {len(pins)=} {pinnames=}")
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
        ctx = kahypar.Context()
        ctx.loadINIconfiguration(ini_file)

        # General settings
        ctx.setK(self.k)
        ctx.setEpsilon(0.90)  # higher => more imbalance
        ctx.setSeed(42)
        ctx.suppressOutput(True)  # <-- squelch KaHyPar logging

        # Create hypergraph for this run
        self.context = ctx
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
        if self.k > 1:
            self.evaluate_run()
        partition_dict = {v: self.g.blockID(v) for v in range(self.g.numNodes())}
        return partition_dict
