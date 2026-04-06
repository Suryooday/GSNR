"""Dataset loader module for optical network simulation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def load_topology(file_path: str, default_length_km: float = 80.0, total_slots: int = 320) -> nx.Graph:
    """Load an optical network topology from a generic text/GML or JSON file.

    Args:
        file_path: Path to the dataset graph file (.txt, .gml, .json).
        default_length_km: Default physical distance to assign to links if unprovided.
        total_slots: The number of frequency slots to initialize per link.

    Returns:
        A NetworkX Graph fully initialized and compatible with the simulation engine.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Topology dataset not found: {file_path}")

    G = None
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Assuming standard node-link JSON format
            if isinstance(data, dict) and "nodes" in data and "links" in data:
                G = nx.node_link_graph(data)
            else:
                raise ValueError("JSON file is not in standard node-link format.")
    else:
        # Assuming GML based on standard NSFNET/GEANT datasets in .txt or .gml format
        try:
            # Provide string path directly to nx.read_gml
            G = nx.read_gml(str(path))
        except Exception as e:
            logger.error("Failed parsing GML structure. Trying simple edgelist fallback.")
            try:
                G = nx.read_edgelist(str(path))
            except Exception as e_edge:
                raise ValueError(f"Could not parse topology from {file_path}. Errors: {e}, {e_edge}")

    # Ensure undirected graph type as optical fibers are generally bidirectional 
    # (Though we can handle directed if specifically modeling independent fibers for rx/tx)
    if G.is_directed():
        # Keep directed if it natively enforces asymmetrical routing, or convert to undirected
        # Optical topologies in standard datasets are usually modeled as single undirected graph
        G = G.to_undirected()

    # Create a fresh graph and map everything over purely for compatibility
    # and ensuring node labels are clean strings (as required by routing engine)
    clean_graph = nx.Graph()

    for node, data in G.nodes(data=True):
        node_id = str(node)
        clean_graph.add_node(node_id, **data)

    for u, v, data in G.edges(data=True):
        u_str = str(u)
        v_str = str(v)
        
        # Check if the dataset provided a physical distance/length (or weight)
        length_val = data.get("length_km") or data.get("distance") or data.get("length")
        if length_val is None:
            # Datasets loosely use 'weight' sometimes to denote distance ratio
            weight_val = data.get("weight")
            if weight_val is not None:
                length_val = float(weight_val) * default_length_km
            else:
                length_val = default_length_km
                
        # Basic compliance metadata for the physical fiber engine & spectrum manager
        edge_attrs = {
            "length_km": float(length_val),
            "available_spectrum_slots": [],
            "occupied_slots": [False] * total_slots,
            "slot_power": [0.0] * total_slots,
            **data  # preserve any original dataset attributes like `bandwidth` or `port`
        }
        
        clean_graph.add_edge(u_str, v_str, **edge_attrs)

    logger.info("Loaded dataset topology %s successfully with %d nodes and %d links.", 
                path.name, clean_graph.number_of_nodes(), clean_graph.number_of_edges())
    
    return clean_graph
