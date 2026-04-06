"""Topology utilities for optical network simulation.

This module provides a leaf-spine topology generator using NetworkX.
Each link stores physical length and available spectrum slots.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx


class OpticalTopology:
    """Encapsulates graph creation and read-only topology accessors."""

    def __init__(self) -> None:
        self.graph = nx.Graph()

    def generate_leaf_spine(self, spines: int, leafs: int) -> None:
        """Generate a fully-connected leaf-spine topology.

        Args:
            spines: Number of spine switches/routers.
            leafs: Number of leaf switches/routers.

        Raises:
            ValueError: If either count is less than 1.
        """
        if spines < 1 or leafs < 1:
            raise ValueError("Both 'spines' and 'leafs' must be >= 1.")

        self.graph.clear()

        spine_nodes = [f"spine_{idx}" for idx in range(spines)]
        leaf_nodes = [f"leaf_{idx}" for idx in range(leafs)]

        self.graph.add_nodes_from(spine_nodes, role="spine")
        self.graph.add_nodes_from(leaf_nodes, role="leaf")

        default_length_km = 80.0

        for leaf in leaf_nodes:
            for spine in spine_nodes:
                self.graph.add_edge(
                    leaf,
                    spine,
                    length_km=default_length_km,
                    available_spectrum_slots=[],
                )

    def get_nodes(self) -> List[str]:
        """Return all node names in the topology."""
        return list(self.graph.nodes())

    def get_links(self) -> List[Dict[str, object]]:
        """Return all links and their attributes."""
        links: List[Dict[str, object]] = []
        for source, target, data in self.graph.edges(data=True):
            links.append(
                {
                    "source": source,
                    "target": target,
                    "length_km": data["length_km"],
                    "available_spectrum_slots": list(data["available_spectrum_slots"]),
                }
            )
        return links


_DEFAULT_TOPOLOGY = OpticalTopology()


def generate_leaf_spine(spines: int, leafs: int) -> None:
    """Generate a leaf-spine topology in the module-level default instance."""
    _DEFAULT_TOPOLOGY.generate_leaf_spine(spines=spines, leafs=leafs)


def get_nodes() -> List[str]:
    """Return nodes from the module-level default topology."""
    return _DEFAULT_TOPOLOGY.get_nodes()


def get_links() -> List[Dict[str, object]]:
    """Return links from the module-level default topology."""
    return _DEFAULT_TOPOLOGY.get_links()
