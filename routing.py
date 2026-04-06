"""Routing utilities for the optical network simulator.

This module provides shortest-path and k-shortest-path routing helpers
that operate directly on a NetworkX graph.
"""

from __future__ import annotations

from typing import List

import networkx as nx


DEFAULT_WEIGHT_ATTR = "length_km"


def get_shortest_path(graph: nx.Graph, src: str, dst: str) -> List[str]:
    """Return the Dijkstra shortest path between two nodes.

    Uses `length_km` edge attribute as link cost by default.
    """
    return nx.dijkstra_path(graph, source=src, target=dst, weight=DEFAULT_WEIGHT_ATTR)


def get_k_shortest_paths(graph: nx.Graph, src: str, dst: str, k: int) -> List[List[str]]:
    """Return up to `k` loopless shortest simple paths from src to dst.

    Paths are ordered by cumulative `length_km`.
    """
    if k < 1:
        raise ValueError("'k' must be >= 1.")

    cache_key = (src, dst, k, len(graph.edges()))
    if not hasattr(graph, '_path_cache'):
        graph._path_cache = {}
        
    if cache_key in graph._path_cache:
        return graph._path_cache[cache_key]

    try:
        paths_iter = nx.shortest_simple_paths(
            graph,
            source=src,
            target=dst,
            weight=DEFAULT_WEIGHT_ATTR,
        )
        paths = [path for _, path in zip(range(k), paths_iter)]
        
        # Limit cache size organically
        if len(graph._path_cache) > 20000:
            graph._path_cache.clear()
            
        graph._path_cache[cache_key] = paths
        return paths
    except nx.NetworkXNoPath:
        return []
