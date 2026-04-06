"""Visualization utilities for optical network simulations."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def _path_edges(path: Sequence[str]) -> List[Tuple[str, str]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def plot_topology(
    graph: nx.Graph,
    selected_path: Optional[Sequence[str]] = None,
    title: str = "Optical Topology",
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot network topology and optionally highlight a selected path."""
    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="#BFD7EA", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(graph, pos, edge_color="#9E9E9E", width=1.4, ax=ax)

    edge_labels = {
        (u, v): f"{float(data.get('length_km', 0.0)):.0f} km"
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    if selected_path and len(selected_path) >= 2:
        path_edges = _path_edges(selected_path)
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=path_edges,
            edge_color="#D62728",
            width=3.0,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=list(selected_path),
            node_size=760,
            node_color="#FFBE7D",
            ax=ax,
        )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def plot_gsnr_vs_distance(
    distances_km: Sequence[float],
    gsnr_db: Sequence[float],
    title: str = "GSNR vs Distance",
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot GSNR (dB) against path distance (km)."""
    if len(distances_km) != len(gsnr_db):
        raise ValueError("distances_km and gsnr_db must have the same length.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(distances_km, gsnr_db, marker="o", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("GSNR (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_latency_vs_load(
    offered_load: Sequence[float],
    latency_ms: Sequence[float],
    title: str = "Latency vs Load",
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot average latency as a function of offered load."""
    if len(offered_load) != len(latency_ms):
        raise ValueError("offered_load and latency_ms must have the same length.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(offered_load, latency_ms, marker="s", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Offered Load")
    ax.set_ylabel("Average Latency (ms)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_blocking_probability(
    x_values: Sequence[float],
    blocking_probabilities: Sequence[float],
    x_label: str = "Offered Load",
    title: str = "Blocking Probability",
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot blocking probability curve for a chosen x-axis metric."""
    if len(x_values) != len(blocking_probabilities):
        raise ValueError("x_values and blocking_probabilities must have the same length.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_values, blocking_probabilities, marker="^", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Blocking Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def show_all() -> None:
    """Convenience helper to render all pending matplotlib figures."""
    plt.show()
