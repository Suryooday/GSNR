"""Fiber and noise model utilities for GSNR-aware optical simulation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx

from topology import _DEFAULT_TOPOLOGY


PLANCK_CONSTANT = 6.62607015e-34  # J.s
DEFAULT_OPTICAL_FREQUENCY_HZ = 193.1e12  # C-band center frequency


def _path_edges(path: Sequence[str]) -> List[Tuple[str, str]]:
    if len(path) < 2:
        raise ValueError("Path must contain at least 2 nodes.")
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _total_path_length_km(graph: nx.Graph, path: Sequence[str]) -> float:
    total = 0.0
    for u, v in _path_edges(path):
        if not graph.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) is not present in the graph.")
        total += float(graph[u][v].get("length_km", 0.0))
    return total


def compute_ase(
    signal_bandwidth_hz: float,
    noise_figure_db: float,
    path_length_km: float,
    span_length_km: float = 80.0,
    attenuation_db_per_km: float = 0.2,
    optical_frequency_hz: float = DEFAULT_OPTICAL_FREQUENCY_HZ,
) -> float:
    """Compute ASE noise power using a simple EDFA-per-span model."""
    if signal_bandwidth_hz <= 0.0:
        raise ValueError("'signal_bandwidth_hz' must be > 0.")
    if span_length_km <= 0.0:
        raise ValueError("'span_length_km' must be > 0.")
    if path_length_km < 0.0:
        raise ValueError("'path_length_km' must be >= 0.")

    number_of_spans = max(1, math.ceil(path_length_km / span_length_km))
    gain_db = attenuation_db_per_km * span_length_km
    gain_linear = 10 ** (gain_db / 10.0)

    noise_figure_linear = 10 ** (noise_figure_db / 10.0)
    spontaneous_emission_factor = noise_figure_linear / 2.0

    ase_per_amplifier = (
        spontaneous_emission_factor
        * PLANCK_CONSTANT
        * optical_frequency_hz
        * (gain_linear - 1.0)
        * signal_bandwidth_hz
    )
    return number_of_spans * ase_per_amplifier


def compute_nli_basic(signal_power_w: float, eta: float) -> float:
    """Compute basic NLI with Log-GN approximation: eta * P^3."""
    if signal_power_w < 0.0:
        raise ValueError("'signal_power_w' must be >= 0.")
    if eta < 0.0:
        raise ValueError("'eta' must be >= 0.")
    return eta * (signal_power_w ** 3)


def compute_nli_dynamic(
    graph: nx.Graph,
    path: Sequence[str],
    channel_slots: Iterable[int],
    signal_power_w: float,
    eta: float,
    slot_width_hz: float = 12.5e9,
    epsilon: float = 1e-18,
) -> float:
    """Compute dynamic NLI from neighboring occupied slots.

    For each requested slot, this function scans occupied slots on each path link,
    computes frequency spacing (`delta_f`), applies weight
    `1 / (delta_f^2 + epsilon)`, and accumulates interference.
    """
    if slot_width_hz <= 0.0:
        raise ValueError("'slot_width_hz' must be > 0.")
    if epsilon <= 0.0:
        raise ValueError("'epsilon' must be > 0.")
    if signal_power_w < 0.0:
        raise ValueError("'signal_power_w' must be >= 0.")

    target_slots = sorted(set(int(s) for s in channel_slots))
    if not target_slots:
        raise ValueError("'channel_slots' must contain at least one slot index.")

    edges = _path_edges(path)
    total_dynamic_nli = 0.0

    for u, v in edges:
        if not graph.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) is not present in the graph.")

        link_data = graph[u][v]
        occupied_slots = link_data.get("occupied_slots", [])
        slot_power = link_data.get("slot_power", [])
        slot_count = min(len(occupied_slots), len(slot_power))

        for target_slot in target_slots:
            if target_slot < 0 or target_slot >= slot_count:
                continue
            for neighbor_slot in range(slot_count):
                if neighbor_slot == target_slot or not occupied_slots[neighbor_slot]:
                    continue
                delta_f = abs(neighbor_slot - target_slot) * slot_width_hz
                weight = 1.0 / ((delta_f ** 2) + epsilon)
                neighbor_power = float(slot_power[neighbor_slot])
                total_dynamic_nli += eta * (neighbor_power ** 3) * weight

    # Self-channel nonlinear term keeps power-dependent behavior even with sparse neighbors.
    total_dynamic_nli += compute_nli_basic(signal_power_w=signal_power_w, eta=eta)
    return total_dynamic_nli


def compute_gsnr(
    graph: nx.Graph,
    path: Sequence[str],
    channel_slots: Iterable[int],
    signal_power_w: float,
    noise_figure_db: float,
    eta: float,
    slot_width_hz: float = 12.5e9,
    strategy: str = "dynamic",
) -> Dict[str, float]:
    """Compute GSNR and corresponding dB SNR for a channel over a path."""
    if signal_power_w <= 0.0:
        raise ValueError("'signal_power_w' must be > 0.")

    path_length_km = _total_path_length_km(graph, path)
    bandwidth_hz = max(1, len(set(int(s) for s in channel_slots))) * slot_width_hz

    ase_noise = compute_ase(
        signal_bandwidth_hz=bandwidth_hz,
        noise_figure_db=noise_figure_db,
        path_length_km=path_length_km,
    )
    if strategy == "static":
        nli_noise = compute_nli_basic(signal_power_w=signal_power_w, eta=eta)
    else:
        nli_noise = compute_nli_dynamic(
            graph=graph,
            path=path,
            channel_slots=channel_slots,
            signal_power_w=signal_power_w,
            eta=eta,
            slot_width_hz=slot_width_hz,
        )
    total_noise = ase_noise + nli_noise
    gsnr_linear = signal_power_w / total_noise if total_noise > 0.0 else math.inf
    snr_db = 10.0 * math.log10(gsnr_linear) if gsnr_linear > 0.0 else -math.inf

    return {
        "gsnr_linear": gsnr_linear,
        "snr_db": snr_db,
        "ase_noise": ase_noise,
        "nli_noise": nli_noise,
        "total_noise": total_noise,
    }


def compute_gsnr_default(
    path: Sequence[str],
    channel_slots: Iterable[int],
    signal_power_w: float,
    noise_figure_db: float,
    eta: float,
    slot_width_hz: float = 12.5e9,
    strategy: str = "dynamic",
) -> Dict[str, float]:
    """Convenience wrapper using the default topology graph."""
    return compute_gsnr(
        graph=_DEFAULT_TOPOLOGY.graph,
        path=path,
        channel_slots=channel_slots,
        signal_power_w=signal_power_w,
        noise_figure_db=noise_figure_db,
        eta=eta,
        slot_width_hz=slot_width_hz,
        strategy=strategy,
    )
