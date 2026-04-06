"""Routing and Spectrum Assignment (RSA) optimizer."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from routing import get_k_shortest_paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModulationProfile:
    """Represents a modulation option and its GSNR requirement."""

    name: str
    spectral_efficiency: float
    min_snr_db: float
    nominal_ber: float


class RSAOptimizer:
    """GSNR-aware optimizer for route and spectrum assignment."""

    def __init__(
        self,
        graph: nx.Graph,
        spectrum_manager: object,
        fiber_model: object,
        slot_width_hz: float = 12.5e9,
        k_paths: int = 5,
        noise_figure_db: float = 5.0,
        eta: float = 1e-3,
        signal_power_w: float = 1e-3,
    ) -> None:
        self.graph = graph
        self.spectrum_manager = spectrum_manager
        self.fiber_model = fiber_model
        self.slot_width_hz = slot_width_hz
        self.k_paths = k_paths
        self.noise_figure_db = noise_figure_db
        self.eta = eta
        self.signal_power_w = signal_power_w
        self.modulations: List[ModulationProfile] = [
            ModulationProfile("16QAM", spectral_efficiency=4.0, min_snr_db=18.0, nominal_ber=1e-3),
            ModulationProfile("8QAM", spectral_efficiency=3.0, min_snr_db=14.0, nominal_ber=2e-3),
            ModulationProfile("QPSK", spectral_efficiency=2.0, min_snr_db=9.0, nominal_ber=1e-2),
            ModulationProfile("BPSK", spectral_efficiency=1.0, min_snr_db=6.0, nominal_ber=5e-2),
        ]

    @staticmethod
    def _path_length_km(graph: nx.Graph, path: Sequence[str]) -> float:
        return sum(float(graph[u][v].get("length_km", 0.0)) for u, v in zip(path, path[1:]))

    @staticmethod
    def _path_latency_ms(path_length_km: float) -> float:
        # Propagation delay with ~2e8 m/s in fiber.
        return (path_length_km * 1000.0 / 2e8) * 1000.0

    def _compute_marginal_power(self, path: Sequence[str]) -> float:
        # Energy-aware factor: Transponders at ends + EDFAs every 80 km
        transponder_power = 100.0  # (200W total per path)
        amplifier_active_marginal = 20.0 # Additional Watts drawn when turning IDLE amp into ACTIVE amp (30W - 10W)
        
        total_marginal_amps = 0
        for u, v in zip(path, path[1:]):
            edge_length = float(self.graph[u][v].get("length_km", 80.0))
            num_amps = max(0, int(edge_length / 80.0))
            
            # Check edge physical state
            occupied_slots = self.graph[u][v].get("occupied_slots", [])
            # If all are False/0
            if not any(occupied_slots):
                total_marginal_amps += num_amps
                
        return (transponder_power * 2.0) + (total_marginal_amps * amplifier_active_marginal)

    @staticmethod
    def _path_edges(path: Sequence[str]) -> List[Tuple[str, str]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    @staticmethod
    def _simple_ber_mapping(snr_db: float, profile: ModulationProfile) -> float:
        margin = snr_db - profile.min_snr_db
        if margin >= 6.0:
            return profile.nominal_ber * 0.1
        if margin >= 3.0:
            return profile.nominal_ber * 0.5
        if margin >= 0.0:
            return profile.nominal_ber
        return min(0.5, profile.nominal_ber * 20.0)

    def _required_slots(self, bit_rate_gbps: float, profile: ModulationProfile) -> int:
        if bit_rate_gbps <= 0.0:
            raise ValueError("'bit_rate_gbps' must be > 0.")
        capacity_per_slot_bps = self.slot_width_hz * profile.spectral_efficiency
        return max(1, math.ceil((bit_rate_gbps * 1e9) / capacity_per_slot_bps))

    def _find_candidate_slot_blocks(
        self,
        path: Sequence[str],
        required_slots: int,
        max_blocks: int,
    ) -> List[List[int]]:
        edges = self._path_edges(path)
        if not edges:
            return []

        total_slots = int(getattr(self.spectrum_manager, "total_slots", 0))
        if total_slots <= 0:
            return []

        common_free = [True] * total_slots
        for u, v in edges:
            if not self.graph.has_edge(u, v):
                return []
            link_data = self.graph[u][v]
            occupied = link_data.get("occupied_slots", [False] * total_slots)
            if len(occupied) < total_slots:
                occupied = occupied + [False] * (total_slots - len(occupied))
            for idx in range(total_slots):
                if occupied[idx]:
                    common_free[idx] = False

        blocks: List[List[int]] = []
        run_start = -1
        run_len = 0
        for idx, free_flag in enumerate(common_free):
            if free_flag:
                if run_start == -1:
                    run_start = idx
                run_len += 1
                if run_len >= required_slots:
                    block = list(range(idx - required_slots + 1, idx + 1))
                    blocks.append(block)
                    if len(blocks) >= max_blocks:
                        break
            else:
                run_start = -1
                run_len = 0
        return blocks

    def _compute_gsnr(self, path: Sequence[str], slots: Iterable[int], strategy: str = "dynamic") -> Dict[str, float]:
        compute_gsnr_fn = getattr(self.fiber_model, "compute_gsnr", None)
        if compute_gsnr_fn is None:
            # Fallback if fiber model lacks compute_gsnr
            return {"snr_db": 20.0, "gsnr_linear": 100.0}
        
        # We pass strategy down. For shortest_path, we can use static GSNR fast math since it isn't scored on NLI anyway.
        gsnr_strategy = "static" if strategy in ("static_gsnr", "shortest_path") else "dynamic"

        return compute_gsnr_fn(
            graph=self.graph,
            path=path,
            channel_slots=slots,
            signal_power_w=0.001,  # 0 dBm per channel
            noise_figure_db=5.0,
            eta=1e-3, # basic non-linear coefficient
            strategy=gsnr_strategy,
            slot_width_hz=self.slot_width_hz,
        )

    def find_best_assignment(
        self,
        source: str,
        destination: str,
        bit_rate_gbps: float = 100.0,
        max_slot_trials_per_path: int = 8,
        allocate: bool = True,
        w_latency: float = 1.0,
        w_gsnr: float = 100.0,
        w_power: float = 1.0,
        w_distance: float = 1.0,
        strategy: str = "dynamic_gsnr",
    ) -> Optional[Dict[str, object]]:
        """Find the optimal path and spectrum assignment.
        
        Strategy Options:
          - 'shortest_path': pure Dijkstra minimizing distance.
          - 'static_gsnr': multi-objective balancing but simple static NLI.
          - 'dynamic_gsnr': full multi-objective with active neighbor NLI interference.
        """
        if strategy == "shortest_path":
            w_latency, w_gsnr, w_power, w_distance = 0.0, 0.0, 0.0, 1.0
            
        rl_candidates = []

        if not self.graph.has_node(source) or not self.graph.has_node(destination):
            return None
        try:
            candidate_paths = get_k_shortest_paths(
                self.graph,
                src=source,
                dst=destination,
                k=self.k_paths,
            )
        except Exception:
            logger.exception("Candidate path search failed for %s -> %s", source, destination)
            return None
        if not candidate_paths:
            return None

        best_result: Optional[Dict[str, object]] = None
        best_score = math.inf

        for path in candidate_paths:
            path_cost = self._path_length_km(self.graph, path)
            latency = self._path_latency_ms(path_cost)
            power_w = self._compute_marginal_power(path)
            
            # Fast rejection based on distance threshold could go here.

            # Try high-efficiency modulation first, then lower modulation fallback.
            for profile in self.modulations:
                required_slots = self._required_slots(bit_rate_gbps, profile)
                slot_blocks = self._find_candidate_slot_blocks(
                    path=path,
                    required_slots=required_slots,
                    max_blocks=max_slot_trials_per_path,
                )
                if not slot_blocks:
                    continue

                for slots in slot_blocks:
                    gsnr_info = self._compute_gsnr(path=path, slots=slots, strategy=strategy)
                    snr_db = float(gsnr_info["snr_db"])
                    
                    if strategy != "shortest_path":
                        if snr_db < profile.min_snr_db:
                            continue

                    if strategy != "rl" and allocate and not self.spectrum_manager.allocate_slots(path, slots):
                        continue

                    # Multi-Objective Score
                    # score = w1*latency + w2*(1/gsnr) + w3*power + w4*distance
                    score = (
                        (w_latency * latency) +
                        (w_gsnr * (1.0 / (float(gsnr_info["gsnr_linear"]) + 1e-9))) +
                        (w_power * power_w) +
                        (w_distance * path_cost)
                    )

                    result = {
                        "path": path,
                        "slots": slots,
                        "modulation": profile.name,
                        "gsnr_linear": float(gsnr_info["gsnr_linear"]),
                        "snr_db": snr_db,
                        "ber": self._simple_ber_mapping(snr_db, profile),
                        "latency_ms": latency,
                        "path_cost_km": path_cost,
                        "power_w": power_w,
                        "score": score
                    }
                    
                    if strategy == "rl":
                        rl_candidates.append(result)
                        continue
                        
                    if score < best_score:
                        if allocate:
                            # if we allocate here dynamically, we must release previous best if we switch.
                            pass # We handle allocation later
                        best_result = result
                        best_score = score
                    break

        if strategy == "rl":
            if not rl_candidates:
                return None
            return {"rl_candidates": rl_candidates, "source": source, "destination": destination}

        if best_result is not None and allocate and strategy != "rl":
            self.spectrum_manager.allocate_slots(best_result["path"], best_result["slots"])

        return best_result


def optimize_rsa(
    graph: nx.Graph,
    source: str,
    destination: str,
    spectrum_manager: object,
    fiber_model: object,
    bit_rate_gbps: float = 100.0,
    strategy: str = "dynamic_gsnr",
) -> Optional[Dict[str, object]]:
    """Convenience wrapper to execute one RSA optimization run."""
    optimizer = RSAOptimizer(
        graph=graph,
        spectrum_manager=spectrum_manager,
        fiber_model=fiber_model,
    )
    return optimizer.find_best_assignment(
        source=source,
        destination=destination,
        bit_rate_gbps=bit_rate_gbps,
        strategy=strategy,
    )
