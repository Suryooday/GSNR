"""API orchestration layer for UI and external clients."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fiber_model
from logging_utils import configure_logging
from optimizer import RSAOptimizer
from routing import get_shortest_path
from spectrum_manager import SpectrumManager
from topology import OpticalTopology
from traffic_engine import TrafficEngine

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    number_of_nodes: int = 12
    input_power_w: float = 1e-3
    noise_figure_db: float = 5.0
    traffic_load_lambda: float = 1.0
    service_rate_mu: float = 1.0
    total_slots: int = 320
    bit_rate_gbps: float = 100.0


class SimulationAPI:
    """Coordinates UI/API requests with core and AI modules."""

    def __init__(self, config: SimulatorConfig) -> None:
        configure_logging()
        self.config = config
        self.topology: Optional[OpticalTopology] = None
        self.spectrum_manager: Optional[SpectrumManager] = None
        self.optimizer: Optional[RSAOptimizer] = None

    @staticmethod
    def _split_nodes(total_nodes: int) -> Tuple[int, int]:
        if total_nodes < 2:
            return 1, 1
        spines = max(1, int(round(total_nodes * 0.3)))
        leafs = max(1, total_nodes - spines)
        return spines, leafs

    def build(self) -> None:
        spines, leafs = self._split_nodes(self.config.number_of_nodes)
        topology = OpticalTopology()
        topology.generate_leaf_spine(spines=spines, leafs=leafs)
        spectrum_manager = SpectrumManager(topology.graph, total_slots=self.config.total_slots)
        optimizer = RSAOptimizer(
            graph=topology.graph,
            spectrum_manager=spectrum_manager,
            fiber_model=fiber_model,
            signal_power_w=self.config.input_power_w,
            noise_figure_db=self.config.noise_figure_db,
        )
        self.topology = topology
        self.spectrum_manager = spectrum_manager
        self.optimizer = optimizer
        logger.info("SimulationAPI built topology with %s nodes.", len(topology.get_nodes()))

    def get_nodes(self) -> List[str]:
        if self.topology is None:
            self.build()
        assert self.topology is not None
        return sorted(self.topology.get_nodes())

    def compute_path(self, source: str, destination: str) -> Optional[Dict[str, object]]:
        if self.optimizer is None:
            self.build()
        assert self.optimizer is not None
        try:
            result = self.optimizer.find_best_assignment(
                source=source,
                destination=destination,
                bit_rate_gbps=self.config.bit_rate_gbps,
                allocate=True,
            )
            if result is None:
                logger.warning("No feasible assignment for %s -> %s", source, destination)
            return result
        except Exception:
            logger.exception("Path computation failed for %s -> %s", source, destination)
            return None

    def gsnr_distance_curve(self, source: str, slots: List[int]) -> Tuple[List[float], List[float]]:
        if self.topology is None or self.optimizer is None:
            self.build()
        assert self.topology is not None and self.optimizer is not None

        distances: List[float] = []
        gsnr_db_values: List[float] = []
        for node in self.get_nodes():
            if node == source:
                continue
            try:
                path = get_shortest_path(self.topology.graph, source, node)
                gsnr_info = fiber_model.compute_gsnr(
                    graph=self.topology.graph,
                    path=path,
                    channel_slots=slots,
                    signal_power_w=self.config.input_power_w,
                    noise_figure_db=self.config.noise_figure_db,
                    eta=self.optimizer.eta,
                    slot_width_hz=self.optimizer.slot_width_hz,
                )
                distance = sum(
                    float(self.topology.graph[u][v].get("length_km", 0.0))
                    for u, v in zip(path, path[1:])
                )
                distances.append(distance)
                gsnr_db_values.append(float(gsnr_info["snr_db"]))
            except Exception:
                logger.exception("Failed GSNR curve point from %s to %s", source, node)
        return distances, gsnr_db_values

    def traffic_sweep(self, load_points: List[float], requests: int) -> Tuple[List[float], List[float]]:
        avg_latency_points: List[float] = []
        blocking_points: List[float] = []

        for lam in load_points:
            try:
                sim_api = SimulationAPI(
                    SimulatorConfig(
                        number_of_nodes=self.config.number_of_nodes,
                        input_power_w=self.config.input_power_w,
                        noise_figure_db=self.config.noise_figure_db,
                        traffic_load_lambda=lam,
                        service_rate_mu=self.config.service_rate_mu,
                        total_slots=self.config.total_slots,
                        bit_rate_gbps=self.config.bit_rate_gbps,
                    )
                )
                sim_api.build()
                assert sim_api.topology is not None
                assert sim_api.spectrum_manager is not None
                assert sim_api.optimizer is not None
                engine = TrafficEngine(
                    graph=sim_api.topology.graph,
                    spectrum_manager=sim_api.spectrum_manager,
                    optimizer=sim_api.optimizer,
                    arrival_rate_lambda=lam,
                    service_rate_mu=self.config.service_rate_mu,
                    bit_rate_gbps=self.config.bit_rate_gbps,
                    random_seed=42,
                )
                kpis = engine.run(total_requests=requests)
                avg_latency_points.append(float(kpis["average_latency_ms"]))
                blocking_points.append(float(kpis["blocking_probability"]))
            except Exception:
                logger.exception("Traffic sweep failed at load=%s", lam)
                avg_latency_points.append(0.0)
                blocking_points.append(1.0)

        return avg_latency_points, blocking_points

