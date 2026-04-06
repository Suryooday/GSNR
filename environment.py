"""RL environment for GSNR-aware routing and spectrum assignment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

import fiber_model
from routing import get_k_shortest_paths


@dataclass
class ActiveConnection:
    """Allocated connection tracked by the RL environment."""

    path: List[str]
    slots: List[int]
    release_step: int


class OpticalNetworkEnv:
    """Environment with discrete actions for path and slot selection."""

    def __init__(
        self,
        graph: nx.Graph,
        spectrum_manager: object,
        k_paths: int = 3,
        required_slots: int = 4,
        min_snr_db: float = 10.0,
        signal_power_w: float = 1e-3,
        noise_figure_db: float = 5.0,
        eta: float = 1e-3,
        service_rate_mu: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> None:
        self.graph = graph
        self.spectrum_manager = spectrum_manager
        self.k_paths = k_paths
        self.required_slots = required_slots
        self.min_snr_db = min_snr_db
        self.signal_power_w = signal_power_w
        self.noise_figure_db = noise_figure_db
        self.eta = eta
        self.service_rate_mu = service_rate_mu
        self.slot_width_hz = 12.5e9
        self._rng = random.Random(random_seed)

        self.nodes = sorted(list(self.graph.nodes()))
        if len(self.nodes) < 2:
            raise ValueError("Graph must contain at least 2 nodes.")

        self.total_slots = int(getattr(self.spectrum_manager, "total_slots", 0))
        if self.total_slots < self.required_slots:
            raise ValueError("Spectrum manager has fewer slots than required_slots.")

        self.max_slot_start = self.total_slots - self.required_slots
        self.action_size = self.k_paths * (self.max_slot_start + 1)

        self.current_step = 0
        self.current_src = self.nodes[0]
        self.current_dst = self.nodes[1]
        self.last_gsnr_db = 0.0
        self.active_connections: List[ActiveConnection] = []

    def reset(self) -> Tuple[int, ...]:
        self.current_step = 0
        self.last_gsnr_db = 0.0
        self.active_connections.clear()
        self._clear_all_allocations()
        self._sample_src_dst()
        return self._build_state()

    def _clear_all_allocations(self) -> None:
        for u, v in self.graph.edges():
            data = self.graph[u][v]
            occupied = data.get("occupied_slots")
            slot_power = data.get("slot_power")
            if occupied is not None and slot_power is not None:
                for idx in range(min(len(occupied), len(slot_power))):
                    occupied[idx] = False
                    slot_power[idx] = 0.0

    def _sample_src_dst(self) -> None:
        self.current_src, self.current_dst = self._rng.sample(self.nodes, 2)

    def _current_utilization(self) -> float:
        total_capacity = len(self.graph.edges()) * self.total_slots
        if total_capacity <= 0:
            return 0.0
        occupied_count = 0
        for u, v in self.graph.edges():
            occupied = self.graph[u][v].get("occupied_slots", [False] * self.total_slots)
            occupied_count += sum(1 for flag in occupied[: self.total_slots] if flag)
        return occupied_count / total_capacity

    @staticmethod
    def _bucketize(value: float, min_value: float, max_value: float, buckets: int = 10) -> int:
        if max_value <= min_value:
            return 0
        clipped = max(min_value, min(max_value, value))
        ratio = (clipped - min_value) / (max_value - min_value)
        return min(buckets - 1, int(ratio * buckets))

    def _build_state(self) -> Tuple[int, ...]:
        src_idx = self.nodes.index(self.current_src)
        dst_idx = self.nodes.index(self.current_dst)
        util_bucket = self._bucketize(self._current_utilization(), 0.0, 1.0)
        gsnr_bucket = self._bucketize(self.last_gsnr_db, -5.0, 30.0)
        active_bucket = self._bucketize(float(len(self.active_connections)), 0.0, 30.0)
        return (src_idx, dst_idx, util_bucket, gsnr_bucket, active_bucket)

    def decode_action(self, action: int) -> Tuple[int, int]:
        if action < 0 or action >= self.action_size:
            raise ValueError("Action index out of range.")
        slots_per_path = self.max_slot_start + 1
        path_index = action // slots_per_path
        slot_start = action % slots_per_path
        return path_index, slot_start

    def _tick_departures(self) -> None:
        remaining: List[ActiveConnection] = []
        for conn in self.active_connections:
            if conn.release_step <= self.current_step:
                self.spectrum_manager.release_slots(conn.path, conn.slots)
            else:
                remaining.append(conn)
        self.active_connections = remaining

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, float]]:
        """Perform one decision step.

        Action encodes:
        - path choice index among k-shortest paths
        - starting slot index for contiguous allocation
        """
        self.current_step += 1
        self._tick_departures()

        path_choice, slot_start = self.decode_action(action)
        candidate_paths = get_k_shortest_paths(
            self.graph,
            src=self.current_src,
            dst=self.current_dst,
            k=self.k_paths,
        )

        reward = -1.0
        done = False
        info: Dict[str, float] = {"success": 0.0, "blocking": 0.0, "poor_gsnr": 0.0}
        self.last_gsnr_db = 0.0

        if not candidate_paths or path_choice >= len(candidate_paths):
            reward = -4.0
            info["blocking"] = 1.0
        else:
            path = candidate_paths[path_choice]
            slots = list(range(slot_start, slot_start + self.required_slots))
            gsnr_info = fiber_model.compute_gsnr(
                graph=self.graph,
                path=path,
                channel_slots=slots,
                signal_power_w=self.signal_power_w,
                noise_figure_db=self.noise_figure_db,
                eta=self.eta,
                slot_width_hz=self.slot_width_hz,
            )
            snr_db = float(gsnr_info["snr_db"])
            self.last_gsnr_db = snr_db

            if snr_db < self.min_snr_db:
                reward = -3.0
                info["poor_gsnr"] = 1.0
            else:
                success = self.spectrum_manager.allocate_slots(path, slots, power=self.signal_power_w)
                if not success:
                    reward = -5.0
                    info["blocking"] = 1.0
                else:
                    holding_time_steps = max(1, int(self._rng.expovariate(self.service_rate_mu)))
                    self.active_connections.append(
                        ActiveConnection(
                            path=path,
                            slots=slots,
                            release_step=self.current_step + holding_time_steps,
                        )
                    )
                    reward = 10.0 + (snr_db - self.min_snr_db) * 0.1
                    info["success"] = 1.0

        self._sample_src_dst()
        next_state = self._build_state()
        return next_state, reward, done, info
