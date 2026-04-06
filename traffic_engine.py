"""Traffic engine for optical network simulation."""

from __future__ import annotations

import heapq
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrafficRequest:
    """Represents one dynamic traffic request."""

    request_id: int
    source: str
    destination: str
    arrival_time: float
    holding_time: float
    bit_rate_gbps: float


@dataclass
class ActiveConnection:
    """State needed to release an accepted request."""

    request_id: int
    path: List[str]
    slots: List[int]
    release_time: float
    latency_ms: float
    source: str = ""
    destination: str = ""
    holding_time: float = 0.0


class TrafficEngine:
    """Event-driven traffic simulator with Poisson arrivals."""

    def __init__(
        self,
        graph: nx.Graph,
        spectrum_manager: object,
        optimizer: object,
        arrival_rate_lambda: float = 1.0,
        service_rate_mu: float = 1.0,
        bit_rate_gbps: float = 100.0,
        random_seed: Optional[int] = None,
        dataset: Optional[Sequence[Union[Dict[str, object], Tuple]]] = None,
        strategy: str = "dynamic_gsnr",
    ) -> None:
        if arrival_rate_lambda <= 0.0 and not dataset:
            raise ValueError("'arrival_rate_lambda' must be > 0 if no dataset is provided.")
        if service_rate_mu <= 0.0 and not dataset:
            raise ValueError("'service_rate_mu' must be > 0 if no dataset is provided.")
        self.graph = graph
        self.spectrum_manager = spectrum_manager
        self.optimizer = optimizer
        self.arrival_rate_lambda = arrival_rate_lambda
        self.service_rate_mu = service_rate_mu
        self.bit_rate_gbps = bit_rate_gbps
        self._rng = random.Random(random_seed)
        self.dataset = list(dataset) if dataset else None
        self.dataset_index = 0
        self.strategy = strategy
        
        self.rl_agent = None
        self.rl_env = None
        if self.strategy == "rl":
            try:
                from ai.rl_agent import DQNAgent
                from ai.environment import OpticalEnv
                self.rl_env = OpticalEnv(self.graph, self.spectrum_manager)
                self.rl_agent = DQNAgent(state_dim=7) # Maps strictly to OpticalEnv.extract_condensed_state
            except ImportError:
                logger.error("Failed to import RL modules. Is torch installed?")

    def _random_src_dst(self) -> Tuple[str, str]:
        nodes = list(self.graph.nodes())
        if len(nodes) < 2:
            raise ValueError("Graph must contain at least 2 nodes.")
        src, dst = self._rng.sample(nodes, 2)
        return str(src), str(dst)

    def _exp_sample(self, rate: float) -> float:
        return self._rng.expovariate(rate)

    def _current_utilization(self) -> float:
        total_slots = int(getattr(self.spectrum_manager, "total_slots", 0))
        if total_slots <= 0:
            return 0.0

        total_capacity = len(self.graph.edges()) * total_slots
        if total_capacity == 0:
            return 0.0

        occupied_count = 0
        for u, v in self.graph.edges():
            occupied = self.graph[u][v].get("occupied_slots", [False] * total_slots)
            if len(occupied) < total_slots:
                occupied = occupied + [False] * (total_slots - len(occupied))
            occupied_count += sum(1 for flag in occupied[:total_slots] if flag)
        return occupied_count / total_capacity

    def _run_optimizer(self, source: str, destination: str) -> Optional[Dict[str, object]]:
        find_fn = getattr(self.optimizer, "find_best_assignment", None)
        if find_fn is None:
            raise AttributeError("optimizer must provide find_best_assignment(...).")

        try:
            # We enforce allocate=False initially so RL agent intercepts unassigned candidates
            allocate_flag = False if self.strategy == "rl" else True
            return find_fn(
                source=source,
                destination=destination,
                bit_rate_gbps=self.bit_rate_gbps,
                strategy=self.strategy,
                allocate=allocate_flag,
            )
        except Exception:
            logger.exception("Optimizer failed for request %s -> %s", source, destination)
            return None

    def run(self, total_requests: int) -> Dict[str, float]:
        """Run simulation for `total_requests` arrivals and return KPI metrics."""
        if total_requests < 1 and not self.dataset:
            raise ValueError("'total_requests' must be >= 1 without a dataset.")

        if self.dataset:
            total_requests = len(self.dataset)

        current_time = 0.0
        
        if self.dataset and len(self.dataset) > 0:
            first_req = self.dataset[0]
            if isinstance(first_req, (tuple, list)):
                next_arrival_time = float(first_req[3]) if len(first_req) > 3 else 0.0
            else:
                next_arrival_time = float(first_req.get("time", 0.0))
        else:
            next_arrival_time = self._exp_sample(self.arrival_rate_lambda)
            
        arrivals_seen = 0

        event_queue: List[Tuple[float, str, int]] = []
        active_connections: Dict[int, ActiveConnection] = {}

        accepted = 0
        blocked = 0
        total_latency_ms = 0.0
        total_gsnr_linear = 0.0
        total_energy_w = 0.0

        # Time-average utilization integral.
        utilization_area = 0.0
        previous_time = 0.0

        while arrivals_seen < total_requests or event_queue:
            next_departure_time = event_queue[0][0] if event_queue else float("inf")

            is_arrival = arrivals_seen < total_requests and next_arrival_time <= next_departure_time

            if is_arrival:
                current_time = next_arrival_time
            else:
                current_time = next_departure_time

            # Integrate utilization from previous event time.
            delta_t = current_time - previous_time
            if delta_t > 0.0:
                utilization_area += self._current_utilization() * delta_t
            previous_time = current_time

            if is_arrival:
                arrivals_seen += 1
                request_id = arrivals_seen

                if self.dataset:
                    req_data = self.dataset[self.dataset_index]
                    self.dataset_index += 1
                    
                    if isinstance(req_data, (tuple, list)):
                        src = str(req_data[0])
                        dst = str(req_data[1])
                        req_bit_rate = float(req_data[2]) if len(req_data) > 2 else self.bit_rate_gbps
                        holding_time = self._exp_sample(self.service_rate_mu)
                    else:
                        src = str(req_data["src"])
                        dst = str(req_data["dst"])
                        holding_time = float(req_data.get("holding_time", self._exp_sample(self.service_rate_mu)))
                        req_bit_rate = float(req_data.get("bit_rate", self.bit_rate_gbps))
                else:
                    src, dst = self._random_src_dst()
                    holding_time = self._exp_sample(self.service_rate_mu)
                    req_bit_rate = self.bit_rate_gbps

                _ = TrafficRequest(
                    request_id=request_id,
                    source=src,
                    destination=dst,
                    arrival_time=current_time,
                    holding_time=holding_time,
                    bit_rate_gbps=req_bit_rate,
                )

                result = self._run_optimizer(source=src, destination=dst)
                
                rl_state_current = None
                rl_reward = 0.0
                
                if self.strategy == "rl" and self.rl_agent is not None:
                    if result is not None and "rl_candidates" in result:
                        candidates = result["rl_candidates"]
                        import numpy as np
                        states = [
                            self.rl_env.extract_condensed_state(src, dst, c["path"], c["slots"], float(c["snr_db"]))
                            for c in candidates
                        ]
                        states_array = np.array(states)
                        
                        # Forward pass Action
                        action_idx = self.rl_agent.select_action(states_array)
                        chosen = candidates[action_idx]
                        
                        # Execute allocation
                        if self.spectrum_manager.allocate_slots(chosen["path"], chosen["slots"]):
                            result = chosen
                            rl_state_current = states[action_idx]
                        else:
                            result = None
                    else:
                        result = None

                if result is None:
                    blocked += 1
                    rl_reward = -10.0
                else:
                    accepted += 1
                    path = list(result["path"])
                    slots = list(result["slots"])
                    latency_ms = float(result.get("latency_ms", 0.0))
                    total_latency_ms += latency_ms
                    total_gsnr_linear += float(result.get("gsnr_linear", 0.0))
                    total_energy_w += float(result.get("power_w", 0.0))

                    release_time = current_time + holding_time
                    active_connections[request_id] = ActiveConnection(
                        request_id=request_id,
                        path=path,
                        slots=slots,
                        release_time=release_time,
                        latency_ms=latency_ms,
                    )
                    heapq.heappush(event_queue, (release_time, "departure", request_id))
                    
                    if self.strategy == "rl":
                        rl_reward = self.rl_env.compute_reward(True, float(result["snr_db"]), latency_ms, float(result.get("power_w", 0.0)))
                        
                if self.strategy == "rl" and self.rl_agent is not None and rl_state_current is not None:
                    import numpy as np
                    # We utilize standard MDP memory: done=True since episodes are singular routing transitions
                    self.rl_agent.remember(rl_state_current, 0, rl_reward, np.zeros_like(rl_state_current), 1)
                    self.rl_agent.optimize_model()

                if self.dataset:
                    if self.dataset_index < len(self.dataset):
                        next_req = self.dataset[self.dataset_index]
                        if isinstance(next_req, (tuple, list)):
                            next_arrival_time = float(next_req[3]) if len(next_req) > 3 else current_time
                        else:
                            next_arrival_time = float(next_req.get("time", current_time))
                    else:
                        next_arrival_time = float('inf')
                else:
                    next_arrival_time = current_time + self._exp_sample(self.arrival_rate_lambda)
            else:
                _, _, request_id = heapq.heappop(event_queue)
                conn = active_connections.pop(request_id, None)
                if conn is not None:
                    self.spectrum_manager.release_slots(conn.path, conn.slots)

        simulation_time = max(current_time, 1e-12)
        average_latency_ms = total_latency_ms / accepted if accepted > 0 else 0.0
        average_gsnr_linear = total_gsnr_linear / accepted if accepted > 0 else 0.0

        import math
        average_snr_db = 10.0 * math.log10(average_gsnr_linear) if average_gsnr_linear > 0.0 else 0.0

        return {
            "total_requests": float(total_requests),
            "accepted_requests": float(accepted),
            "blocked_requests": float(blocked),
            "blocking_probability": blocked / total_requests,
            "network_utilization": min(1.0, utilization_area / simulation_time),
            "average_latency_ms": average_latency_ms,
            "average_snr_db": average_snr_db,
            "total_energy_consumption_w": total_energy_w,
            "energy_per_bit": total_energy_w / (accepted * self.bit_rate_gbps) if accepted > 0 else 0.0,
            "simulation_time": simulation_time,
        }
