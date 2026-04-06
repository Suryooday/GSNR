import numpy as np
import networkx as nx

class OpticalEnv:
    """
    Translates Optical Simulation constraints into mathematical features for the RL Agent.
    """
    def __init__(self, graph: nx.Graph, spectrum_manager):
        self.graph = graph
        self.spectrum_manager = spectrum_manager
        self._node_idx_map = {n: i for i, n in enumerate(graph.nodes())}
        self.total_network_slots = len(graph.edges()) * spectrum_manager.total_slots
        
    def _compute_global_utilization(self) -> float:
        """Percentage of the network currently occupied."""
        if self.total_network_slots == 0:
            return 0.0
            
        occupied = 0
        for u, v in self.graph.edges():
            occupied += len(self.graph[u][v].get("occupied_slots", set()))
        return occupied / self.total_network_slots
        
    def _compute_path_distance(self, path) -> float:
        return sum(float(self.graph[path[k]][path[k+1]].get("length_km", 80.0)) for k in range(len(path)-1))
        
    def extract_condensed_state(self, src: str, dst: str, candidate_path: list, candidate_slots: list, snr_db: float) -> np.ndarray:
        """
        Builds the 7-Dimensional State + Action vector for the Q-Network.
        Features:
        0: Global Network Utilization [0.0 - 1.0]
        1: Source Node Index (normalized)
        2: Destination Node Index (normalized)
        3: Path Hop Count (normalized ~ 20 hops max)
        4: Path Distance km (normalized ~ 2000 km max)
        5: Required Slot Width
        6: Candidate SNR dB (normalized ~ 40 dB max)
        """
        utilization = self._compute_global_utilization()
        
        src_id = self._node_idx_map.get(src, 0) / max(1, len(self._node_idx_map))
        dst_id = self._node_idx_map.get(dst, 0) / max(1, len(self._node_idx_map))
        
        hops = len(candidate_path) / 20.0
        dist = self._compute_path_distance(candidate_path) / 2000.0
        
        slot_width = len(candidate_slots) / 320.0
        norm_snr = snr_db / 40.0
        
        return np.array([
            utilization,
            src_id,
            dst_id,
            hops,
            dist,
            slot_width,
            norm_snr
        ], dtype=np.float32)

    def compute_reward(self, is_accepted: bool, snr_db: float, latency_ms: float, energy_w: float) -> float:
        """
        Maps physical simulation returns into Reinforcement scalar reward.
        """
        if not is_accepted:
            return -10.0 # Heavy penalty for blocking the connection
            
        reward = 10.0 # Base success
        
        # Bonus for high SNR (Scale 0 to 5)
        snr_bonus = min(5.0, max(0.0, (snr_db - 15.0) / 5.0))
        reward += snr_bonus
        
        # Penalty for extremely high latency (+10ms)
        if latency_ms > 10.0:
            reward -= min(5.0, (latency_ms - 10.0) * 0.5)
            
        # Penalty for aggressive energy usage 
        # (Assuming typical block is ~300W using 2 transponders, punish anything above)
        if energy_w > 500.0:
            reward -= min(3.0, (energy_w - 500.0) / 100.0)
            
        return reward
