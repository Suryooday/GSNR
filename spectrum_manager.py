"""Spectrum manager for elastic optical networks (flex-grid)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import networkx as nx

from topology import _DEFAULT_TOPOLOGY


class SpectrumManager:
    """Manage spectrum occupancy and power per link for a NetworkX topology."""

    def __init__(self, graph: nx.Graph, total_slots: int = 320) -> None:
        if total_slots < 1:
            raise ValueError("'total_slots' must be >= 1.")
        self.graph = graph
        self.total_slots = total_slots
        self._initialize_all_links()

    def _initialize_all_links(self) -> None:
        for u, v in self.graph.edges():
            self._ensure_link_state(u, v)

    def _ensure_link_state(self, u: str, v: str) -> None:
        data = self.graph[u][v]
        if "occupied_slots" not in data:
            data["occupied_slots"] = [False] * self.total_slots
        if "slot_power" not in data:
            data["slot_power"] = [0.0] * self.total_slots
        self._sync_available_slots(data)

    @staticmethod
    def _path_edges(path: Sequence[str]) -> List[Tuple[str, str]]:
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 nodes.")
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def _validate_path(self, path: Sequence[str]) -> List[Tuple[str, str]]:
        edges = self._path_edges(path)
        for u, v in edges:
            if not self.graph.has_edge(u, v):
                raise ValueError(f"Edge ({u}, {v}) is not present in the topology graph.")
            self._ensure_link_state(u, v)
        return edges

    def _normalize_slots(self, slots: Iterable[int]) -> List[int]:
        slot_list = sorted(set(int(s) for s in slots))
        if not slot_list:
            raise ValueError("At least one slot index is required.")
        if slot_list[0] < 0 or slot_list[-1] >= self.total_slots:
            raise ValueError("Slot index out of range.")
        return slot_list

    @staticmethod
    def _is_contiguous(slot_list: Sequence[int]) -> bool:
        return all(curr == prev + 1 for prev, curr in zip(slot_list, slot_list[1:]))

    @staticmethod
    def _sync_available_slots(data: dict) -> None:
        occupied = data["occupied_slots"]
        data["available_spectrum_slots"] = [idx for idx, used in enumerate(occupied) if not used]

    def find_contiguous_slots(
        self,
        required_slots: int,
        path: Sequence[str] | None = None,
    ) -> List[int]:
        """Find first-fit contiguous free slots.

        If `path` is provided, continuity is enforced across that path.
        If `path` is None, it finds slots free across all current links.
        """
        if required_slots < 1:
            raise ValueError("'required_slots' must be >= 1.")
        if path is None:
            edges = list(self.graph.edges())
            for u, v in edges:
                self._ensure_link_state(u, v)
        else:
            edges = self._validate_path(path)

        common_free = [True] * self.total_slots
        for u, v in edges:
            occupied = self.graph[u][v]["occupied_slots"]
            for idx in range(self.total_slots):
                if occupied[idx]:
                    common_free[idx] = False

        run_start = -1
        run_len = 0
        for idx, is_free in enumerate(common_free):
            if is_free:
                if run_start == -1:
                    run_start = idx
                run_len += 1
                if run_len == required_slots:
                    return list(range(run_start, run_start + required_slots))
            else:
                run_start = -1
                run_len = 0
        return []

    def allocate_slots(self, path: Sequence[str], slots: Iterable[int], power: float = 1.0) -> bool:
        """Allocate contiguous slots on all links along `path`.

        Returns:
            True if allocation succeeds. False when any slot is unavailable.
        """
        edges = self._validate_path(path)
        slot_list = self._normalize_slots(slots)
        if not self._is_contiguous(slot_list):
            raise ValueError("Slots must be contiguous for flex-grid allocation.")

        for u, v in edges:
            occupied = self.graph[u][v]["occupied_slots"]
            if any(occupied[idx] for idx in slot_list):
                return False

        for u, v in edges:
            data = self.graph[u][v]
            for idx in slot_list:
                data["occupied_slots"][idx] = True
                data["slot_power"][idx] = float(power)
            self._sync_available_slots(data)
        return True

    def release_slots(self, path: Sequence[str], slots: Iterable[int]) -> None:
        """Release previously allocated slots on every link in `path`."""
        edges = self._validate_path(path)
        slot_list = self._normalize_slots(slots)

        for u, v in edges:
            data = self.graph[u][v]
            for idx in slot_list:
                data["occupied_slots"][idx] = False
                data["slot_power"][idx] = 0.0
            self._sync_available_slots(data)


_DEFAULT_SPECTRUM_MANAGER = SpectrumManager(_DEFAULT_TOPOLOGY.graph)


def find_contiguous_slots(required_slots: int, path: Sequence[str] | None = None) -> List[int]:
    """Module-level helper using default topology graph."""
    return _DEFAULT_SPECTRUM_MANAGER.find_contiguous_slots(required_slots, path)


def allocate_slots(path: Sequence[str], slots: Iterable[int]) -> bool:
    """Module-level helper using default topology graph."""
    return _DEFAULT_SPECTRUM_MANAGER.allocate_slots(path, slots)


def release_slots(path: Sequence[str], slots: Iterable[int]) -> None:
    """Module-level helper using default topology graph."""
    _DEFAULT_SPECTRUM_MANAGER.release_slots(path, slots)
