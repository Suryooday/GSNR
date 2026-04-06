"""Integration tests for API -> Core flow."""

from __future__ import annotations

import unittest

from api_service import SimulationAPI, SimulatorConfig


class TestSimulationAPI(unittest.TestCase):
    def test_build_and_nodes(self) -> None:
        api = SimulationAPI(SimulatorConfig(number_of_nodes=10))
        api.build()
        nodes = api.get_nodes()
        self.assertGreaterEqual(len(nodes), 2)

    def test_compute_path_result_shape(self) -> None:
        api = SimulationAPI(SimulatorConfig(number_of_nodes=10))
        api.build()
        nodes = api.get_nodes()
        result = api.compute_path(nodes[0], nodes[-1])

        if result is not None:
            self.assertIn("path", result)
            self.assertIn("slots", result)
            self.assertIn("snr_db", result)
            self.assertIn("ber", result)
            self.assertIn("latency_ms", result)


if __name__ == "__main__":
    unittest.main()
