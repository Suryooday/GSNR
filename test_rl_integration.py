"""Integration tests for environment and Q-learning agent."""

from __future__ import annotations

import unittest

from environment import OpticalNetworkEnv
from rl_agent import QLearningAgent, train_q_learning
from spectrum_manager import SpectrumManager
from topology import OpticalTopology


class TestRLIntegration(unittest.TestCase):
    def test_q_learning_training_loop(self) -> None:
        topology = OpticalTopology()
        topology.generate_leaf_spine(spines=2, leafs=4)
        spectrum_manager = SpectrumManager(topology.graph, total_slots=64)

        env = OpticalNetworkEnv(
            graph=topology.graph,
            spectrum_manager=spectrum_manager,
            k_paths=2,
            required_slots=3,
            random_seed=7,
        )
        agent = QLearningAgent(action_size=env.action_size, random_seed=7)
        stats = train_q_learning(env=env, agent=agent, episodes=3, steps_per_episode=15)

        self.assertEqual(len(stats.episode_rewards), 3)
        self.assertEqual(len(stats.episode_success_rate), 3)
        self.assertEqual(len(stats.episode_blocking_rate), 3)


if __name__ == "__main__":
    unittest.main()
