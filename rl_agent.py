"""Reinforcement learning agents for optical RSA decisions."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

import numpy as np


@dataclass
class TrainingStats:
    """Aggregated training traces."""

    episode_rewards: List[float]
    episode_success_rate: List[float]
    episode_blocking_rate: List[float]


class QLearningAgent:
    """Tabular Q-learning baseline agent."""

    def __init__(
        self,
        action_size: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        random_seed: int | None = None,
    ) -> None:
        if action_size < 1:
            raise ValueError("'action_size' must be >= 1.")
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = random.Random(random_seed)

        self.q_table: DefaultDict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_size, dtype=float)
        )

    def select_action(self, state: Tuple[int, ...]) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.action_size)
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
    ) -> None:
        current_q = self.q_table[state][action]
        next_max_q = float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * next_max_q
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(
    env: object,
    agent: QLearningAgent,
    episodes: int = 300,
    steps_per_episode: int = 100,
) -> TrainingStats:
    """Train tabular Q-learning agent on the optical environment."""
    if episodes < 1 or steps_per_episode < 1:
        raise ValueError("'episodes' and 'steps_per_episode' must be >= 1.")

    rewards_trace: List[float] = []
    success_trace: List[float] = []
    blocking_trace: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        successes = 0.0
        blockings = 0.0

        for _step in range(steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, _done, info = env.step(action)
            agent.update(state, action, reward, next_state)

            state = next_state
            episode_reward += reward
            successes += float(info.get("success", 0.0))
            blockings += float(info.get("blocking", 0.0))

        agent.decay_epsilon()
        rewards_trace.append(episode_reward)
        success_trace.append(successes / steps_per_episode)
        blocking_trace.append(blockings / steps_per_episode)

    return TrainingStats(
        episode_rewards=rewards_trace,
        episode_success_rate=success_trace,
        episode_blocking_rate=blocking_trace,
    )


class DQNAgentPlaceholder:
    """Scaffold for future PyTorch DQN implementation."""

    def __init__(self, state_size: int, action_size: int) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.backend = "pytorch"

    def notes(self) -> Dict[str, str]:
        return {
            "status": "placeholder",
            "next_steps": (
                "Implement policy/target networks, replay buffer, "
                "mini-batch optimization, and target synchronization."
            ),
        }
