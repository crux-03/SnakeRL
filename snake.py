"""
Snake environment for reinforcement learning (optimized).

Gym-style interface:
    env = SnakeEnv(width=20, height=20)
    state = env.reset()
    state, reward, done, info = env.step(action)

Actions:
    0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT

State representation (via get_state()):
    A numpy array of 19 floats (normalized 0-1):
        [0]  head_y
        [1]  head_x
        [2]  food_y
        [3]  food_x
        [4]  manhattan distance to food
        [5]  distance to wall UP
        [6]  distance to wall DOWN
        [7]  distance to wall LEFT
        [8]  distance to wall RIGHT
        [9]  snake length
        [10] current direction (0-3 mapped to 0-1)
        [11-14] danger flags (up, down, left, right)
        [15-18] body distance (up, down, left, right)

Rewards (configurable via RewardConfig):
    food_reward:    +50  (ate food)
    death_reward:   -10  (hit wall or self)
    step_reward:    -0.001 (each step, encourages efficiency)
    closer_reward:  +0.5  (moved closer to food)
    farther_reward: -0.1  (moved away from food)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# Direction vectors (dy, dx) — plain tuple lookups indexed by action int
_DY = (-1, 1, 0, 0)
_DX = (0, 0, -1, 1)

# Opposite action lookup indexed by action int
_OPPOSITE = (1, 0, 3, 2)  # UP<->DOWN, LEFT<->RIGHT


@dataclass
class RewardConfig:
    """Tune these to shape agent behavior."""

    food_reward: float = 50.0
    death_reward: float = -10.0
    step_reward: float = -0.001
    closer_reward: float = 0.5
    farther_reward: float = -0.1


@dataclass
class StepResult:
    """Returned by env.step()."""

    state: Any
    reward: float
    done: bool
    info: dict[str, Any]


class SnakeEnv:
    """
    Snake game environment with gym-style interface.

    Optimizations vs. original:
        - deque for O(1) head insert / tail pop
        - persistent snake_set updated incrementally (no rebuild per step)
        - pre-allocated numpy state buffer (no allocation per get_state call)
        - rejection-sampling food spawn (avoids 400-cell list comprehension)
        - _make_result no longer calls get_state() (caller does it anyway)
        - direction map / opposite map as plain tuples indexed by int (no dict lookup)
        - body-distance scans use cached snake_set
    """

    __slots__ = (
        "width",
        "height",
        "rewards",
        "max_steps_without_food",
        "rng",
        "snake",
        "snake_set",
        "direction",
        "food",
        "score",
        "done",
        "steps",
        "steps_since_food",
        "_state_buf",
        "_grid_buf",
        "_inv_h",
        "_inv_w",
        "_inv_max_dist",
        "_inv_max_len",
        "_inv_max_dim",
        "_max_rejection_attempts",
    )

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        reward_config: RewardConfig | None = None,
        max_steps_without_food: int | None = None,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        self.rewards = reward_config or RewardConfig()
        self.max_steps_without_food = max_steps_without_food or (width * height)
        self.rng = random.Random(seed)

        # Pre-computed reciprocals to replace divisions with multiplications
        self._inv_h = 1.0 / (height - 1)
        self._inv_w = 1.0 / (width - 1)
        self._inv_max_dist = 1.0 / (height + width)
        self._inv_max_len = 1.0 / (width * height)
        self._inv_max_dim = 1.0 / max(height, width)

        # Rejection sampling limit before falling back to full scan
        self._max_rejection_attempts = 100

        # Pre-allocated output buffers
        self._state_buf = np.empty(19, dtype=np.float32)
        self._grid_buf = np.empty(height * width, dtype=np.float32)

        # State — initialized in reset()
        self.snake: deque[tuple[int, int]] = deque()
        self.snake_set: set[tuple[int, int]] = set()
        self.direction: int = Action.RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.score: int = 0
        self.done: bool = True
        self.steps: int = 0
        self.steps_since_food: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Any:
        """Reset the environment and return the initial state."""
        if seed is not None:
            self.rng = random.Random(seed)

        mid_y = self.height // 2
        mid_x = self.width // 2
        initial = [(mid_y, mid_x), (mid_y, mid_x - 1), (mid_y, mid_x - 2)]

        self.snake = deque(initial)
        self.snake_set = set(initial)
        self.direction = Action.RIGHT
        self.food = self._spawn_food()
        self.score = 0
        self.done = False
        self.steps = 0
        self.steps_since_food = 0

        return self.get_state()

    def step(self, action: int) -> StepResult:
        """Execute one step and return StepResult."""
        if self.done:
            raise RuntimeError("Episode is over. Call reset() first.")

        self.steps += 1
        self.steps_since_food += 1

        direction = self.direction

        # Prevent 180° reversal
        if _OPPOSITE[action] != direction:
            direction = action
            self.direction = action

        # New head position (inlined direction lookup)
        head = self.snake[0]
        new_head = (head[0] + _DY[direction], head[1] + _DX[direction])

        # Distance to food before move (inlined manhattan)
        food = self.food
        old_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

        # --- Collision checks ---
        ny, nx = new_head

        # Wall collision
        if not (0 <= ny < self.height and 0 <= nx < self.width):
            self.done = True
            return self._make_result(self.rewards.death_reward, "wall")

        # Self collision (O(1) with cached set)
        if new_head in self.snake_set:
            self.done = True
            return self._make_result(self.rewards.death_reward, "self")

        # --- Move snake ---
        reward = self.rewards.step_reward
        self.snake.appendleft(new_head)  # O(1) vs list.insert(0) which is O(n)
        self.snake_set.add(new_head)

        if new_head == food:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            reward += self.rewards.food_reward
        else:
            # Remove tail
            tail = self.snake.pop()  # O(1)
            self.snake_set.discard(tail)

            # Distance shaping
            new_dist = abs(ny - food[0]) + abs(nx - food[1])
            if new_dist < old_dist:
                reward += self.rewards.closer_reward
            else:
                reward += self.rewards.farther_reward

        # Starvation check
        if self.steps_since_food >= self.max_steps_without_food:
            self.done = True
            return self._make_result(self.rewards.death_reward, "starvation")

        return self._make_result(reward)

    def get_state(self) -> np.ndarray:
        """Write state into pre-allocated buffer and return it.

        NOTE: The returned array is a reference to an internal buffer.
        torch.tensor() copies the data, so this is safe in the training loop.
        If you need a persistent copy, call .copy() on the result.
        """
        head_y, head_x = self.snake[0]
        food_y, food_x = self.food
        snake_set = self.snake_set
        inv_h = self._inv_h
        inv_w = self._inv_w
        height = self.height
        width = self.width

        buf = self._state_buf

        buf[0] = head_y * inv_h
        buf[1] = head_x * inv_w
        buf[2] = food_y * inv_h
        buf[3] = food_x * inv_w
        buf[4] = (abs(head_y - food_y) + abs(head_x - food_x)) * self._inv_max_dist
        buf[5] = head_y * inv_h
        buf[6] = (height - 1 - head_y) * inv_h
        buf[7] = head_x * inv_w
        buf[8] = (width - 1 - head_x) * inv_w
        buf[9] = len(self.snake) * self._inv_max_len
        buf[10] = self.direction / 3.0

        # Danger flags: wall or body in adjacent cell
        buf[11] = float(head_y == 0 or (head_y - 1, head_x) in snake_set)
        buf[12] = float(head_y == height - 1 or (head_y + 1, head_x) in snake_set)
        buf[13] = float(head_x == 0 or (head_y, head_x - 1) in snake_set)
        buf[14] = float(head_x == width - 1 or (head_y, head_x + 1) in snake_set)

        # Body distance scans (using cached set — O(1) per check)
        inv_max_dim = self._inv_max_dim
        buf[15] = self._body_dist_scan(head_y, head_x, -1, 0, snake_set, inv_max_dim)
        buf[16] = self._body_dist_scan(head_y, head_x, 1, 0, snake_set, inv_max_dim)
        buf[17] = self._body_dist_scan(head_y, head_x, 0, -1, snake_set, inv_max_dim)
        buf[18] = self._body_dist_scan(head_y, head_x, 0, 1, snake_set, inv_max_dim)

        return buf

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _body_dist_scan(
        self,
        head_y: int,
        head_x: int,
        dy: int,
        dx: int,
        snake_set: set,
        inv_max_dim: float,
    ) -> float:
        """Scan in a direction for the nearest body segment. Returns normalized distance."""
        y = head_y + dy
        x = head_x + dx
        steps = 1
        h = self.height
        w = self.width
        while 0 <= y < h and 0 <= x < w:
            if (y, x) in snake_set:
                return steps * inv_max_dim
            y += dy
            x += dx
            steps += 1
        return 1.0

    def get_grid_state(self) -> np.ndarray:
        """Return the board as a flat float32 array (H*W).

        Encoding (normalized 0-1):
            0.0  = empty
            0.33 = snake body
            0.66 = snake head
            1.0  = food
        """
        buf = self._grid_buf
        buf[:] = 0.0

        w = self.width
        head_y, head_x = self.snake[0]
        for y, x in self.snake:
            buf[y * w + x] = 0.33
        buf[head_y * w + head_x] = 0.66
        buf[self.food[0] * w + self.food[1]] = 1.0

        return buf

    @property
    def grid_state_size(self) -> int:
        return self.height * self.width

    @property
    def action_space_size(self) -> int:
        return 4

    @property
    def state_size(self) -> int:
        return 19

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def sample_action(self) -> int:
        return self.rng.randint(0, 3)

    def _spawn_food(self) -> tuple[int, int]:
        """Spawn food using rejection sampling, falling back to full scan if board is crowded."""
        rng = self.rng
        h = self.height
        w = self.width
        snake_set = self.snake_set

        # Fast path: rejection sampling (very efficient when board is mostly empty)
        for _ in range(self._max_rejection_attempts):
            pos = (rng.randint(0, h - 1), rng.randint(0, w - 1))
            if pos not in snake_set:
                return pos

        # Slow fallback for nearly-full boards
        empty_cells = [
            (r, c) for r in range(h) for c in range(w) if (r, c) not in snake_set
        ]
        if not empty_cells:
            return self.snake[0]
        return rng.choice(empty_cells)

    def _make_result(self, reward: float, death_cause: str | None = None) -> StepResult:
        info: dict[str, Any] = {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
        }
        if death_cause:
            info["death_cause"] = death_cause
        # NOTE: state is set to None to avoid redundant get_state() call.
        # The training loop calls get_state() / get_grid_state() separately.
        return StepResult(
            state=None,
            reward=reward,
            done=self.done,
            info=info,
        )
