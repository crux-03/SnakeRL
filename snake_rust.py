"""
Drop-in replacement for the Python SnakeEnv, backed by Rust via PyO3.

Usage in agent.py — change only the import:
    from snake_wrapper import SnakeEnv
    # (or keep using `from snake import SnakeEnv` and rename this file)

Everything else stays the same: env.step(), env.get_state(), env.get_grid_state(),
env.score, env.steps, env.action_space_size, etc.

Build instructions:
    cd snake_rs
    pip install maturin
    maturin develop --release
"""

from snake_rs import SnakeEnv as _RustEnv
from snake_rs import StepResult


class SnakeEnv:
    """Thin wrapper that preserves the original Python API."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        reward_config=None,
        max_steps_without_food: int | None = None,
        seed: int | None = None,
    ):
        # reward_config is ignored — rewards are hardcoded in Rust for speed.
        # If you need to tune them, update the constants in lib.rs and rebuild.
        self._env = _RustEnv(
            width=width,
            height=height,
            seed=seed,
            max_steps_without_food=max_steps_without_food,
        )

    def reset(self, seed: int | None = None):
        return self._env.reset(seed=seed)

    def step(self, action: int) -> StepResult:
        return self._env.step(action)

    def get_state(self):
        return self._env.get_state()

    def get_grid_state(self):
        return self._env.get_grid_state()

    def sample_action(self) -> int:
        return self._env.sample_action()

    @property
    def score(self) -> int:
        return self._env.score

    @property
    def steps(self) -> int:
        return self._env.steps

    @property
    def done(self) -> bool:
        return self._env.done

    @property
    def width(self) -> int:
        return self._env.width

    @property
    def height(self) -> int:
        return self._env.height

    @property
    def action_space_size(self) -> int:
        return self._env.action_space_size

    @property
    def state_size(self) -> int:
        return self._env.state_size

    @property
    def state_shape(self) -> tuple[int, int]:
        return self._env.state_shape
        
    @property
    def grid_channels(self) -> int:
        return self._env.grid_channels
        
    @property
    def grid_state_size(self) -> int:
        return self._env.grid_state_size
