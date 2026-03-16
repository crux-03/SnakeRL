import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn

from dqn import DQN, ConvDQN, HybridDQN
from logger import Logger
from snake_rust import SnakeEnv

device = "cuda" if torch.cuda.is_available() else "cpu"


class TensorReplayBuffer:
    """Pre-allocated circular replay buffer stored as contiguous pinned tensors.

    Eliminates per-step:
        - tuple construction
        - torch.stack() during sampling
        - Python list overhead

    All data lives in pinned CPU memory for fast async GPU transfers.
    Sampling is a single index_select on each tensor.
    """

    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.size = 0
        self.pos = 0

        # Pre-allocate contiguous pinned tensors
        self.states = torch.zeros(capacity, state_size, pin_memory=True)
        self.actions = torch.zeros(capacity, dtype=torch.int64, pin_memory=True)
        self.next_states = torch.zeros(capacity, state_size, pin_memory=True)
        self.rewards = torch.zeros(capacity, pin_memory=True)
        self.dones = torch.zeros(capacity, pin_memory=True)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        reward: float,
        done: bool,
    ):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.next_states[i] = next_state
        self.rewards[i] = reward
        self.dones[i] = float(done)

        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Returns (states, actions, next_states, rewards, dones) already on GPU."""
        indices = torch.randint(0, self.size, (batch_size,))

        # index_select on contiguous pinned memory → fast async transfer
        return (
            self.states[indices].to(device, non_blocking=True),
            self.actions[indices].to(device, non_blocking=True),
            self.next_states[indices].to(device, non_blocking=True),
            self.rewards[indices].to(device, non_blocking=True),
            self.dones[indices].to(device, non_blocking=True),
        )

    def __len__(self):
        return self.size


class Agent:
    def __init__(self, logger: Logger, network_type: str = "mlp") -> None:
        if network_type not in ("mlp", "conv", "hybrid"):
            raise ValueError(
                f"Unknown network_type '{network_type}'. Use 'mlp', 'conv', or 'hybrid'."
            )
        self.network_type = network_type

        with open("hyperparams.yml", "r") as file:
            all_param_sets = yaml.safe_load(file)
            hyperparams = all_param_sets[f"snake_{network_type}"]
        self.logger = logger

        self.epochs = hyperparams["epochs"]
        self.target_dqn_sync_interval = hyperparams["target_dqn_sync_interval"]
        self.replay_memory_size = hyperparams["replay_memory_size"]
        self.mini_batch_size = hyperparams["mini_batch_size"]
        self.epsilon_init = hyperparams["epsilon_init"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.learning_rate_a = hyperparams["learning_rate_a"]
        self.discount_factor_g = hyperparams["discount_factor_g"]

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None
        self.last_loss = 0.0

        os.makedirs("models", exist_ok=True)

    def train(self, layer_size=256, layer_count=2):
        env = SnakeEnv()

        num_channels = env.grid_channels
        grid_dim = env.grid_state_size
        vec_dim = env.state_size

        if self.network_type == "conv":
            policy_dqn = ConvDQN(env.height, env.action_space_size, num_channels).to(
                device
            )
            target_dqn = ConvDQN(env.height, env.action_space_size, num_channels).to(
                device
            )
            get_state = env.get_grid_state
            state_dim = grid_dim

        elif self.network_type == "hybrid":
            policy_dqn = HybridDQN(
                env.height,
                vec_dim,
                env.action_space_size,
                num_channels,
                layer_size,
                layer_count,
            ).to(device)
            target_dqn = HybridDQN(
                env.height,
                vec_dim,
                env.action_space_size,
                num_channels,
                layer_size,
                layer_count,
            ).to(device)
            # State is concatenated [grid_flat | vec_state]
            state_dim = grid_dim + vec_dim

            def get_state():
                grid = env.get_grid_state()
                vec = env.get_state()
                return np.concatenate([grid, vec])

        else:
            policy_dqn = DQN(
                vec_dim, env.action_space_size, layer_size, layer_count
            ).to(device)
            target_dqn = DQN(
                vec_dim, env.action_space_size, layer_size, layer_count
            ).to(device)
            get_state = env.get_state
            state_dim = vec_dim

        target_dqn.load_state_dict(policy_dqn.state_dict())

        rewards_per_episode = []
        epsilon_history = []

        memory = TensorReplayBuffer(self.replay_memory_size, state_dim)

        epsilon = self.epsilon_init
        step_count = 0
        log_interval = 100

        self.optimizer = torch.optim.AdamW(
            policy_dqn.parameters(), lr=self.learning_rate_a
        )

        recent_scores = []
        recent_steps = []
        recent_rewards = []

        best_avg_reward = float("-inf")

        # Pre-allocate a CPU tensor for state to avoid repeated allocation.
        # torch.from_numpy shares memory — we copy into this once, reuse it.
        state_cpu = torch.empty(state_dim, dtype=torch.float32)
        new_state_cpu = torch.empty(state_dim, dtype=torch.float32)

        for episode in range(self.epochs):
            start = time.time()
            env.reset()

            # Zero-copy numpy → torch on CPU, then async transfer for inference
            np_state = get_state()
            state_cpu.copy_(torch.from_numpy(np_state))

            episode_reward = 0.0
            interrupted = False

            while not interrupted:
                # Action selection
                if random.random() < epsilon:
                    action = env.sample_action()
                else:
                    with torch.no_grad():
                        # Only transfer state to GPU when we actually need inference
                        state_gpu = state_cpu.to(device, non_blocking=True)
                        action = (
                            policy_dqn(state_gpu.unsqueeze(0)).squeeze().argmax().item()
                        )

                result = env.step(action)
                episode_reward += result.reward
                interrupted = result.done

                # Get new state directly into CPU tensor
                np_new_state = get_state()
                new_state_cpu.copy_(torch.from_numpy(np_new_state))

                # Store in replay buffer — no GPU involvement, no tuple, no tensor creation
                memory.push(
                    state_cpu, action, new_state_cpu, result.reward, result.done
                )
                step_count += 1

                # Swap buffers (just swap references, no copy)
                state_cpu, new_state_cpu = new_state_cpu, state_cpu

                if step_count % 8 == 0 and len(memory) > self.mini_batch_size:
                    self.optimize(memory, policy_dqn, target_dqn)

                    if step_count > self.target_dqn_sync_interval:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

            rewards_per_episode.append(episode_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            recent_scores.append(env.score)
            recent_steps.append(env.steps)
            recent_rewards.append(episode_reward)

            if (episode + 1) % log_interval == 0:
                avg_score = np.mean(recent_scores)
                avg_steps = np.mean(recent_steps)
                avg_reward = np.mean(recent_rewards)

                self.logger.log_epoch(
                    episode + 1,
                    self.epochs,
                    self.last_loss,
                    epsilon,
                    avg_score,  # pyright: ignore[reportArgumentType]
                    avg_steps,  # pyright: ignore[reportArgumentType]
                    avg_reward,  # pyright: ignore[reportArgumentType]
                    time.time() - start,
                )

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(
                        policy_dqn.state_dict(),
                        f"models/snake_best_{self.network_type}.pt",
                    )

                    with open(f"models/best_{self.network_type}.txt", "w") as f:
                        f.write(f"Episode: {episode + 1}/{self.epochs}\n")
                        f.write(f"Loss: {self.last_loss:.4f}\n")
                        f.write(f"Epsilon: {epsilon:.4f}\n")
                        f.write(f"Avg Score: {avg_score:.2f}\n")
                        f.write(f"Avg Steps: {avg_steps:.1f}\n")
                        f.write(f"Avg Reward: {avg_reward:.2f}\n")

                recent_scores.clear()
                recent_steps.clear()
                recent_rewards.clear()

        window = 10000
        smoothed = np.convolve(
            rewards_per_episode, np.ones(window) / window, mode="valid"
        )

        torch.save(
            policy_dqn.state_dict(), f"models/snake_final_{self.network_type}.pt"
        )

        plt.plot(smoothed)
        plt.ylabel("Rewards (moving avg)")
        plt.show()

    def optimize(self, memory: TensorReplayBuffer, policy_dqn, target_dqn):
        # Sample returns tensors already on GPU via pinned memory async transfer
        states, actions, new_states, rewards, dones = memory.sample(
            self.mini_batch_size
        )

        with torch.no_grad():
            best_actions = policy_dqn(new_states).argmax(dim=1)
            target_q = (
                rewards
                + (1 - dones)
                * self.discount_factor_g
                * target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            )

        current_q = (
            policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()  # pyright: ignore[reportOptionalMemberAccess]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10)
        self.optimizer.step()  # pyright: ignore[reportOptionalMemberAccess]
        self.last_loss = loss.item()


class DebugLogger(Logger):
    def log_epoch(
        self,
        curr: int,
        max: int,
        loss: float,
        epsilon: float,
        avg_score: float,
        avg_steps: float,
        avg_reward: float,
        time: float,
    ):
        avg_ms = time * 1000
        print(
            f"[DEBUG] Episode {curr}/{max} | "
            f"Loss: {loss:.4f} | "
            f"Epsilon: {epsilon:.4f} | "
            f"Avg Score: {avg_score:.2f} | "
            f"Avg Steps: {avg_steps:.1f} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"{avg_ms:.5f}ms ({avg_ms / avg_steps:.5f}ms per step)"
        )


if __name__ == "__main__":
    import sys

    network_type = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    logger = DebugLogger()
    agent = Agent(logger, network_type=network_type)
    agent.train(layer_size=256, layer_count=4 if network_type == "conv" else 2)
