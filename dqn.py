import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_hidden=2) -> None:
        super(DQN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dim))

        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)


class ConvDQN(nn.Module):
    def __init__(self, grid_size, action_dim, num_channels=3):
        super().__init__()
        self.grid_size = grid_size
        self.num_channels = num_channels

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        reduced = grid_size // 4
        fc_input = 64 * reduced * reduced

        self.fc = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.num_channels, self.grid_size, self.grid_size)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class HybridDQN(nn.Module):
    def __init__(
        self,
        grid_size,
        state_dim,
        action_dim,
        num_channels=3,
        hidden_dim=128,
        num_hidden=2,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.state_dim = state_dim
        self.grid_dim = num_channels * grid_size * grid_size

        # Conv stream (no Q-value head)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        reduced = grid_size // 4
        conv_out = 64 * reduced * reduced

        # MLP stream (no Q-value head)
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Linear(state_dim, hidden_dim))
        for _ in range(num_hidden - 1):
            self.mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Shared head merges both streams
        self.head = nn.Sequential(
            nn.Linear(conv_out + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        """Accept a single concatenated tensor [grid_flat | vec_state] and split internally."""
        grid_state = x[:, : self.grid_dim]
        vec_state = x[:, self.grid_dim :]

        # Conv stream
        cx = grid_state.view(-1, self.num_channels, self.grid_size, self.grid_size)
        cx = self.conv(cx)
        cx = cx.flatten(start_dim=1)

        # MLP stream
        mx = vec_state
        for layer in self.mlp_layers:
            mx = F.relu(layer(mx))

        return self.head(torch.cat([cx, mx], dim=1))


if __name__ == "__main__":
    state_dim = 12
    action_dim = 4
    net = DQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)
