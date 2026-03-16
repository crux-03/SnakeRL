"""
Snake AI Viewer — load a trained DQN model and watch it play.

Usage:
    python watch.py                          # MLP model, loads models/snake_best_mlp.pt
    python watch.py --type conv              # Conv model, loads models/snake_best_conv.pt
    python watch.py --type hybrid            # Hybrid model, loads models/snake_best_hybrid.pt
    python watch.py --model path/to/model.pt
    python watch.py --speed 10               # frames per second
    python watch.py --layer-size 256 --layer-count 2  # MLP architecture params
"""

import argparse
import os
import sys

import numpy as np
import pygame
import pygame._freetype as _ft
import torch

from dqn import DQN, ConvDQN, HybridDQN
from snake_rs import SnakeEnv

# ── Colours ──────────────────────────────────────────────────────────
BG = (15, 15, 25)
GRID_BG = (22, 22, 35)
GRID_LINE = (35, 35, 50)
SNAKE_HEAD = (0, 220, 120)
SNAKE_BODY = (0, 180, 100)
SNAKE_TAIL = (0, 140, 80)
FOOD_COL = (255, 60, 80)
FOOD_GLOW = (255, 60, 80, 60)
PANEL_BG = (20, 20, 32)
TEXT_PRIMARY = (230, 230, 240)
TEXT_SECONDARY = (140, 140, 165)
TEXT_ACCENT = (80, 200, 255)
BAR_POS = (50, 205, 120)
BAR_NEG = (220, 60, 80)
BAR_NEUTRAL = (60, 60, 80)
BAR_BG = (35, 35, 50)
DIVIDER = (45, 45, 65)
DANGER_ON = (255, 70, 70)
DANGER_OFF = (40, 60, 40)
QVAL_BEST = (80, 220, 160)
QVAL_OTHER = (70, 70, 95)
DEATH_OVERLAY = (200, 50, 60)

DIRECTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

device = "cuda" if torch.cuda.is_available() else "cpu"


def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class _FontWrapper:
    """Thin wrapper so freetype fonts behave like the old pygame.font API."""

    def __init__(self, ft_font):
        self._f = ft_font

    def render(self, text, _antialias_ignored=True, fgcolor=(255, 255, 255)):
        surf, rect = self._f.render(text, fgcolor)
        return surf


class Viewer:
    def __init__(
        self, model_path, network_type="mlp", layer_size=256, layer_count=2, fps=8
    ):
        self.fps = fps
        self.network_type = network_type
        self.env = SnakeEnv()

        num_channels = self.env.grid_channels

        if network_type == "conv":
            self.model = ConvDQN(
                self.env.height, self.env.action_space_size, num_channels
            ).to(device)
            self.get_state = self.env.get_grid_state

        elif network_type == "hybrid":
            self.model = HybridDQN(
                self.env.height,
                self.env.state_size,
                self.env.action_space_size,
                num_channels,
                layer_size,
                layer_count,
            ).to(device)

            def _get_hybrid_state():
                grid = self.env.get_grid_state()
                vec = self.env.get_state()
                return np.concatenate([grid, vec])

            self.get_state = _get_hybrid_state

        else:
            self.model = DQN(
                self.env.state_size, self.env.action_space_size, layer_size, layer_count
            ).to(device)
            self.get_state = self.env.get_state

        self.model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self.model.eval()

        # ── Layout ───────────────────────────────────────────────────
        self.cell = 28
        self.grid_px = self.cell * self.env.width  # 560
        self.panel_w = 340
        self.padding = 24
        self.grid_origin = (self.padding, self.padding)
        self.win_w = (
            self.padding + self.grid_px + self.padding + self.panel_w + self.padding
        )
        self.win_h = self.padding + self.grid_px + self.padding

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Snake AI Viewer")
        self.clock = pygame.time.Clock()

        _ft.init()
        _font_path = os.path.join(os.path.dirname(pygame.__file__), "freesansbold.ttf")
        self.font_lg = _FontWrapper(_ft.Font(_font_path, 20))
        self.font_md = _FontWrapper(_ft.Font(_font_path, 14))
        self.font_sm = _FontWrapper(_ft.Font(_font_path, 12))
        self.font_xs = _FontWrapper(_ft.Font(_font_path, 10))
        self.font_arrow = _FontWrapper(_ft.Font(_font_path, 24))

        # Stats across episodes
        self.episode = 0
        self.best_score = 0
        self.scores_history = []

        # Pause / step
        self.paused = False
        self.step_once = False

    # ── Main loop ────────────────────────────────────────────────────
    def run(self):
        while True:
            self._play_episode()

    def _play_episode(self):
        self.env.reset()
        state_np = self.get_state()
        state = torch.tensor(state_np, dtype=torch.float, device=device)
        self.episode += 1
        total_reward = 0.0
        alive = True
        q_np = None
        action = None

        while alive:
            # Events
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE or ev.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    if ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if ev.key == pygame.K_RIGHT:
                        self.step_once = True
                    if ev.key == pygame.K_UP:
                        self.fps = min(60, self.fps + 2)
                    if ev.key == pygame.K_DOWN:
                        self.fps = max(1, self.fps - 2)

            if self.paused and not self.step_once:
                self.clock.tick(30)
                self._draw(state_np, None, total_reward, alive)
                continue
            self.step_once = False

            # Agent picks action
            with torch.no_grad():
                q_values = self.model(state.unsqueeze(0)).squeeze()
            action = q_values.argmax().item()
            q_np = q_values.cpu().numpy()

            result = self.env.step(action)
            total_reward += result.reward
            alive = not result.done
            state_np = self.get_state()
            state = torch.tensor(state_np, dtype=torch.float, device=device)

            self._draw(state_np, q_np, total_reward, alive, action)
            self.clock.tick(self.fps)

        # Death frame — hold briefly
        self.scores_history.append(self.env.score)
        self.best_score = max(self.best_score, self.env.score)
        self._draw(state_np, q_np, total_reward, False, action)
        pygame.time.wait(600)

    # ── Drawing ──────────────────────────────────────────────────────
    def _draw(self, state_np, q_values, total_reward, alive, action=None):
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_panel(state_np, q_values, total_reward, alive, action)

        if not alive:
            self._draw_death_flash()

        pygame.display.flip()

    def _draw_grid(self):
        ox, oy = self.grid_origin
        # Background
        pygame.draw.rect(
            self.screen, GRID_BG, (ox, oy, self.grid_px, self.grid_px), border_radius=6
        )

        # Grid lines
        for i in range(self.env.width + 1):
            x = ox + i * self.cell
            pygame.draw.line(self.screen, GRID_LINE, (x, oy), (x, oy + self.grid_px))
        for i in range(self.env.height + 1):
            y = oy + i * self.cell
            pygame.draw.line(self.screen, GRID_LINE, (ox, y), (ox + self.grid_px, y))

        # Food glow
        fy, fx = self.env.food
        food_rect = pygame.Rect(
            ox + fx * self.cell + 2,
            oy + fy * self.cell + 2,
            self.cell - 4,
            self.cell - 4,
        )
        glow_surf = pygame.Surface((self.cell + 8, self.cell + 8), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, FOOD_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, (food_rect.x - 4, food_rect.y - 4))
        pygame.draw.rect(self.screen, FOOD_COL, food_rect, border_radius=5)

        # Snake
        snake = self.env.snake
        n = len(snake)
        direction = self.env.direction
        for i, (sy, sx) in enumerate(snake):
            rect = pygame.Rect(
                ox + sx * self.cell + 1,
                oy + sy * self.cell + 1,
                self.cell - 2,
                self.cell - 2,
            )
            if i == 0:
                col = SNAKE_HEAD
                pygame.draw.rect(self.screen, col, rect, border_radius=6)
                # Eyes
                dy, dx = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[int(direction)]
                cx, cy_c = rect.centerx, rect.centery
                eye_off = 5
                if dx == 0:  # vertical
                    e1 = (cx - eye_off, cy_c + dy * 4)
                    e2 = (cx + eye_off, cy_c + dy * 4)
                else:
                    e1 = (cx + dx * 4, cy_c - eye_off)
                    e2 = (cx + dx * 4, cy_c + eye_off)
                pygame.draw.circle(self.screen, (255, 255, 255), e1, 3)
                pygame.draw.circle(self.screen, (255, 255, 255), e2, 3)
                pygame.draw.circle(self.screen, (0, 0, 0), e1, 1)
                pygame.draw.circle(self.screen, (0, 0, 0), e2, 1)
            else:
                t = i / max(n - 1, 1)
                col = lerp_color(SNAKE_BODY, SNAKE_TAIL, t)
                pygame.draw.rect(self.screen, col, rect, border_radius=4)

    def _draw_panel(self, state_np, q_values, total_reward, alive, action):
        px = self.padding + self.grid_px + self.padding
        py = self.padding
        pw = self.panel_w
        ph = self.grid_px

        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG, (px, py, pw, ph), border_radius=6)

        x = px + 14
        y = py + 12
        col_w = pw - 28

        # ── Header: Score / Episode ──────────────────────────────────
        score_txt = self.font_lg.render(f"Score: {self.env.score}", True, TEXT_PRIMARY)
        self.screen.blit(score_txt, (x, y))
        y += 28

        ep_txt = self.font_sm.render(
            f"Ep {self.episode}   Best {self.best_score}   FPS {self.fps}",
            True,
            TEXT_SECONDARY,
        )
        self.screen.blit(ep_txt, (x, y))
        y += 20

        rew_txt = self.font_sm.render(f"Reward: {total_reward:+.1f}", True, TEXT_ACCENT)
        self.screen.blit(rew_txt, (x, y))
        y += 18

        steps_txt = self.font_xs.render(
            f"Steps: {self.env.steps}   Since food: {self.env.steps_since_food}",
            True,
            TEXT_SECONDARY,
        )
        self.screen.blit(steps_txt, (x, y))
        y += 20

        # Divider
        pygame.draw.line(self.screen, DIVIDER, (x, y), (x + col_w, y))
        y += 10

        # ── Q-Values + Action ────────────────────────────────────────
        section = self.font_md.render("Q-Values", True, TEXT_ACCENT)
        self.screen.blit(section, (x, y))
        y += 22

        if q_values is not None:
            best_a = int(np.argmax(q_values))
            q_min = float(np.min(q_values))
            q_max = float(np.max(q_values))
            q_range = max(q_max - q_min, 1e-6)

            for a_i in range(4):
                label = f"{ACTION_ARROWS[a_i]} {DIRECTION_NAMES[a_i]:>5}"
                is_best = a_i == best_a
                col = QVAL_BEST if is_best else TEXT_SECONDARY
                lbl = self.font_sm.render(label, True, col)
                self.screen.blit(lbl, (x, y))

                # Bar
                bar_x = x + 100
                bar_w = col_w - 145
                bar_h = 12
                pygame.draw.rect(
                    self.screen, BAR_BG, (bar_x, y + 2, bar_w, bar_h), border_radius=3
                )
                fill = (q_values[a_i] - q_min) / q_range
                fill_col = (
                    QVAL_BEST if is_best else lerp_color(BAR_NEUTRAL, BAR_POS, fill)
                )
                pygame.draw.rect(
                    self.screen,
                    fill_col,
                    (bar_x, y + 2, max(int(bar_w * fill), 2), bar_h),
                    border_radius=3,
                )

                val_txt = self.font_xs.render(f"{q_values[a_i]:+.2f}", True, col)
                self.screen.blit(val_txt, (bar_x + bar_w + 4, y + 1))
                y += 18
        else:
            y += 72

        y += 6
        pygame.draw.line(self.screen, DIVIDER, (x, y), (x + col_w, y))
        y += 10

        # ── Network-specific panels ───────────────────────────────────
        if self.network_type == "mlp":
            y = self._draw_panel_mlp(state_np, x, y, col_w)
        elif self.network_type == "conv":
            y = self._draw_panel_conv(state_np, x, y, col_w)
        elif self.network_type == "hybrid":
            y = self._draw_panel_hybrid(state_np, x, y, col_w)

        y += 8
        pygame.draw.line(self.screen, DIVIDER, (x, y), (x + col_w, y))
        y += 8

        # ── Controls hint ────────────────────────────────────────────
        hint = self.font_xs.render(
            "SPACE pause  ←→ step  ↑↓ speed  Q quit", True, TEXT_SECONDARY
        )
        self.screen.blit(hint, (x, y))

        # Pause badge
        if self.paused:
            badge = self.font_md.render("⏸ PAUSED", True, (255, 200, 60))
            self.screen.blit(badge, (x + col_w - badge.get_width(), py + 12))

    def _draw_panel_mlp(self, state_np, x, y, col_w):
        """Draw danger indicators, body distance bars, and state vector for MLP."""
        # ── Danger Indicators ────────────────────────────────────────
        section = self.font_md.render("Danger Map", True, TEXT_ACCENT)
        self.screen.blit(section, (x, y))
        y += 24

        if state_np is not None and len(state_np) >= 19:
            dangers = {
                "UP": state_np[11],
                "DOWN": state_np[12],
                "LEFT": state_np[13],
                "RIGHT": state_np[14],
            }
            cx = x + col_w // 2
            cy = y + 28
            r = 14
            positions = {
                "UP": (cx, cy - 26),
                "DOWN": (cx, cy + 26),
                "LEFT": (cx - 40, cy),
                "RIGHT": (cx + 40, cy),
            }
            pygame.draw.circle(self.screen, SNAKE_HEAD, (cx, cy), 6)

            for d_name, pos in positions.items():
                on = dangers[d_name] > 0.5
                col = DANGER_ON if on else DANGER_OFF
                pygame.draw.circle(self.screen, col, pos, r)
                arrow = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}[d_name]
                a_surf = self.font_sm.render(
                    arrow, True, (255, 255, 255) if on else TEXT_SECONDARY
                )
                a_rect = a_surf.get_rect(center=pos)
                self.screen.blit(a_surf, a_rect)

            y = cy + 50

            # Body distance bars
            body_dists = {
                "UP": state_np[15],
                "DOWN": state_np[16],
                "LEFT": state_np[17],
                "RIGHT": state_np[18],
            }
            bd_label = self.font_xs.render(
                "Body distance (per direction):", True, TEXT_SECONDARY
            )
            self.screen.blit(bd_label, (x, y))
            y += 16
            for d_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
                v = body_dists[d_name]
                lbl = self.font_xs.render(f" {d_name:>5}", True, TEXT_SECONDARY)
                self.screen.blit(lbl, (x, y))
                bar_x = x + 55
                bar_w = col_w - 55
                bar_h = 10
                pygame.draw.rect(
                    self.screen, BAR_BG, (bar_x, y + 1, bar_w, bar_h), border_radius=3
                )
                fill_col = lerp_color(BAR_NEG, BAR_POS, v)
                pygame.draw.rect(
                    self.screen,
                    fill_col,
                    (bar_x, y + 1, max(int(bar_w * v), 2), bar_h),
                    border_radius=3,
                )
                y += 14
        else:
            y += 100

        y += 8
        pygame.draw.line(self.screen, DIVIDER, (x, y), (x + col_w, y))
        y += 10

        # ── State Vector ─────────────────────────────────────────────
        section = self.font_md.render("State Vector", True, TEXT_ACCENT)
        self.screen.blit(section, (x, y))
        y += 20

        labels = [
            "head_y",
            "head_x",
            "food_y",
            "food_x",
            "food_dist",
            "wall_up",
            "wall_dn",
            "wall_lt",
            "wall_rt",
            "length",
            "dir",
            "dng_up",
            "dng_dn",
            "dng_lt",
            "dng_rt",
            "bd_up",
            "bd_dn",
            "bd_lt",
            "bd_rt",
        ]
        if state_np is not None:
            for i, (lbl, val) in enumerate(zip(labels, state_np)):
                txt = self.font_xs.render(f"{lbl:>9} {val:.3f}", True, TEXT_SECONDARY)
                col_offset = (i // 10) * (col_w // 2)
                row = i % 10
                self.screen.blit(txt, (x + col_offset, y + row * 14))

        y += 10 * 14 + 6
        return y

    def _draw_panel_conv(self, state_np, x, y, col_w):
        """Draw individual channel heatmaps for the multi-channel ConvDQN input."""
        section = self.font_md.render("Conv Input Channels", True, TEXT_ACCENT)
        self.screen.blit(section, (x, y))
        y += 22

        if state_np is None:
            return y + 100

        h = self.env.height
        w = self.env.width
        area = h * w
        num_channels = self.env.grid_channels

        y = self._draw_channel_heatmaps(
            state_np, 0, num_channels, h, w, area, x, y, col_w
        )
        return y

    def _draw_panel_hybrid(self, state_np, x, y, col_w):
        """Draw both conv channels and MLP danger/distance info for the hybrid model."""
        if state_np is None:
            return y + 100

        h = self.env.height
        w = self.env.width
        area = h * w
        num_channels = self.env.grid_channels
        grid_dim = num_channels * area

        # ── Conv channels ────────────────────────────────────────────
        section = self.font_md.render("Conv Stream", True, TEXT_ACCENT)
        self.screen.blit(section, (x, y))
        y += 20

        grid_data = state_np[:grid_dim]
        y = self._draw_channel_heatmaps(
            grid_data, 0, num_channels, h, w, area, x, y, col_w
        )

        y += 4
        pygame.draw.line(self.screen, DIVIDER, (x, y), (x + col_w, y))
        y += 8

        # ── MLP stream: compact danger + body distance ───────────────
        vec_data = state_np[grid_dim:]
        section = self.font_md.render("MLP Stream", True, TEXT_ACCENT)
        self.screen.blit(section, (x, y))
        y += 20

        if len(vec_data) >= 19:
            # Compact danger indicators (inline row)
            dangers = {
                "UP": vec_data[11],
                "DOWN": vec_data[12],
                "LEFT": vec_data[13],
                "RIGHT": vec_data[14],
            }
            lbl = self.font_xs.render("Danger:", True, TEXT_SECONDARY)
            self.screen.blit(lbl, (x, y))
            dx = x + 50
            for d_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
                on = dangers[d_name] > 0.5
                col = DANGER_ON if on else DANGER_OFF
                pygame.draw.circle(self.screen, col, (dx + 6, y + 5), 5)
                arrow = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}[d_name]
                a_surf = self.font_xs.render(
                    arrow, True, (255, 255, 255) if on else TEXT_SECONDARY
                )
                self.screen.blit(a_surf, (dx + 14, y - 1))
                dx += 38
            y += 18

            # Compact body distance bars
            body_dists = {
                "UP": vec_data[15],
                "DOWN": vec_data[16],
                "LEFT": vec_data[17],
                "RIGHT": vec_data[18],
            }
            lbl = self.font_xs.render("Body dist:", True, TEXT_SECONDARY)
            self.screen.blit(lbl, (x, y))
            y += 14
            for d_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
                v = body_dists[d_name]
                lbl = self.font_xs.render(f" {d_name:>5}", True, TEXT_SECONDARY)
                self.screen.blit(lbl, (x, y))
                bar_x = x + 55
                bar_w = col_w - 55
                bar_h = 8
                pygame.draw.rect(
                    self.screen, BAR_BG, (bar_x, y + 1, bar_w, bar_h), border_radius=3
                )
                fill_col = lerp_color(BAR_NEG, BAR_POS, v)
                pygame.draw.rect(
                    self.screen,
                    fill_col,
                    (bar_x, y + 1, max(int(bar_w * v), 2), bar_h),
                    border_radius=3,
                )
                y += 12

            y += 4

            # Key scalar values in a compact row
            food_dist = vec_data[4]
            snake_len = vec_data[9]
            direction = vec_data[10]
            dir_name = DIRECTION_NAMES.get(int(round(direction * 3)), "?")
            info = f"Food dist: {food_dist:.2f}   Length: {snake_len:.2f}   Dir: {dir_name}"
            info_surf = self.font_xs.render(info, True, TEXT_SECONDARY)
            self.screen.blit(info_surf, (x, y))
            y += 14

        return y

    def _draw_channel_heatmaps(
        self, data, data_offset, num_channels, h, w, area, x, y, col_w
    ):
        """Shared helper to draw side-by-side channel heatmaps."""
        channel_info = [
            ("Body Gradient", (0, 180, 100), (22, 22, 35)),
            ("Head", (0, 220, 120), (22, 22, 35)),
            ("Food", (255, 60, 80), (22, 22, 35)),
        ]

        gap = 6
        available_w = col_w - gap * (num_channels - 1)
        cell = max(available_w // (w * num_channels), 1)
        side_by_side = cell >= 2

        if side_by_side:
            hmap_w = cell * w
            hmap_h = cell * h
            total_w = hmap_w * num_channels + gap * (num_channels - 1)
            start_x = x + (col_w - total_w) // 2

            for ch_idx in range(num_channels):
                name, active_col, bg_col = channel_info[ch_idx]
                ch_x = start_x + ch_idx * (hmap_w + gap)
                lbl = self.font_xs.render(name, True, TEXT_SECONDARY)
                lbl_x = ch_x + (hmap_w - lbl.get_width()) // 2
                self.screen.blit(lbl, (lbl_x, y))

            y += 14

            for ch_idx in range(num_channels):
                name, active_col, bg_col = channel_info[ch_idx]
                ch_x = start_x + ch_idx * (hmap_w + gap)
                ch_start = data_offset + ch_idx * area
                ch_data = data[ch_start : ch_start + area].reshape(h, w)

                for row in range(h):
                    for col_i in range(w):
                        v = ch_data[row, col_i]
                        if v <= 0.0:
                            c = bg_col
                        else:
                            c = lerp_color(bg_col, active_col, v)
                        rect = pygame.Rect(
                            ch_x + col_i * cell,
                            y + row * cell,
                            cell - (1 if cell > 2 else 0),
                            cell - (1 if cell > 2 else 0),
                        )
                        pygame.draw.rect(self.screen, c, rect)

            y += hmap_h + 8
        else:
            cell = max(col_w // w, 2)
            hmap_w = cell * w
            hmap_h = cell * h
            start_x = x + (col_w - hmap_w) // 2

            for ch_idx in range(num_channels):
                name, active_col, bg_col = channel_info[ch_idx]
                ch_start = data_offset + ch_idx * area
                ch_data = data[ch_start : ch_start + area].reshape(h, w)

                lbl = self.font_xs.render(name, True, TEXT_SECONDARY)
                self.screen.blit(lbl, (start_x, y))
                y += 14

                for row in range(h):
                    for col_i in range(w):
                        v = ch_data[row, col_i]
                        if v <= 0.0:
                            c = bg_col
                        else:
                            c = lerp_color(bg_col, active_col, v)
                        rect = pygame.Rect(
                            start_x + col_i * cell,
                            y + row * cell,
                            cell - 1,
                            cell - 1,
                        )
                        pygame.draw.rect(self.screen, c, rect)

                y += hmap_h + 8

        # Legend
        y += 2
        legend_items = [
            ((22, 22, 35), "Empty"),
            ((0, 180, 100), "Body"),
            ((0, 220, 120), "Head"),
            ((255, 60, 80), "Food"),
        ]
        lx = x
        for color, label in legend_items:
            pygame.draw.rect(self.screen, color, (lx, y, 10, 10))
            lbl = self.font_xs.render(label, True, TEXT_SECONDARY)
            self.screen.blit(lbl, (lx + 14, y - 1))
            lx += 14 + lbl.get_width() + 12

        y += 20
        return y

    def _draw_death_flash(self):
        ox, oy = self.grid_origin
        overlay = pygame.Surface((self.grid_px, self.grid_px), pygame.SRCALPHA)
        overlay.fill((200, 40, 40, 45))
        self.screen.blit(overlay, (ox, oy))

        txt = self.font_lg.render("GAME OVER", True, DEATH_OVERLAY)
        rect = txt.get_rect(center=(ox + self.grid_px // 2, oy + self.grid_px // 2))
        self.screen.blit(txt, rect)


def main():
    parser = argparse.ArgumentParser(description="Watch a trained Snake AI play.")
    parser.add_argument(
        "--type",
        type=str,
        choices=["mlp", "conv", "hybrid"],
        default="mlp",
        help="Network architecture type (default: mlp)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model .pt file (default: models/snake_best_{type}.pt)",
    )
    parser.add_argument("--speed", type=int, default=8, help="Frames per second")
    parser.add_argument(
        "--layer-size",
        type=int,
        default=256,
        help="Hidden layer size (MLP/Hybrid only, must match training)",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=2,
        help="Number of hidden layers (MLP/Hybrid only, must match training)",
    )
    args = parser.parse_args()

    model_path = args.model or f"models/snake_best_{args.type}.pt"
    viewer = Viewer(
        model_path, args.type, args.layer_size, args.layer_count, args.speed
    )
    viewer.run()


if __name__ == "__main__":
    main()
