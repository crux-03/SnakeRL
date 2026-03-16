use numpy::PyArray1;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;

// Direction vectors indexed by action: UP=0, DOWN=1, LEFT=2, RIGHT=3
const DY: [i32; 4] = [-1, 1, 0, 0];
const DX: [i32; 4] = [0, 0, -1, 1];
const OPPOSITE: [u8; 4] = [1, 0, 3, 2]; // UP<->DOWN, LEFT<->RIGHT

const STATE_SIZE: usize = 19;
const GRID_CHANNELS: usize = 3; // body gradient, head, food

/// Minimal result returned by step(). Avoids allocating a Python dict for info
/// on every step — the training loop only reads reward and done.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct StepResult {
    #[pyo3(get)]
    reward: f32,
    #[pyo3(get)]
    done: bool,
}

#[pyclass]
struct SnakeEnv {
    width: usize,
    height: usize,
    board_size: usize,

    // Positions stored as flat indices (y * width + x) for cache-friendly access.
    // VecDeque gives O(1) push_front / pop_back — allocated once, never resized.
    snake: VecDeque<u16>,

    // Flat occupancy grid: occupied[y * width + x] = true if snake is there.
    // Replaces HashSet — no hashing, no allocation, perfect cache locality.
    occupied: Vec<bool>,

    dir: u8,
    food_pos: u16, // flat index
    score: u32,
    done: bool,
    steps: u32,
    steps_since_food: u32,
    max_steps_without_food: u32,

    rng: ChaCha8Rng,

    // Reward config
    food_reward: f32,
    death_reward: f32,
    step_reward: f32,
    closer_reward: f32,
    farther_reward: f32,

    // Pre-computed reciprocals
    inv_h: f32,
    inv_w: f32,
    inv_max_dist: f32,
    inv_max_len: f32,
    inv_max_dim: f32,

    // Pre-allocated output buffers — written in-place, copied out by numpy
    state_buf: Vec<f32>,
    grid_buf: Vec<f32>,
}

// ---- Internal helpers (not exposed to Python) ----
impl SnakeEnv {
    #[inline(always)]
    fn pack(&self, y: usize, x: usize) -> u16 {
        (y * self.width + x) as u16
    }

    #[inline(always)]
    fn unpack(&self, pos: u16) -> (usize, usize) {
        let p = pos as usize;
        (p / self.width, p % self.width)
    }

    #[inline(always)]
    fn manhattan(&self, a: u16, b: u16) -> u32 {
        let (ay, ax) = self.unpack(a);
        let (by, bx) = self.unpack(b);
        ay.abs_diff(by) as u32 + ax.abs_diff(bx) as u32
    }

    fn spawn_food(&mut self) -> u16 {
        let bs = self.board_size;

        // Rejection sampling — nearly always hits on first try when board is mostly empty
        for _ in 0..100 {
            let idx = self.rng.gen_range(0..bs);
            if !self.occupied[idx] {
                return idx as u16;
            }
        }

        // Fallback: scan the grid (only when board is nearly full)
        let start = self.rng.gen_range(0..bs);
        for offset in 0..bs {
            let idx = (start + offset) % bs;
            if !self.occupied[idx] {
                return idx as u16;
            }
        }

        // Board completely full (shouldn't happen in practice)
        self.snake[0]
    }

    fn fill_state_buf(&mut self) {
        let (head_y, head_x) = self.unpack(self.snake[0]);
        let (food_y, food_x) = self.unpack(self.food_pos);
        let h = self.height;
        let w = self.width;
        let inv_h = self.inv_h;
        let inv_w = self.inv_w;
        let snake_len = self.snake.len() as f32;
        let direction = self.dir;
        let manhattan = self.manhattan(self.snake[0], self.food_pos) as f32;

        let occ = &self.occupied;

        // Danger flags (short-circuit avoids out-of-bounds)
        let d_up = (head_y == 0 || occ[(head_y - 1) * w + head_x]) as u8 as f32;
        let d_down = (head_y == h - 1 || occ[(head_y + 1) * w + head_x]) as u8 as f32;
        let d_left = (head_x == 0 || occ[head_y * w + head_x - 1]) as u8 as f32;
        let d_right = (head_x == w - 1 || occ[head_y * w + head_x + 1]) as u8 as f32;

        // Body distance scans
        let bd_up = self.body_dist_scan(head_y, head_x, -1, 0);
        let bd_down = self.body_dist_scan(head_y, head_x, 1, 0);
        let bd_left = self.body_dist_scan(head_y, head_x, 0, -1);
        let bd_right = self.body_dist_scan(head_y, head_x, 0, 1);

        // Now write — no conflicting borrows
        let buf = &mut self.state_buf;
        buf[0] = head_y as f32 * inv_h;
        buf[1] = head_x as f32 * inv_w;
        buf[2] = food_y as f32 * inv_h;
        buf[3] = food_x as f32 * inv_w;
        buf[4] = manhattan * self.inv_max_dist;
        buf[5] = head_y as f32 * inv_h;
        buf[6] = (h - 1 - head_y) as f32 * inv_h;
        buf[7] = head_x as f32 * inv_w;
        buf[8] = (w - 1 - head_x) as f32 * inv_w;
        buf[9] = snake_len * self.inv_max_len;
        buf[10] = direction as f32 / 3.0;
        buf[11] = d_up;
        buf[12] = d_down;
        buf[13] = d_left;
        buf[14] = d_right;
        buf[15] = bd_up;
        buf[16] = bd_down;
        buf[17] = bd_left;
        buf[18] = bd_right;
    }

    fn body_dist_scan(&self, head_y: usize, head_x: usize, dy: i32, dx: i32) -> f32 {
        let h = self.height as i32;
        let w = self.width as i32;
        let occ = &self.occupied;
        let mut y = head_y as i32 + dy;
        let mut x = head_x as i32 + dx;
        let mut steps: u32 = 1;

        while y >= 0 && y < h && x >= 0 && x < w {
            if occ[y as usize * self.width + x as usize] {
                return steps as f32 * self.inv_max_dim;
            }
            y += dy;
            x += dx;
            steps += 1;
        }
        1.0
    }

    fn fill_grid_buf(&mut self) {
        let buf = &mut self.grid_buf;
        let area = self.board_size;
        let len = self.snake.len();
        let inv_len = if len > 1 { 1.0 / (len - 1) as f32 } else { 1.0 };

        buf.fill(0.0);

        // Channel 0: body gradient
        // Segment near head (i=1) ≈ 1.0 (most dangerous, expires last)
        // Tail (i=len-1) → small positive value (expires soonest)
        for (i, &pos) in self.snake.iter().enumerate().skip(1) {
            buf[pos as usize] = 1.0 - (i as f32) * inv_len;
        }

        // Channel 1: head
        buf[area + self.snake[0] as usize] = 1.0;

        // Channel 2: food
        buf[2 * area + self.food_pos as usize] = 1.0;
    }
}

// ---- Python API ----
#[pymethods]
impl SnakeEnv {
    #[new]
    #[pyo3(signature = (width=20, height=20, seed=None, max_steps_without_food=None))]
    fn new(
        width: usize,
        height: usize,
        seed: Option<u64>,
        max_steps_without_food: Option<u32>,
    ) -> Self {
        let board_size = width * height;
        let s = seed.unwrap_or(42);

        SnakeEnv {
            width,
            height,
            board_size,
            snake: VecDeque::with_capacity(board_size),
            occupied: vec![false; board_size],
            dir: 3, // RIGHT
            food_pos: 0,
            score: 0,
            done: true,
            steps: 0,
            steps_since_food: 0,
            max_steps_without_food: max_steps_without_food.unwrap_or(board_size as u32),
            rng: ChaCha8Rng::seed_from_u64(s),
            food_reward: 50.0,
            death_reward: -1000.0,
            step_reward: -0.001,
            closer_reward: 0.15,
            farther_reward: -0.1,
            inv_h: 1.0 / (height - 1) as f32,
            inv_w: 1.0 / (width - 1) as f32,
            inv_max_dist: 1.0 / (height + width) as f32,
            inv_max_len: 1.0 / board_size as f32,
            inv_max_dim: 1.0 / width.max(height) as f32,
            state_buf: vec![0.0; STATE_SIZE],
            grid_buf: vec![0.0; GRID_CHANNELS * board_size],
        }
    }

    #[pyo3(signature = (seed=None))]
    fn reset<'py>(&mut self, py: Python<'py>, seed: Option<u64>) -> Bound<'py, PyArray1<f32>> {
        if let Some(s) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(s);
        }

        // Clear occupancy
        self.occupied.fill(false);
        self.snake.clear();

        let mid_y = self.height / 2;
        let mid_x = self.width / 2;

        // Place initial snake (3 segments, facing right)
        for dx in 0..3 {
            let pos = self.pack(mid_y, mid_x - dx);
            self.snake.push_back(pos);
            self.occupied[pos as usize] = true;
        }

        self.dir = 3; // RIGHT
        self.score = 0;
        self.done = false;
        self.steps = 0;
        self.steps_since_food = 0;
        self.food_pos = self.spawn_food();

        self.get_state(py)
    }

    fn step<'py>(&mut self, _py: Python<'py>, action: u8) -> PyResult<StepResult> {
        if self.done {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Episode is over. Call reset() first.",
            ));
        }

        self.steps += 1;
        self.steps_since_food += 1;

        let direction = self.dir;

        // Prevent 180 reversal
        if OPPOSITE[action as usize] != direction {
            self.dir = action;
        }
        let dir = self.dir as usize;

        // Compute new head position
        let (head_y, head_x) = self.unpack(self.snake[0]);
        let ny = head_y as i32 + DY[dir];
        let nx = head_x as i32 + DX[dir];

        // Manhattan distance before move
        let old_dist = self.manhattan(self.snake[0], self.food_pos);

        // Wall collision
        if ny < 0 || ny >= self.height as i32 || nx < 0 || nx >= self.width as i32 {
            self.done = true;
            return Ok(StepResult {
                reward: self.death_reward,
                done: true,
            });
        }

        let new_head = self.pack(ny as usize, nx as usize);

        // Self collision (O(1) array lookup)
        if self.occupied[new_head as usize] {
            self.done = true;
            return Ok(StepResult {
                reward: self.death_reward,
                done: true,
            });
        }

        // Move snake
        let mut reward = self.step_reward;
        self.snake.push_front(new_head);
        self.occupied[new_head as usize] = true;

        if new_head == self.food_pos {
            self.score += 1;
            self.steps_since_food = 0;
            self.food_pos = self.spawn_food();
            reward += self.food_reward;
        } else {
            // Remove tail
            let tail = self.snake.pop_back().unwrap();
            self.occupied[tail as usize] = false;

            // Distance shaping (quadratic drop-off)
            let ratio = self.steps_since_food as f32 / self.max_steps_without_food as f32;
            let urgency = 1.0 - 8.0 * ratio * ratio;
            let new_dist = self.manhattan(new_head, self.food_pos);
            if new_dist < old_dist {
                reward += self.closer_reward * urgency * urgency;
            } else {
                reward += self.farther_reward * urgency * urgency;
            }
        }

        // Starvation
        if self.steps_since_food >= self.max_steps_without_food {
            self.done = true;
            return Ok(StepResult {
                reward: self.death_reward,
                done: true,
            });
        }

        Ok(StepResult {
            reward,
            done: false,
        })
    }

    fn get_state<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.fill_state_buf();
        PyArray1::from_slice(py, &self.state_buf)
    }

    fn get_grid_state<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.fill_grid_buf();
        PyArray1::from_slice(py, &self.grid_buf)
    }

    fn sample_action(&mut self) -> u8 {
        self.rng.gen_range(0..4)
    }

    // ---- Properties matching the Python API ----

    #[getter]
    fn score(&self) -> u32 {
        self.score
    }

    #[getter]
    fn steps(&self) -> u32 {
        self.steps
    }

    #[getter]
    fn done(&self) -> bool {
        self.done
    }

    #[getter]
    fn width(&self) -> usize {
        self.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.height
    }

    #[getter]
    fn action_space_size(&self) -> usize {
        4
    }

    #[getter]
    fn state_size(&self) -> usize {
        STATE_SIZE
    }

    #[getter]
    fn state_shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    #[getter]
    fn food(&self) -> (usize, usize) {
        self.unpack(self.food_pos)
    }

    #[getter]
    fn direction(&self) -> u8 {
        self.dir
    }

    #[getter]
    fn steps_since_food(&self) -> u32 {
        self.steps_since_food
    }

    #[getter]
    fn snake(&self) -> Vec<(usize, usize)> {
        self.snake.iter().map(|&pos| self.unpack(pos)).collect()
    }

    #[getter]
    fn grid_channels(&self) -> usize {
        GRID_CHANNELS
    }

    #[getter]
    fn grid_state_size(&self) -> usize {
        GRID_CHANNELS * self.board_size
    }
}

#[pymodule]
fn snake_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SnakeEnv>()?;
    m.add_class::<StepResult>()?;
    Ok(())
}
