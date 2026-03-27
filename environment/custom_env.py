"""
Savanna Acoustic Threat Detection Environment
=============================================
A custom Gymnasium environment where an AI ranger drone navigates a 20x20
African savanna grid, responds to animal vocalizations, and identifies
ecological threats (poaching, predator proximity, habitat disturbance).

Mission: Analyze vocalizations of wildlife species to detect changes in
spatial and behavioral patterns for early identification of ecological threats.

Author: RL Summative Assignment
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

GRID_SIZE       = 20          # 20x20 savanna grid
MAX_STEPS       = 500         # episode length
BATTERY_MAX     = 100         # drone battery units
SENSOR_RANGE    = 3           # acoustic sensor radius (cells)
NUM_THREATS     = 5           # threats spawned per episode
NUM_SPECIES     = 4           # species monitored

# Cell biome types
BIOME_OPEN      = 0           # open savanna
BIOME_WOODLAND  = 1           # woodland (signal attenuated)
BIOME_WATER     = 2           # waterhole (wildlife magnet)
BIOME_ROCKY     = 3           # rocky outcrop (lion territory)

# Threat types
THREAT_NONE     = 0
THREAT_POACHING = 1           # highest priority
THREAT_PREDATOR = 2           # predator near herd
THREAT_HABITAT  = 3           # habitat disturbance

# Species IDs
SPECIES_ELEPHANT = 0
SPECIES_LION     = 1
SPECIES_HYENA    = 2
SPECIES_ZEBRA    = 3

# Acoustic call signatures (frequency bands, used for reward shaping)
SPECIES_NAMES = {
    SPECIES_ELEPHANT: "Elephant",
    SPECIES_LION:     "Lion",
    SPECIES_HYENA:    "Hyena",
    SPECIES_ZEBRA:    "Zebra"
}

# Action indices
ACTION_NORTH  = 0
ACTION_SOUTH  = 1
ACTION_EAST   = 2
ACTION_WEST   = 3
ACTION_SCAN   = 4  # deep acoustic scan of current cell
ACTION_ALERT  = 5  # radio alert to ranger base
ACTION_HOVER  = 6  # stay & conserve battery (low-cost idle)


class SavannaAcousticEnv(gym.Env):
    """
    Custom Gymnasium environment simulating an AI ranger drone that
    monitors wildlife vocalizations across an African savanna to detect
    ecological threats.

    Observation Space (Box, float32, shape=(847,)):
      - Drone x, y position            [0..19]  (2,)
      - Battery level                  [0..1]   (1,)
      - Step fraction                  [0..1]   (1,)
      - Acoustic heatmap               [0..1]   (400,) flattened 20×20
      - Confirmed threat map           [0..1]   (400,) flattened 20×20
      - Species presence map           [0..1]   (4×10×10 = 400) coarse grid
      - Visited cells map              [0,1]    (400,) binary
      - Last 3 actions (one-hot 7×3)            (21,)
      - Biome map                      [0..3]   (400,) flattened 20×20
      ─────────────────────────────────────────
      Total: 2+1+1+400+400+400+400+21+400 = 2025  → we'll use subset below

    We use a flattened obs of shape (847,) for SB3 compatibility:
      [drone_x, drone_y, battery, step_frac,          (4)
       acoustic_heatmap (400),                        (400)
       confirmed_threats (400),                       (400)
       visited (400),                                 (400)
       last_action (7)]                               (7)  → 36 - no: 4+400+400+400+7 = 1211

    Actual shape used: 847
      drone_xy(2) + battery(1) + step_frac(1) +
      acoustic_flat(400) + threat_flat(400) + visited_flat(400) +
      local_species_calls(36) + last_action_onehot(7)
      = 2+1+1+400+400+400+36+7 = 1247 → trimmed to 847 by removing visited
      Final: 2+1+1+400+400+36+7 = 847

    Action Space (Discrete, 7):
      0=North, 1=South, 2=East, 3=West,
      4=Scan, 5=Alert, 6=Hover

    Rewards:
      +10   Confirmed new threat via SCAN
      +5    Detected rare/flagship species (elephant, lion)
      +2    Coverage bonus (visiting new cell)
      +1    Approaching known acoustic hotspot
      -1    Revisiting recently-visited cell (last 20 steps)
      -3    Taking ALERT without prior SCAN (false alarm penalty)
      -5    Step cost (encourages efficiency)
      -15   Missing confirmed threat (threat escalates unresolved)
      -20   Battery depleted without returning to base
      -0.5  Per-step battery penalty (proportional to action cost)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Battery cost per action
    BATTERY_COST = {
        ACTION_NORTH: 2,
        ACTION_SOUTH: 2,
        ACTION_EAST:  2,
        ACTION_WEST:  2,
        ACTION_SCAN:  5,
        ACTION_ALERT: 1,
        ACTION_HOVER: 0.5,
    }

    def __init__(self, render_mode=None, grid_size=GRID_SIZE, seed=None):
        super().__init__()
        self.grid_size    = grid_size
        self.render_mode  = render_mode
        self._seed        = seed

        # ── Observation space ──────────────────────────────────────────────
        obs_dim = (
            2 +    # drone x, y
            1 +    # battery fraction
            1 +    # step fraction
            grid_size * grid_size +   # acoustic heatmap (400)
            grid_size * grid_size +   # threat map (400)
            36 +   # local 6×6 species call pattern (4 species × 9 cells = 36)
            7      # last action one-hot
        )  # = 847

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # ── Action space ───────────────────────────────────────────────────
        self.action_space = spaces.Discrete(7)

        # ── Internal state (initialised in reset) ─────────────────────────
        self.drone_pos       = None
        self.battery         = None
        self.step_count      = None
        self.biome_map       = None
        self.threats         = None          # list of dicts
        self.acoustic_map    = None          # float32 20×20 signal intensity
        self.threat_map      = None          # float32 20×20 confirmed threats
        self.visited         = None          # set of (x, y)
        self.recent_visited  = None          # deque for revisit penalty
        self.last_action     = None
        self.scan_ready      = False         # True if last action was SCAN
        self.resolved_threats= 0
        self.renderer        = None          # lazy-import pygame renderer

        self._np_random = np.random.default_rng(seed)

    # ── Helpers ─────────────────────────────────────────────────────────── #

    def _generate_biome(self):
        """Generate a realistic savanna biome map."""
        biome = np.full((self.grid_size, self.grid_size), BIOME_OPEN, dtype=np.int32)
        # Woodland patches (3-4 blobs)
        for _ in range(4):
            cx = self._np_random.integers(2, self.grid_size - 2)
            cy = self._np_random.integers(2, self.grid_size - 2)
            r  = self._np_random.integers(2, 5)
            for x in range(max(0, cx-r), min(self.grid_size, cx+r)):
                for y in range(max(0, cy-r), min(self.grid_size, cy+r)):
                    if (x-cx)**2 + (y-cy)**2 <= r**2:
                        biome[x, y] = BIOME_WOODLAND
        # Waterholes (2-3)
        for _ in range(3):
            x = self._np_random.integers(1, self.grid_size - 1)
            y = self._np_random.integers(1, self.grid_size - 1)
            biome[x, y]     = BIOME_WATER
            biome[x+1, y]   = BIOME_WATER
            biome[x, y+1]   = BIOME_WATER
        # Rocky outcrops (lion territory)
        for _ in range(2):
            x = self._np_random.integers(1, self.grid_size - 1)
            y = self._np_random.integers(1, self.grid_size - 1)
            biome[x, y] = BIOME_ROCKY
        return biome

    def _spawn_threats(self):
        """Spawn threats across the savanna; avoid base (0,0)."""
        threats = []
        for _ in range(NUM_THREATS):
            t_type = self._np_random.choice(
                [THREAT_POACHING, THREAT_PREDATOR, THREAT_HABITAT],
                p=[0.4, 0.4, 0.2]
            )
            # Associated species (e.g. poaching near elephants)
            species = {
                THREAT_POACHING: SPECIES_ELEPHANT,
                THREAT_PREDATOR: SPECIES_LION,
                THREAT_HABITAT:  SPECIES_ZEBRA
            }[t_type]

            pos = (
                self._np_random.integers(2, self.grid_size),
                self._np_random.integers(2, self.grid_size)
            )
            threats.append({
                "pos":       pos,
                "type":      t_type,
                "species":   species,
                "confirmed": False,
                "alerted":   False,
                "severity":  self._np_random.uniform(0.3, 1.0),
                "call_freq": self._np_random.uniform(0.5, 1.5),  # vocalization frequency
            })
        return threats

    def _build_acoustic_map(self):
        """Diffuse acoustic signal from threat locations and biome modifiers."""
        amap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for t in self.threats:
            tx, ty = t["pos"]
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    dist = np.sqrt((x - tx)**2 + (y - ty)**2)
                    if dist < SENSOR_RANGE + 2:
                        signal = t["severity"] * t["call_freq"] / (1 + dist)
                        # Woodland attenuates signal
                        if self.biome_map[x, y] == BIOME_WOODLAND:
                            signal *= 0.6
                        # Water amplifies (animals congregate)
                        if self.biome_map[x, y] == BIOME_WATER:
                            signal *= 1.3
                        amap[x, y] = min(1.0, amap[x, y] + signal)
        # Add low-level ambient noise
        amap += self._np_random.uniform(0, 0.05, amap.shape).astype(np.float32)
        return np.clip(amap, 0, 1)

    def _local_species_calls(self):
        """
        6×6 region around drone encoding per-species acoustic call intensity.
        Returns flat array of shape (36,) = 4 species × 9-cell region.
        """
        cx, cy   = self.drone_pos
        out      = np.zeros((4, 9), dtype=np.float32)
        offsets  = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
        for i, (dx, dy) in enumerate(offsets):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                for t in self.threats:
                    if t["pos"] == (nx, ny) or (
                        abs(t["pos"][0]-nx) <= 1 and abs(t["pos"][1]-ny) <= 1
                    ):
                        s = t["species"]
                        out[s, i] = min(1.0, out[s, i] + t["severity"])
        # Flatten: first 9 = elephant, next 9 = lion, etc.
        return out.flatten()[:36]

    def _get_obs(self):
        x, y = self.drone_pos
        obs = np.concatenate([
            np.array([x / (self.grid_size - 1),
                      y / (self.grid_size - 1)],        dtype=np.float32),
            np.array([self.battery / BATTERY_MAX],       dtype=np.float32),
            np.array([self.step_count / MAX_STEPS],      dtype=np.float32),
            self.acoustic_map.flatten(),                                   # (400)
            self.threat_map.flatten(),                                     # (400)
            self._local_species_calls(),                                   # (36)
            self._action_onehot(self.last_action),                        # (7)
        ])
        assert obs.shape == self.observation_space.shape, (
            f"Obs shape mismatch: {obs.shape} vs {self.observation_space.shape}"
        )
        return obs

    def _action_onehot(self, action):
        v = np.zeros(7, dtype=np.float32)
        if action is not None:
            v[action] = 1.0
        return v

    def _get_info(self):
        return {
            "step":             self.step_count,
            "battery":          self.battery,
            "drone_pos":        self.drone_pos,
            "resolved_threats": self.resolved_threats,
            "total_threats":    NUM_THREATS,
            "visited_cells":    len(self.visited),
        }

    # ── Core Gym API ─────────────────────────────────────────────────────── #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.drone_pos        = (0, 0)          # base station = top-left
        self.battery          = float(BATTERY_MAX)
        self.step_count       = 0
        self.biome_map        = self._generate_biome()
        self.threats          = self._spawn_threats()
        self.acoustic_map     = self._build_acoustic_map()
        self.threat_map       = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visited          = {(0, 0)}
        self.recent_visited   = []              # track last 20 positions
        self.last_action      = None
        self.scan_ready       = False
        self.resolved_threats = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward    = 0.0
        terminated = False
        truncated  = False

        # ── Battery cost ──────────────────────────────────────────────────
        cost = self.BATTERY_COST[action]
        self.battery -= cost

        # ── Movement ──────────────────────────────────────────────────────
        x, y   = self.drone_pos
        new_x, new_y = x, y

        if action == ACTION_NORTH:
            new_x = max(0, x - 1)
        elif action == ACTION_SOUTH:
            new_x = min(self.grid_size - 1, x + 1)
        elif action == ACTION_EAST:
            new_y = min(self.grid_size - 1, y + 1)
        elif action == ACTION_WEST:
            new_y = max(0, y - 1)

        self.drone_pos = (new_x, new_y)

        # ── Coverage bonus ────────────────────────────────────────────────
        if self.drone_pos not in self.visited:
            self.visited.add(self.drone_pos)
            reward += 2.0

        # ── Revisit penalty ───────────────────────────────────────────────
        if self.drone_pos in self.recent_visited[-20:]:
            reward -= 1.0

        self.recent_visited.append(self.drone_pos)
        if len(self.recent_visited) > 50:
            self.recent_visited.pop(0)

        # ── Approach hotspot reward ───────────────────────────────────────
        ax, ay = self.drone_pos
        acoustic_val = self.acoustic_map[ax, ay]
        if acoustic_val > 0.5:
            reward += acoustic_val * 1.0   # signal-proportional bonus

        # ── SCAN action ───────────────────────────────────────────────────
        if action == ACTION_SCAN:
            self.scan_ready = True
            found_new = False
            for t in self.threats:
                tx, ty = t["pos"]
                dist = np.sqrt((ax - tx)**2 + (ay - ty)**2)
                if dist <= SENSOR_RANGE and not t["confirmed"]:
                    t["confirmed"] = True
                    self.threat_map[tx, ty] = t["severity"]
                    reward += 10.0
                    # Bonus for flagship species
                    if t["species"] in [SPECIES_ELEPHANT, SPECIES_LION]:
                        reward += 5.0
                    found_new = True
                    self.resolved_threats += 1

            if not found_new:
                reward -= 2.0   # wasted scan

        # ── ALERT action ──────────────────────────────────────────────────
        elif action == ACTION_ALERT:
            if self.scan_ready:
                # Reward proportional to nearby confirmed threats
                nearby_conf = sum(
                    1 for t in self.threats
                    if t["confirmed"] and not t["alerted"] and
                    np.sqrt((ax-t["pos"][0])**2 + (ay-t["pos"][1])**2) <= SENSOR_RANGE
                )
                if nearby_conf > 0:
                    reward += 8.0 * nearby_conf
                    for t in self.threats:
                        if (t["confirmed"] and not t["alerted"] and
                                np.sqrt((ax-t["pos"][0])**2+(ay-t["pos"][1])**2) <= SENSOR_RANGE):
                            t["alerted"] = True
                else:
                    reward -= 3.0   # false alarm
                self.scan_ready = False
            else:
                reward -= 3.0       # alerting without scanning

        # ── HOVER action ──────────────────────────────────────────────────
        elif action == ACTION_HOVER:
            reward += acoustic_val * 0.5   # passive listening bonus

        # ── Step cost (encourages efficiency) ─────────────────────────────
        reward -= 0.5

        # ── Penalty: unresolved poaching threats escalate ─────────────────
        for t in self.threats:
            if t["type"] == THREAT_POACHING and not t["alerted"]:
                dist = np.sqrt((ax-t["pos"][0])**2 + (ay-t["pos"][1])**2)
                if dist > SENSOR_RANGE * 2:
                    reward -= 0.2  # distant unresolved poaching drains score

        # ── Battery exhaustion ────────────────────────────────────────────
        if self.battery <= 0:
            reward -= 20.0
            terminated = True

        # ── Max steps ─────────────────────────────────────────────────────
        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            # Penalty for unresolved threats
            unresolved = sum(1 for t in self.threats if not t["alerted"])
            reward -= unresolved * 5.0
            truncated = True

        # ── All threats resolved ───────────────────────────────────────────
        if all(t["alerted"] for t in self.threats):
            reward += 50.0   # mission complete bonus
            terminated = True

        self.last_action = action
        obs  = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Delegate to pygame renderer (defined in rendering.py)."""
        if self.render_mode is None:
            return
        if self.renderer is None:
            from environment.rendering import SavannaRenderer
            self.renderer = SavannaRenderer(self.grid_size)
        return self.renderer.render(self, self.render_mode)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
