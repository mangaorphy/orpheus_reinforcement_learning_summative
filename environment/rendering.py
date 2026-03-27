"""
Savanna Acoustic Monitoring — Enhanced Pygame Renderer v2
==========================================================
Visual improvements:
  • Textured biome tiles (grass tufts, tree dots, water shimmer, rocky stipple)
  • Smooth animated drone trail showing recent path
  • Spinning rotor blur animation on drone
  • Expanding acoustic wave rings on SCAN action
  • Species halo rings pulsing per call frequency
  • Mini-map in HUD showing full 20x20 grid
  • Live reward sparkline graph in HUD
  • Gradient sky background behind grid
  • Smooth battery arc with colour-coded gradient
  • Real-time action label floating above drone
  • Day-cycle sky tint based on step progress
"""

import os
import sys
import math
import random
import pygame
import numpy as np
from collections import deque

# --------------------------------------------------------------------------- #
# Colour palette
# --------------------------------------------------------------------------- #
C = {
    "savanna_l":  (225, 205, 130), "savanna_d":  (195, 170,  95),
    "woodland_l": (110, 155,  75), "woodland_d": ( 75, 115,  50),
    "water_l":    ( 90, 175, 220), "water_d":    ( 55, 130, 185),
    "rocky_l":    (175, 162, 142), "rocky_d":    (135, 122, 105),
    "poaching":   (235,  55,  55), "predator":   (235, 145,  30),
    "habitat":    (150,  80, 215), "confirmed":  (255, 230,  30),
    "alerted":    ( 60, 220,  90),
    "drone_body": ( 40, 140, 255), "drone_arm":  ( 20,  90, 200),
    "drone_rotor":(180, 210, 255), "trail":      ( 80, 160, 255),
    "hud_bg":     ( 12,  16,  24), "hud_panel":  ( 20,  28,  40),
    "hud_border": ( 45,  60,  80), "hud_text":   (220, 225, 235),
    "hud_dim":    (130, 140, 155), "good":       ( 70, 215,  95),
    "warn":       (215, 195,  45), "bad":        (215,  70,  70),
    "info":       ( 70, 160, 255),
    "elephant":   (180, 125,  65), "lion":       (235, 168,  35),
    "hyena":      (165,  95, 195), "zebra":      (235, 235, 235),
    "white":      (255, 255, 255), "black":      (  0,   0,   0),
    "bg":         (  8,  12,  18), "grid_line":  ( 40,  50,  60),
    "sky_top":    ( 15,  25,  45), "sky_bot":    ( 30,  50,  80),
}

BIOME_BASE  = {0: C["savanna_l"], 1: C["woodland_l"], 2: C["water_l"],   3: C["rocky_l"]}
BIOME_DARK  = {0: C["savanna_d"], 1: C["woodland_d"], 2: C["water_d"],   3: C["rocky_d"]}
THREAT_COL  = {1: C["poaching"],  2: C["predator"],   3: C["habitat"]}
THREAT_LBL  = {1: "POACH",        2: "PRED",           3: "HAB"}
SPECIES_COL = {0: C["elephant"],  1: C["lion"],        2: C["hyena"],    3: C["zebra"]}
SPECIES_LBL = {0: "ELEPH",        1: "LION",           2: "HYENA",       3: "ZEBRA"}

CELL      = 36
HUD_W     = 300
GRID_PX   = CELL * 20
WIN_W     = GRID_PX + HUD_W
WIN_H     = GRID_PX
FPS       = 15
TRAIL_LEN = 30


class SavannaRenderer:

    def __init__(self, grid_size=20):
        self.grid_size          = grid_size
        self.screen             = None
        self.surf               = None
        self.clock              = None
        self.font_title         = None
        self.font_md            = None
        self.font_sm            = None
        self.font_xs            = None
        self._tick              = 0
        self._initialized       = False
        self._trail             = deque(maxlen=TRAIL_LEN)
        self._reward_hist       = deque(maxlen=80)
        self._last_action       = None
        self._wave_rings        = []
        self._tile_cache        = {}
        self._rotor_angle       = 0.0
        self._cumulative_reward = 0.0

    # ── Init ──────────────────────────────────────────────────────────────── #

    def _init(self):
        if self._initialized:
            return
        pygame.init()
        pygame.display.set_caption(
            "🦁  Savanna Acoustic Threat Detection  |  AI Ranger Drone"
        )
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.surf   = pygame.Surface((WIN_W, WIN_H))
        self.clock  = pygame.time.Clock()
        try:
            self.font_title = pygame.font.SysFont("dejavusansmono", 14, bold=True)
            self.font_md    = pygame.font.SysFont("dejavusansmono", 12, bold=True)
            self.font_sm    = pygame.font.SysFont("dejavusansmono", 11)
            self.font_xs    = pygame.font.SysFont("dejavusansmono", 10)
        except Exception:
            self.font_title = pygame.font.Font(None, 16)
            self.font_md    = pygame.font.Font(None, 14)
            self.font_sm    = pygame.font.Font(None, 13)
            self.font_xs    = pygame.font.Font(None, 12)
        self._build_tile_cache()
        self._initialized = True

    def _build_tile_cache(self):
        """Pre-render textured biome tiles once at startup."""
        rng = random.Random(42)
        for biome in range(4):
            surf = pygame.Surface((CELL, CELL))
            base = BIOME_BASE[biome]
            dark = BIOME_DARK[biome]
            surf.fill(base)
            if biome == 0:    # savanna — grass tufts
                for _ in range(6):
                    x = rng.randint(2, CELL-4)
                    y = rng.randint(2, CELL-4)
                    pygame.draw.line(surf, dark, (x, y), (x-1, y-3), 1)
                    pygame.draw.line(surf, dark, (x, y), (x+1, y-3), 1)
            elif biome == 1:  # woodland — tree blobs
                for _ in range(4):
                    x = rng.randint(4, CELL-6)
                    y = rng.randint(4, CELL-6)
                    pygame.draw.circle(surf, dark, (x, y), rng.randint(3, 5))
                    pygame.draw.circle(surf, base, (x-1, y-1), 2)
            elif biome == 2:  # water — shimmer lines
                for i in range(3):
                    yy = 6 + i * 9
                    pygame.draw.line(surf, C["water_d"], (3, yy), (CELL-3, yy), 1)
                    pygame.draw.line(surf, (200, 235, 255), (5, yy-1), (CELL-8, yy-1), 1)
            elif biome == 3:  # rocky — stipple
                for _ in range(8):
                    x = rng.randint(2, CELL-3)
                    y = rng.randint(2, CELL-3)
                    pygame.draw.rect(surf, dark, (x, y, 2, 2))
            self._tile_cache[biome] = surf

    # ── Main render loop ──────────────────────────────────────────────────── #

    def render(self, env, render_mode, last_reward=0.0, last_action=None):
        self._init()
        self._tick         += 1
        self._rotor_angle  += 18
        self._last_action   = last_action
        if last_reward != 0:
            self._reward_hist.append(last_reward)
            self._cumulative_reward += last_reward
        if last_action == 4:   # SCAN — spawn expanding ring
            dx, dy = env.drone_pos
            rc = self._cell_rect(dx, dy)
            self._wave_rings.append(
                [rc.centerx, rc.centery, 0, CELL * 3, C["info"], 200]
            )
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close(); sys.exit()
        self._trail.append(env.drone_pos)
        self.surf.fill(C["bg"])
        self._draw_sky()
        self._draw_grid(env)
        self._draw_acoustic_overlay(env)
        self._draw_trail()
        self._draw_wave_rings()
        self._draw_species_halos(env)
        self._draw_threats(env)
        self._draw_drone(env)
        self._draw_action_label(env)
        self._draw_hud(env, last_reward)
        self.screen.blit(self.surf, (0, 0))
        if render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        elif render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self.surf), axes=(1, 0, 2)
            )

    # ── Drawing helpers ───────────────────────────────────────────────────── #

    def _cell_rect(self, x, y):
        return pygame.Rect(y * CELL, x * CELL, CELL, CELL)

    def _draw_sky(self):
        for py in range(WIN_H):
            t   = py / WIN_H
            col = tuple(int(C["sky_top"][i] + t*(C["sky_bot"][i]-C["sky_top"][i]))
                        for i in range(3))
            pygame.draw.line(self.surf, col, (0, py), (GRID_PX, py))

    def _draw_grid(self, env):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = self._cell_rect(x, y)
                self.surf.blit(self._tile_cache[env.biome_map[x, y]], rect.topleft)
                if (x, y) in env.visited:
                    dim = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                    dim.fill((255, 255, 255, 18))
                    self.surf.blit(dim, rect.topleft)
                pygame.draw.rect(self.surf, C["grid_line"], rect, 1)

    def _draw_acoustic_overlay(self, env):
        overlay = pygame.Surface((GRID_PX, WIN_H), pygame.SRCALPHA)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                v = float(env.acoustic_map[x, y])
                if v < 0.05:
                    continue
                rc = int(min(255, v * 2.2 * 200))
                gc = int(min(255, v * 120))
                bc = int(max(0,   180 - v * 200))
                a  = int(v * 140)
                pygame.draw.rect(overlay, (rc, gc, bc, a), self._cell_rect(x, y))
        self.surf.blit(overlay, (0, 0))

    def _draw_trail(self):
        pts = list(self._trail)
        if len(pts) < 2:
            return
        s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for i in range(1, len(pts)):
            alpha = int(255 * (i / len(pts)) * 0.55)
            r1    = self._cell_rect(*pts[i-1])
            r2    = self._cell_rect(*pts[i])
            w     = max(1, int(3 * i / len(pts)))
            pygame.draw.line(s, (*C["trail"], alpha), r1.center, r2.center, w)
        self.surf.blit(s, (0, 0))

    def _draw_wave_rings(self):
        alive = []
        for ring in self._wave_rings:
            cx, cy, r, max_r, col, alpha = ring
            if r < max_r and alpha > 10:
                ring[2] += 4
                ring[5]  = max(0, alpha - 8)
                s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                pygame.draw.circle(s, (*col, ring[5]), (cx, cy), int(ring[2]), 2)
                self.surf.blit(s, (0, 0))
                alive.append(ring)
        self._wave_rings = alive

    def _draw_species_halos(self, env):
        for t in env.threats:
            if t["confirmed"]:
                continue
            tx, ty = t["pos"]
            rect   = self._cell_rect(tx, ty)
            cx, cy = rect.centerx, rect.centery
            sc     = SPECIES_COL[t["species"]]
            pulse  = (math.sin(self._tick * t["call_freq"] * 0.4) + 1) / 2
            halo_r = int(CELL * 1.1 + pulse * 10)
            alpha  = int(35 + pulse * 55)
            s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            pygame.draw.circle(s, (*sc, alpha), (cx, cy), halo_r, 2)
            self.surf.blit(s, (0, 0))

    def _draw_threats(self, env):
        for t in env.threats:
            tx, ty = t["pos"]
            rect   = self._cell_rect(tx, ty)
            cx, cy = rect.centerx, rect.centery
            tc     = THREAT_COL[t["type"]]
            r      = CELL // 3 + 1
            if not t["alerted"]:
                pulse  = (math.sin(self._tick * 0.25 + tx) + 1) / 2
                glow_r = int(r + 4 + pulse * 7)
                glow_a = int(80 + pulse * 120)
                gs = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                pygame.draw.circle(gs, (*tc, glow_a), (cx, cy), glow_r)
                self.surf.blit(gs, (0, 0))
            fill = (C["alerted"]  if t["alerted"] else
                    C["confirmed"] if t["confirmed"] else tc)
            pygame.draw.circle(self.surf, fill,    (cx, cy), r)
            pygame.draw.circle(self.surf, C["white"], (cx, cy), r, 1)
            lbl = self.font_xs.render(THREAT_LBL[t["type"]], True, C["black"])
            self.surf.blit(lbl, (cx - lbl.get_width()//2, cy - lbl.get_height()//2))
            sp  = self.font_xs.render(SPECIES_LBL[t["species"]], True, SPECIES_COL[t["species"]])
            self.surf.blit(sp, (cx - sp.get_width()//2, cy + r + 1))
            if t["alerted"]:
                ck = self.font_md.render("v", True, C["good"])
                self.surf.blit(ck, (cx + r, cy - r))

    def _draw_drone(self, env):
        dx, dy = env.drone_pos
        rect   = self._cell_rect(dx, dy)
        cx, cy = rect.centerx, rect.centery
        arm_r  = CELL // 3
        for base_angle in [45, 135, 225, 315]:
            angle = math.radians(base_angle + self._rotor_angle)
            px = cx + int(arm_r * math.cos(angle))
            py = cy + int(arm_r * math.sin(angle))
            pygame.draw.line(self.surf, C["drone_arm"], (cx, cy), (px, py), 2)
            pygame.draw.circle(self.surf, C["drone_rotor"], (px, py), 5)
            pygame.draw.circle(self.surf, C["drone_arm"],   (px, py), 5, 1)
        shadow = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        pygame.draw.circle(shadow, (0, 0, 0, 60), (cx+2, cy+2), CELL//4 + 2)
        self.surf.blit(shadow, (0, 0))
        pygame.draw.circle(self.surf, C["white"],      (cx, cy), CELL//4 + 2)
        pygame.draw.circle(self.surf, C["drone_body"], (cx, cy), CELL//4)
        batt_frac = env.battery / 100.0
        bc = (C["good"] if batt_frac > 0.5 else
              C["warn"] if batt_frac > 0.25 else C["bad"])
        arc_r = pygame.Rect(cx - CELL//4 - 5, cy - CELL//4 - 5,
                            (CELL//4 + 5)*2, (CELL//4 + 5)*2)
        if batt_frac > 0.01:
            pygame.draw.arc(self.surf, bc, arc_r,
                            math.radians(90),
                            math.radians(90 + 360 * batt_frac), 3)
        pygame.draw.circle(self.surf, C["white"],      (cx, cy), 4)
        pygame.draw.circle(self.surf, C["drone_body"], (cx, cy), 2)

    def _draw_action_label(self, env):
        ACTION_NAMES = {
            0:"N NORTH", 1:"S SOUTH", 2:"E EAST",
            3:"W WEST",  4:"SCAN",    5:"ALERT",  6:"HOVER"
        }
        if self._last_action is None:
            return
        dx, dy = env.drone_pos
        rect   = self._cell_rect(dx, dy)
        cx     = rect.centerx
        cy     = rect.centery - CELL
        lbl    = self.font_xs.render(ACTION_NAMES.get(self._last_action,""), True, C["info"])
        bg     = pygame.Surface((lbl.get_width()+6, lbl.get_height()+4), pygame.SRCALPHA)
        bg.fill((10, 20, 40, 190))
        self.surf.blit(bg,  (cx - lbl.get_width()//2 - 3, cy - 2))
        self.surf.blit(lbl, (cx - lbl.get_width()//2,     cy))

    # ── HUD ───────────────────────────────────────────────────────────────── #

    def _draw_hud(self, env, last_reward):
        hx = GRID_PX
        pygame.draw.rect(self.surf, C["hud_bg"], (hx, 0, HUD_W, WIN_H))
        pygame.draw.line(self.surf, C["hud_border"], (hx, 0), (hx, WIN_H), 2)
        lm = hx + 10
        y  = 0

        def text(t, color=C["hud_text"], font=None, bold=False):
            nonlocal y
            f = font or (self.font_md if bold else self.font_sm)
            s = f.render(str(t), True, color)
            self.surf.blit(s, (lm, y))
            y += s.get_height() + 2

        def sep():
            nonlocal y
            pygame.draw.line(self.surf, C["hud_border"],
                             (lm, y+2), (hx+HUD_W-10, y+2), 1)
            y += 7

        def bar(val, max_val, w, h, fg, bg_c=(40,50,65)):
            nonlocal y
            pygame.draw.rect(self.surf, bg_c, (lm, y, w, h), border_radius=3)
            filled = int(w * max(0, min(1, val/max_val)))
            if filled > 0:
                pygame.draw.rect(self.surf, fg, (lm, y, filled, h), border_radius=3)
            y += h + 4

        # Title bar
        pygame.draw.rect(self.surf, C["hud_panel"], (hx, 0, HUD_W, 28))
        title = self.font_title.render("RANGER DRONE HUD", True, C["info"])
        self.surf.blit(title, (lm, 7))
        y = 34
        sep()

        # Mission
        text("MISSION", C["hud_text"], bold=True)
        text(f"Step   {env.step_count:>4} / 500", C["hud_dim"])
        bar(env.step_count, 500, HUD_W-20, 5, C["info"])
        batt_c = (C["good"] if env.battery > 50 else
                  C["warn"] if env.battery > 25 else C["bad"])
        text(f"Battery {env.battery:>5.1f}%", batt_c)
        bar(env.battery, 100, HUD_W-20, 6, batt_c)
        text(f"Pos    ({env.drone_pos[0]:>2}, {env.drone_pos[1]:>2})", C["hud_dim"])
        cov = len(env.visited)
        text(f"Cover  {cov:>3}/400  ({100*cov/400:.0f}%)", C["good"])
        bar(cov, 400, HUD_W-20, 5, C["good"])
        sep()

        # Threats
        text("THREATS", C["warn"], bold=True)
        res  = sum(1 for t in env.threats if t["alerted"])
        conf = sum(1 for t in env.threats if t["confirmed"])
        text(f"Alerted   {res}/{len(env.threats)}",
             C["good"] if res > 0 else C["bad"])
        text(f"Confirmed {conf}/{len(env.threats)}", C["warn"])
        text(f"Pending   {len(env.threats)-res}",
             C["bad"] if res < len(env.threats) else C["good"])
        for t in env.threats:
            status = "OK" if t["alerted"] else ("??" if t["confirmed"] else "!!")
            sc = (C["good"] if t["alerted"] else
                  C["warn"] if t["confirmed"] else C["bad"])
            s  = self.font_xs.render(
                f"  [{status}] {THREAT_LBL[t['type']]} "
                f"({t['pos'][0]:2},{t['pos'][1]:2}) {SPECIES_LBL[t['species']]}",
                True, sc)
            self.surf.blit(s, (lm, y))
            y += s.get_height() + 2
        sep()

        # Signal
        ax, ay = env.drone_pos
        sig    = float(env.acoustic_map[ax, ay])
        sig_c  = (C["good"] if sig > 0.5 else
                  C["warn"] if sig > 0.2 else C["bad"])
        text("ACOUSTIC SIGNAL", C["hud_text"], bold=True)
        text(f"Intensity  {sig*100:.0f}%", sig_c)
        bar(sig, 1.0, HUD_W-20, 7, sig_c)
        text(f"Scan ready {'YES' if env.scan_ready else 'NO'}",
             C["good"] if env.scan_ready else C["bad"])
        sep()

        # Reward
        rc = (C["good"] if last_reward > 0 else
              C["bad"]  if last_reward < 0 else C["hud_dim"])
        text("REWARD", C["hud_text"], bold=True)
        text(f"Last    {last_reward:>+8.1f}", rc)
        text(f"Total   {self._cumulative_reward:>+8.1f}", C["info"])
        self._draw_sparkline(lm, y, HUD_W-20, 36)
        y += 42
        sep()

        # Mini-map
        text("MINI-MAP", C["hud_text"], bold=True)
        self._draw_minimap(env, lm, y, HUD_W-20, 80)
        y += 86
        sep()

        # Legend
        text("BIOME LEGEND", C["hud_dim"], font=self.font_xs)
        for name, col in [("Open savanna", C["savanna_l"]),
                           ("Woodland",    C["woodland_l"]),
                           ("Waterhole",   C["water_l"]),
                           ("Rocky",       C["rocky_l"])]:
            pygame.draw.rect(self.surf, col, (lm, y+1, 9, 9), border_radius=2)
            s = self.font_xs.render(f"  {name}", True, C["hud_dim"])
            self.surf.blit(s, (lm, y))
            y += 12

    def _draw_sparkline(self, x, y, w, h):
        pygame.draw.rect(self.surf, C["hud_panel"], (x, y, w, h), border_radius=3)
        pygame.draw.rect(self.surf, C["hud_border"], (x, y, w, h), 1, border_radius=3)
        data = list(self._reward_hist)
        if len(data) < 2:
            return
        mn, mx = min(data), max(data)
        span   = max(mx - mn, 1)
        pts    = []
        for i, v in enumerate(data):
            px = x + 1 + int(i / (len(data)-1) * (w-2))
            py = y + h - 2 - int((v - mn) / span * (h-4))
            pts.append((px, py))
        zero_y = y + h - 2 - int((0 - mn) / span * (h-4))
        pygame.draw.line(self.surf, C["hud_border"], (x+1, zero_y), (x+w-1, zero_y), 1)
        for i in range(1, len(pts)):
            col = C["good"] if data[i] >= 0 else C["bad"]
            pygame.draw.line(self.surf, col, pts[i-1], pts[i], 1)

    def _draw_minimap(self, env, x, y, w, h):
        cw = w / self.grid_size
        ch = h / self.grid_size
        pygame.draw.rect(self.surf, C["hud_panel"], (x, y, w, h), border_radius=2)
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                col = BIOME_BASE[env.biome_map[gx, gy]]
                px  = x + int(gy * cw)
                py  = y + int(gx * ch)
                pw  = max(1, int(cw))
                ph  = max(1, int(ch))
                pygame.draw.rect(self.surf, col, (px, py, pw, ph))
                if (gx, gy) in env.visited:
                    dim = pygame.Surface((pw, ph), pygame.SRCALPHA)
                    dim.fill((255, 255, 255, 45))
                    self.surf.blit(dim, (px, py))
        for t in env.threats:
            tx, ty = t["pos"]
            px = x + int(ty * cw + cw//2)
            py = y + int(tx * ch + ch//2)
            tc = C["alerted"] if t["alerted"] else THREAT_COL[t["type"]]
            pygame.draw.circle(self.surf, tc, (px, py), max(2, int(cw)))
        dx, dy = env.drone_pos
        px = x + int(dy * cw + cw//2)
        py = y + int(dx * ch + ch//2)
        pygame.draw.circle(self.surf, C["drone_body"], (px, py), max(2, int(cw)+1))
        pygame.draw.circle(self.surf, C["white"],      (px, py), max(2, int(cw)+1), 1)
        pygame.draw.rect(self.surf, C["hud_border"], (x, y, w, h), 1, border_radius=2)

    def close(self):
        if self._initialized:
            pygame.quit()
            self._initialized = False

    @property
    def MAX_STEPS(self):
        return 500


# --------------------------------------------------------------------------- #
# Standalone random-agent demo (Task 2a — no model)
# --------------------------------------------------------------------------- #

def run_random_demo(num_steps=300, seed=42):
    """
    Demonstrates the environment with purely random actions.
    No model or training — visualization only.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment.custom_env import SavannaAcousticEnv

    env = SavannaAcousticEnv(render_mode="human", seed=seed)
    obs, info = env.reset(seed=seed)

    renderer = SavannaRenderer(env.grid_size)
    renderer._init()   # must init before event loop on macOS

    print("=" * 60)
    print("  SAVANNA ACOUSTIC THREAT DETECTION — RANDOM AGENT DEMO")
    print("  (No model — pure random actions for visualization)")
    print("=" * 60)
    print(f"  Grid size    : {env.grid_size}x{env.grid_size}")
    print(f"  Action space : {env.action_space.n} discrete actions")
    print(f"  Obs shape    : {env.observation_space.shape}")
    print(f"  Threats      : {len(env.threats)}")
    print("  Press ESC or close window to quit")
    print("=" * 60)

    ACTION_NAMES = ["NORTH","SOUTH","EAST","WEST","SCAN","ALERT","HOVER"]
    total_reward = 0.0

    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        renderer.render(env, "human", last_reward=reward, last_action=action)

        if step % 50 == 0:
            print(f"  Step {step:>3} | {ACTION_NAMES[action]:<6} | "
                  f"Reward: {reward:>+7.1f} | Battery: {env.battery:.0f}% | "
                  f"Visited: {len(env.visited):>3}")

        if terminated or truncated:
            reason = ("battery depleted" if env.battery <= 0 else
                      "mission complete" if all(t["alerted"] for t in env.threats)
                      else "max steps")
            print(f"\n  Episode ended — {reason} at step {step}")
            print(f"  Total reward    : {total_reward:.1f}")
            print(f"  Threats resolved: {info['resolved_threats']}/{info['total_threats']}")
            obs, info = env.reset()
            renderer._trail.clear()
            total_reward = 0.0

    env.close()
    print("\nDemo complete.")


if __name__ == "__main__":
    run_random_demo(num_steps=500)