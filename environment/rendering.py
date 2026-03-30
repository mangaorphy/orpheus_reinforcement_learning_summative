"""
Savanna Acoustic Monitoring — OpenGL 3D Renderer
=================================================
True 3D environment using PyOpenGL + pygame:
  • 3D terrain blocks with per-biome heights and directional lighting
  • Tree canopy cylinders on woodland tiles
  • Semi-transparent acoustic heat overlay on terrain
  • 3D drone model with animated spinning rotors and shadow disc
  • Floating 3D threat markers with pulsing glow spheres
  • Expanding scan-wave rings in 3D space
  • Animated drone trail as a 3D line ribbon
  • HUD panel drawn to a pygame surface and composited via OpenGL texture
  • Perspective camera looking down over the savanna
  • Gradient sky quad rendered behind the scene
"""

import os
import sys
import math
import random
import pygame
import numpy as np
from collections import deque

try:
    from OpenGL.GL  import *
    from OpenGL.GLU import *
except ImportError:
    raise ImportError(
        "PyOpenGL is required for 3D rendering.\n"
        "Install with:  pip install PyOpenGL PyOpenGL_accelerate"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Window / camera constants
# ─────────────────────────────────────────────────────────────────────────────
WIN_W   = 1100
WIN_H   = 580
VIEW_W  = 800          # 3D viewport width; HUD fills the rest
HUD_W   = WIN_W - VIEW_W   # 300
FPS     = 8
TRAIL_LEN = 25

CAM_EYE    = (10.0, 20.0, 30.0)
CAM_CENTER = (10.0,  0.5, 10.0)
CAM_UP     = ( 0.0,  1.0,  0.0)

# Biome block heights (world units; 1 unit = 1 grid cell)
BIOME_H = {0: 0.25, 1: 0.75, 2: 0.12, 3: 1.10}

# Drone hover height above tile surface
DRONE_HOVER = 0.55

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (all RGB 0-255)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    # biome top / side
    "sav_top":    (215, 192, 108), "sav_side":   (168, 147,  76),
    "wood_top":   ( 92, 140,  58), "wood_side":  ( 58,  90,  32),
    "water_top":  ( 68, 158, 215), "water_side": ( 42, 105, 162),
    "rock_top":   (162, 150, 128), "rock_side":  (110, 100,  82),
    # trees on woodland
    "tree_trunk": ( 82,  58,  35), "tree_canopy":( 58,  98,  32),
    # threats / species
    "poaching":   (235,  55,  55), "predator":   (235, 145,  30),
    "habitat":    (150,  80, 215), "confirmed":  (255, 230,  30),
    "alerted":    ( 60, 220,  90),
    "elephant":   (180, 125,  65), "lion":       (235, 168,  35),
    "hyena":      (165,  95, 195), "zebra":      (235, 235, 235),
    # drone
    "drone_body": ( 40, 140, 255), "drone_arm":  ( 20,  90, 200),
    "drone_rotor":(175, 210, 255), "trail":      ( 80, 160, 255),
    # HUD (drawn on pygame surface)
    "hud_bg":     ( 12,  16,  24), "hud_panel":  ( 20,  28,  40),
    "hud_border": ( 45,  60,  80), "hud_text":   (220, 225, 235),
    "hud_dim":    (130, 140, 155), "good":       ( 70, 215,  95),
    "warn":       (215, 195,  45), "bad":        (215,  70,  70),
    "info":       ( 70, 160, 255),
    "white":      (255, 255, 255), "black":      (  0,   0,   0),
    # legend
    "savanna_l":  (215, 192, 108), "woodland_l": ( 92, 140,  58),
    "water_l":    ( 68, 158, 215), "rocky_l":    (162, 150, 128),
}

BIOME_TOP  = {0: C["sav_top"],   1: C["wood_top"],  2: C["water_top"], 3: C["rock_top"]}
BIOME_SIDE = {0: C["sav_side"],  1: C["wood_side"], 2: C["water_side"],3: C["rock_side"]}
THREAT_COL = {1: C["poaching"],  2: C["predator"],  3: C["habitat"]}
THREAT_LBL = {1: "POACH",        2: "PRED",          3: "HAB"}
SPECIES_COL= {0: C["elephant"],  1: C["lion"],       2: C["hyena"],    3: C["zebra"]}
SPECIES_LBL= {0: "ELEPH",        1: "LION",          2: "HYENA",       3: "ZEBRA"}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level OpenGL primitives
# ─────────────────────────────────────────────────────────────────────────────

def _gl_color(rgb, a=1.0):
    glColor4f(rgb[0]/255, rgb[1]/255, rgb[2]/255, a)


def draw_box(x1, y1, z1, x2, y2, z2, top_col, side_col):
    """Axis-aligned box with distinct top and side colours."""
    _gl_color(top_col)
    glBegin(GL_QUADS)                       # top face
    glNormal3f(0, 1, 0)
    glVertex3f(x1, y2, z1); glVertex3f(x2, y2, z1)
    glVertex3f(x2, y2, z2); glVertex3f(x1, y2, z2)
    glEnd()

    _gl_color(side_col)
    glBegin(GL_QUADS)
    glNormal3f(0, 0, 1)                     # front
    glVertex3f(x1,y1,z2); glVertex3f(x2,y1,z2); glVertex3f(x2,y2,z2); glVertex3f(x1,y2,z2)
    glNormal3f(0, 0, -1)                    # back
    glVertex3f(x2,y1,z1); glVertex3f(x1,y1,z1); glVertex3f(x1,y2,z1); glVertex3f(x2,y2,z1)
    glNormal3f(1, 0, 0)                     # right
    glVertex3f(x2,y1,z1); glVertex3f(x2,y1,z2); glVertex3f(x2,y2,z2); glVertex3f(x2,y2,z1)
    glNormal3f(-1, 0, 0)                    # left
    glVertex3f(x1,y1,z2); glVertex3f(x1,y1,z1); glVertex3f(x1,y2,z1); glVertex3f(x1,y2,z2)
    glEnd()

    # darker bottom (rarely seen but fills gaps)
    r,g,b = side_col[0]//2, side_col[1]//2, side_col[2]//2
    glColor4f(r/255,g/255,b/255,1)
    glBegin(GL_QUADS)
    glNormal3f(0,-1,0)
    glVertex3f(x1,y1,z2); glVertex3f(x2,y1,z2); glVertex3f(x2,y1,z1); glVertex3f(x1,y1,z1)
    glEnd()


def draw_cylinder(cx, yb, cz, radius, height, color, segs=12):
    """Solid cylinder with caps."""
    _gl_color(color)
    # side
    glBegin(GL_QUAD_STRIP)
    for i in range(segs + 1):
        a = 2*math.pi*i/segs
        nx, nz = math.cos(a), math.sin(a)
        glNormal3f(nx, 0, nz)
        glVertex3f(cx + radius*nx, yb+height, cz + radius*nz)
        glVertex3f(cx + radius*nx, yb,        cz + radius*nz)
    glEnd()
    # top cap
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0,1,0); glVertex3f(cx, yb+height, cz)
    for i in range(segs+1):
        a = 2*math.pi*i/segs
        glVertex3f(cx+radius*math.cos(a), yb+height, cz+radius*math.sin(a))
    glEnd()
    # bottom cap
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0,-1,0); glVertex3f(cx, yb, cz)
    for i in range(segs, -1, -1):
        a = 2*math.pi*i/segs
        glVertex3f(cx+radius*math.cos(a), yb, cz+radius*math.sin(a))
    glEnd()


def draw_sphere(cx, cy, cz, radius, color, stacks=10, slices=12):
    """Latitude-longitude sphere."""
    _gl_color(color)
    for i in range(stacks):
        lat0 = math.pi*(-0.5 + i/stacks)
        lat1 = math.pi*(-0.5 + (i+1)/stacks)
        s0, c0 = math.sin(lat0), math.cos(lat0)
        s1, c1 = math.sin(lat1), math.cos(lat1)
        glBegin(GL_QUAD_STRIP)
        for j in range(slices+1):
            lng = 2*math.pi*j/slices
            cl, sl = math.cos(lng), math.sin(lng)
            glNormal3f(cl*c0, s0, sl*c0)
            glVertex3f(cx+radius*cl*c0, cy+radius*s0, cz+radius*sl*c0)
            glNormal3f(cl*c1, s1, sl*c1)
            glVertex3f(cx+radius*cl*c1, cy+radius*s1, cz+radius*sl*c1)
        glEnd()


def draw_disc(cx, cy, cz, radius, color, segs=10):
    """Flat horizontal disc."""
    _gl_color(color)
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0,1,0); glVertex3f(cx, cy, cz)
    for i in range(segs+1):
        a = 2*math.pi*i/segs
        glVertex3f(cx+radius*math.cos(a), cy, cz+radius*math.sin(a))
    glEnd()


# ─────────────────────────────────────────────────────────────────────────────
# Renderer class
# ─────────────────────────────────────────────────────────────────────────────

class SavannaRenderer:

    def __init__(self, grid_size=20):
        self.grid_size          = grid_size
        self.screen             = None
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
        self._wave_rings        = []          # [cx, cz, y, r, max_r, alpha]
        self._rotor_angle       = 0.0
        self._cumulative_reward = 0.0
        self._hud_surf          = None
        self._hud_tex           = None

    # ── Init ──────────────────────────────────────────────────────────────── #

    def _init(self):
        if self._initialized:
            return
        pygame.init()
        pygame.display.set_caption(
            "Savanna Acoustic Threat Detection  |  AI Ranger Drone  [OpenGL 3D]"
        )
        self.screen = pygame.display.set_mode(
            (WIN_W, WIN_H), pygame.OPENGL | pygame.DOUBLEBUF
        )
        self.clock  = pygame.time.Clock()

        import platform, subprocess
        if platform.system() == "Darwin":
            subprocess.Popen(
                ["osascript", "-e",
                 f"tell application \"System Events\" to set frontmost of "
                 f"every process whose unix id is {os.getpid()} to true"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        self._setup_gl()

        # Pygame fonts (used to draw the HUD onto a surface)
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

        # HUD pygame surface and OpenGL texture
        self._hud_surf = pygame.Surface((HUD_W, WIN_H), pygame.SRCALPHA)
        self._hud_tex  = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self._initialized = True

    def _setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Warm directional light from upper-front-right
        glLightfv(GL_LIGHT0, GL_POSITION, [ 8.0, 30.0, -4.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.88, 0.82,  0.74, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.30, 0.30,  0.38, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.20, 0.20,  0.20, 1.0])

        glClearColor(0.06, 0.12, 0.26, 1.0)

    def _setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(48.0, VIEW_W / WIN_H, 0.5, 200.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*CAM_EYE, *CAM_CENTER, *CAM_UP)

    # ── Main render ────────────────────────────────────────────────────────── #

    def render(self, env, render_mode, last_reward=0.0, last_action=None):
        self._init()
        self._tick        += 1
        self._rotor_angle += 3.5        # degrees per frame
        self._last_action  = last_action

        if last_reward != 0:
            self._reward_hist.append(last_reward)
            self._cumulative_reward += last_reward

        if last_action == 4:            # SCAN — spawn wave ring
            row, col = env.drone_pos
            tile_h   = BIOME_H[int(env.biome_map[row, col])]
            self._wave_rings.append(
                [col + 0.5, row + 0.5, tile_h + DRONE_HOVER, 0.3, 6.0, 1.0]
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close(); sys.exit()

        self._trail.append(env.drone_pos)

        # ── 3D scene ──────────────────────────────────────────────────────
        glViewport(0, 0, VIEW_W, WIN_H)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera()

        self._draw_sky()
        self._draw_terrain(env)
        self._draw_decorations(env)
        self._draw_acoustic_overlay(env)
        self._draw_trail(env)
        self._draw_wave_rings()
        self._draw_threats(env)
        self._draw_drone(env)

        # ── HUD overlay ──────────────────────────────────────────────────
        glViewport(0, 0, WIN_W, WIN_H)
        self._build_hud_surface(env, last_reward)
        self._render_hud_overlay()

        pygame.display.flip()
        self.clock.tick(FPS)

        if render_mode == "rgb_array":
            pixels = glReadPixels(0, 0, WIN_W, WIN_H, GL_RGB, GL_UNSIGNED_BYTE)
            arr = np.frombuffer(pixels, dtype=np.uint8).reshape(WIN_H, WIN_W, 3)
            return arr[::-1]            # flip: OpenGL origin is bottom-left
        return None

    # ── Sky ───────────────────────────────────────────────────────────────── #

    def _draw_sky(self):
        """Gradient background quad (drawn in ortho, then restore 3D state)."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity(); glOrtho(-1,1,-1,1,-1,1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix(); glLoadIdentity()
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)

        # Oscillating day-cycle tint
        t = (math.sin(self._tick * 0.008) + 1) * 0.5
        top = (0.04+0.04*t, 0.10+0.06*t, 0.24+0.08*t, 1.0)
        bot = (0.10+0.10*t, 0.22+0.10*t, 0.45+0.08*t, 1.0)
        glBegin(GL_QUADS)
        glColor4f(*top); glVertex2f(-1, 1); glVertex2f(1, 1)
        glColor4f(*bot); glVertex2f(1, -1); glVertex2f(-1,-1)
        glEnd()

        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()

    # ── Terrain ───────────────────────────────────────────────────────────── #

    def _draw_terrain(self, env):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                biome = int(env.biome_map[row, col])
                h     = BIOME_H[biome]
                top_c = BIOME_TOP[biome]
                sid_c = BIOME_SIDE[biome]
                # Slightly brighten visited cells
                if (row, col) in env.visited:
                    top_c = tuple(min(255, v + 18) for v in top_c)
                draw_box(col, 0, row, col+1, h, row+1, top_c, sid_c)

    def _draw_decorations(self, env):
        """Small trees on woodland tiles."""
        rng = random.Random(42)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if int(env.biome_map[row, col]) != 1:
                    continue
                h = BIOME_H[1]
                # 1-2 trees per tile
                for _ in range(rng.randint(1, 2)):
                    ox = rng.uniform(0.2, 0.8)
                    oz = rng.uniform(0.2, 0.8)
                    tx, tz = col + ox, row + oz
                    draw_cylinder(tx, h, tz, 0.06, 0.40, C["tree_trunk"], segs=6)
                    draw_sphere(tx, h + 0.62, tz, 0.28, C["tree_canopy"], stacks=6, slices=8)

    def _draw_acoustic_overlay(self, env):
        """Semi-transparent heat quads just above each tile surface."""
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glBegin(GL_QUADS)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                v = float(env.acoustic_map[row, col])
                if v < 0.05:
                    continue
                biome = int(env.biome_map[row, col])
                y = BIOME_H[biome] + 0.04
                rv = min(1.0, v * 2.2)
                gv = min(1.0, v * 0.55)
                bv = max(0.0, 0.72 - v * 0.9)
                a  = v * 0.55
                glColor4f(rv, gv, bv, a)
                glNormal3f(0,1,0)
                glVertex3f(col,   y, row)
                glVertex3f(col+1, y, row)
                glVertex3f(col+1, y, row+1)
                glVertex3f(col,   y, row+1)
        glEnd()
        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)

    # ── Trail ─────────────────────────────────────────────────────────────── #

    def _draw_trail(self, env):
        pts = list(self._trail)
        if len(pts) < 2:
            return
        glDisable(GL_LIGHTING)
        glLineWidth(2.5)
        glBegin(GL_LINE_STRIP)
        for i, (row, col) in enumerate(pts):
            th = BIOME_H.get(int(env.biome_map[row, col]), 0.25)
            y  = th + DRONE_HOVER
            a  = (i / len(pts)) * 0.65
            r, g, b = C["trail"][0]/255, C["trail"][1]/255, C["trail"][2]/255
            glColor4f(r, g, b, a)
            glVertex3f(col + 0.5, y, row + 0.5)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    # ── Wave rings ────────────────────────────────────────────────────────── #

    def _draw_wave_rings(self):
        alive = []
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        for ring in self._wave_rings:
            cx, cz, y, r, max_r, alpha = ring
            if r < max_r and alpha > 0.02:
                ring[3] += 0.18
                ring[5]  = max(0, alpha - 0.04)
                segs = 32
                glColor4f(C["info"][0]/255, C["info"][1]/255, C["info"][2]/255, ring[5])
                glBegin(GL_LINE_LOOP)
                for i in range(segs):
                    a = 2*math.pi*i/segs
                    glVertex3f(cx + r*math.cos(a), y, cz + r*math.sin(a))
                glEnd()
                alive.append(ring)
        self._wave_rings = alive
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    # ── Threats ───────────────────────────────────────────────────────────── #

    def _draw_threats(self, env):
        for t in env.threats:
            row, col = t["pos"]
            biome    = int(env.biome_map[row, col])
            tile_h   = BIOME_H[biome]
            cx, cz   = col + 0.5, row + 0.5
            tc       = THREAT_COL[t["type"]]

            # Pole
            pole_col = C["alerted"] if t["alerted"] else tc
            draw_cylinder(cx, tile_h, cz, 0.035, 1.6, pole_col, segs=6)

            marker_y = tile_h + 1.7

            # Pulsing glow sphere for unresolved
            if not t["alerted"]:
                pulse  = (math.sin(self._tick * 0.18 + row) + 1) * 0.5
                glow_r = 0.26 + pulse * 0.10
                glow_a = 0.25 + pulse * 0.35
                glDisable(GL_LIGHTING); glDepthMask(GL_FALSE)
                r,g,b = tc[0]/255, tc[1]/255, tc[2]/255
                # Simple radial glow via transparent sphere
                draw_sphere(cx, marker_y, cz, glow_r,
                            (int(r*255), int(g*255), int(b*255)))
                glDepthMask(GL_TRUE); glEnable(GL_LIGHTING)

            fill = (C["alerted"]  if t["alerted"] else
                    C["confirmed"] if t["confirmed"] else tc)
            draw_sphere(cx, marker_y, cz, 0.18, fill)

    # ── Drone ─────────────────────────────────────────────────────────────── #

    def _draw_drone(self, env):
        row, col = env.drone_pos
        biome    = int(env.biome_map[row, col])
        tile_h   = BIOME_H[biome]
        cx       = col + 0.5
        cz       = row + 0.5
        cy       = tile_h + DRONE_HOVER

        # Shadow disc on tile
        glDisable(GL_LIGHTING); glDepthMask(GL_FALSE)
        glColor4f(0, 0, 0, 0.35)
        draw_disc(cx, tile_h + 0.01, cz, 0.30, (0, 0, 0))
        glDepthMask(GL_TRUE); glEnable(GL_LIGHTING)

        # Central body — flat octagon cylinder
        draw_cylinder(cx, cy - 0.05, cz, 0.18, 0.08, C["drone_body"], segs=8)

        # 4 arms at 45° angles with rotor tip discs
        arm_len = 0.42
        rotor_a = math.radians(self._rotor_angle)
        for k in range(4):
            base_a = math.radians(45 + 90 * k)
            ax = cx + arm_len * math.cos(base_a)
            az = cz + arm_len * math.sin(base_a)
            ay = cy

            # Arm box
            draw_box(
                cx + (arm_len*0.05)*math.cos(base_a) - 0.03,
                ay - 0.02,
                cz + (arm_len*0.05)*math.sin(base_a) - 0.03,
                ax + 0.03, ay + 0.02,
                az + 0.03,
                C["drone_arm"], C["drone_arm"],
            )

            # Rotor disc (animated spin)
            draw_disc(ax, ay + 0.03, az, 0.14, C["drone_rotor"])

            # Rotor blur lines
            glDisable(GL_LIGHTING)
            glLineWidth(1.5)
            glColor4f(*[v/255 for v in C["drone_rotor"]], 0.7)
            glBegin(GL_LINES)
            for blade in range(2):
                ba = rotor_a + math.pi * blade / 2
                glVertex3f(ax + 0.13*math.cos(ba),   ay+0.03, az + 0.13*math.sin(ba))
                glVertex3f(ax - 0.13*math.cos(ba),   ay+0.03, az - 0.13*math.sin(ba))
            glEnd()
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)

        # Battery indicator ring arc around body
        batt = env.battery / 100.0
        bc   = (C["good"] if batt > 0.5 else C["warn"] if batt > 0.25 else C["bad"])
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor4f(*[v/255 for v in bc], 1.0)
        segs = 32
        glBegin(GL_LINE_STRIP)
        for i in range(int(segs * batt) + 1):
            a = math.pi * 0.5 - 2*math.pi * i / segs
            glVertex3f(cx + 0.24*math.cos(a), cy + 0.05, cz + 0.24*math.sin(a))
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    # ── HUD (pygame surface → OpenGL texture) ─────────────────────────────── #

    def _build_hud_surface(self, env, last_reward):
        surf = self._hud_surf
        surf.fill(C["hud_bg"] + (255,))
        lm = 10
        y  = 0

        def text(t, color=C["hud_text"], font=None, bold=False):
            nonlocal y
            f = font or (self.font_md if bold else self.font_sm)
            s = f.render(str(t), True, color)
            surf.blit(s, (lm, y));  y += s.get_height() + 2

        def sep():
            nonlocal y
            pygame.draw.line(surf, C["hud_border"], (lm, y+2), (HUD_W-10, y+2), 1)
            y += 7

        def bar(val, mx, w, h, fg, bg_c=(40,50,65)):
            nonlocal y
            pygame.draw.rect(surf, bg_c, (lm, y, w, h), border_radius=3)
            filled = int(w * max(0, min(1, val/mx)))
            if filled > 0:
                pygame.draw.rect(surf, fg, (lm, y, filled, h), border_radius=3)
            y += h + 4

        # Title
        pygame.draw.rect(surf, C["hud_panel"], (0, 0, HUD_W, 28))
        ts = self.font_title.render("RANGER DRONE HUD", True, C["info"])
        surf.blit(ts, (lm, 7));  y = 34;  sep()

        # Mission
        text("MISSION", C["hud_text"], bold=True)
        text(f"Step   {env.step_count:>4} / 500", C["hud_dim"])
        bar(env.step_count, 500, HUD_W-20, 5, C["info"])
        bc = C["good"] if env.battery > 50 else C["warn"] if env.battery > 25 else C["bad"]
        text(f"Battery {env.battery:>5.1f}%", bc)
        bar(env.battery, 100, HUD_W-20, 6, bc)
        text(f"Pos    ({env.drone_pos[0]:>2}, {env.drone_pos[1]:>2})", C["hud_dim"])
        cov = len(env.visited)
        text(f"Cover  {cov:>3}/400  ({100*cov/400:.0f}%)", C["good"])
        bar(cov, 400, HUD_W-20, 5, C["good"])
        sep()

        # Threats
        text("THREATS", C["warn"], bold=True)
        res  = sum(1 for t in env.threats if t["alerted"])
        conf = sum(1 for t in env.threats if t["confirmed"])
        text(f"Alerted   {res}/{len(env.threats)}", C["good"] if res > 0 else C["bad"])
        text(f"Confirmed {conf}/{len(env.threats)}", C["warn"])
        text(f"Pending   {len(env.threats)-res}",
             C["bad"] if res < len(env.threats) else C["good"])
        for t in env.threats:
            st = "OK" if t["alerted"] else ("??" if t["confirmed"] else "!!")
            sc = C["good"] if t["alerted"] else C["warn"] if t["confirmed"] else C["bad"]
            s  = self.font_xs.render(
                f"  [{st}] {THREAT_LBL[t['type']]} "
                f"({t['pos'][0]:2},{t['pos'][1]:2}) {SPECIES_LBL[t['species']]}",
                True, sc)
            surf.blit(s, (lm, y));  y += s.get_height() + 2
        sep()

        # Acoustic
        ax, ay = env.drone_pos
        sig    = float(env.acoustic_map[ax, ay])
        sig_c  = C["good"] if sig > 0.5 else C["warn"] if sig > 0.2 else C["bad"]
        text("ACOUSTIC SIGNAL", C["hud_text"], bold=True)
        text(f"Intensity  {sig*100:.0f}%", sig_c)
        bar(sig, 1.0, HUD_W-20, 7, sig_c)
        text(f"Scan ready {'YES' if env.scan_ready else 'NO'}",
             C["good"] if env.scan_ready else C["bad"])
        sep()

        # Reward
        rc = C["good"] if last_reward > 0 else C["bad"] if last_reward < 0 else C["hud_dim"]
        text("REWARD", C["hud_text"], bold=True)
        text(f"Last    {last_reward:>+8.1f}", rc)
        text(f"Total   {self._cumulative_reward:>+8.1f}", C["info"])
        self._draw_sparkline(surf, lm, y, HUD_W-20, 34);  y += 40;  sep()

        # Mini-map
        text("MINI-MAP", C["hud_text"], bold=True)
        self._draw_minimap(surf, env, lm, y, HUD_W-20, 80);  y += 86;  sep()

        # Legend
        text("BIOME LEGEND", C["hud_dim"], font=self.font_xs)
        for name, col in [("Open savanna", C["savanna_l"]), ("Woodland", C["woodland_l"]),
                           ("Waterhole",   C["water_l"]),   ("Rocky",    C["rocky_l"])]:
            if y + 12 > WIN_H:
                break
            pygame.draw.rect(surf, col, (lm, y+1, 9, 9), border_radius=2)
            s = self.font_xs.render(f"  {name}", True, C["hud_dim"])
            surf.blit(s, (lm, y));  y += 12

    def _draw_sparkline(self, surf, x, y, w, h):
        pygame.draw.rect(surf, C["hud_panel"],  (x, y, w, h), border_radius=3)
        pygame.draw.rect(surf, C["hud_border"], (x, y, w, h), 1, border_radius=3)
        data = list(self._reward_hist)
        if len(data) < 2:
            return
        mn, mx = min(data), max(data); span = max(mx - mn, 1)
        pts = [(x+1+int(i/(len(data)-1)*(w-2)),
                y+h-2-int((v-mn)/span*(h-4))) for i,v in enumerate(data)]
        zero_y = y+h-2-int((0-mn)/span*(h-4))
        pygame.draw.line(surf, C["hud_border"], (x+1,zero_y), (x+w-1,zero_y), 1)
        for i in range(1, len(pts)):
            pygame.draw.line(surf, C["good"] if data[i]>=0 else C["bad"], pts[i-1], pts[i], 1)

    def _draw_minimap(self, surf, env, x, y, w, h):
        cw, ch = w/self.grid_size, h/self.grid_size
        pygame.draw.rect(surf, C["hud_panel"], (x, y, w, h), border_radius=2)
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                col = BIOME_TOP[int(env.biome_map[gx, gy])]
                px, py = x+int(gy*cw), y+int(gx*ch)
                pw, ph = max(1,int(cw)), max(1,int(ch))
                pygame.draw.rect(surf, col, (px, py, pw, ph))
                if (gx, gy) in env.visited:
                    dim = pygame.Surface((pw,ph), pygame.SRCALPHA)
                    dim.fill((255,255,255,45));  surf.blit(dim,(px,py))
        for t in env.threats:
            tx, ty = t["pos"]
            px = x+int(ty*cw+cw//2);  py = y+int(tx*ch+ch//2)
            tc = C["alerted"] if t["alerted"] else THREAT_COL[t["type"]]
            pygame.draw.circle(surf, tc, (px,py), max(2,int(cw)))
        dx, dy = env.drone_pos
        px = x+int(dy*cw+cw//2);  py = y+int(dx*ch+ch//2)
        pygame.draw.circle(surf, C["drone_body"], (px,py), max(2,int(cw)+1))
        pygame.draw.circle(surf, C["white"],      (px,py), max(2,int(cw)+1), 1)
        pygame.draw.rect(surf, C["hud_border"], (x, y, w, h), 1, border_radius=2)

    def _render_hud_overlay(self):
        """Upload HUD pygame surface as texture and render as ortho quad."""
        # Upload texture (flip vertically for OpenGL convention)
        flipped  = pygame.transform.flip(self._hud_surf, False, True)
        tex_data = pygame.image.tostring(flipped, "RGBA")
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     HUD_W, WIN_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)

        # Switch to ortho mode
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, WIN_W, 0, WIN_H, -1, 1)
        glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()

        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glColor4f(1, 1, 1, 1)

        hx = float(VIEW_W)
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(hx,      0)
        glTexCoord2f(1,0); glVertex2f(WIN_W,   0)
        glTexCoord2f(1,1); glVertex2f(WIN_W, WIN_H)
        glTexCoord2f(0,1); glVertex2f(hx,   WIN_H)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()

    # ── Cleanup ───────────────────────────────────────────────────────────── #

    def close(self):
        if self._initialized:
            if self._hud_tex is not None:
                glDeleteTextures([self._hud_tex])
            pygame.quit()
            self._initialized = False

    @property
    def MAX_STEPS(self):
        return 500


# ─────────────────────────────────────────────────────────────────────────────
# Standalone random-agent demo
# ─────────────────────────────────────────────────────────────────────────────

def run_random_demo(num_steps=300, seed=42):
    """Demonstrates the environment with purely random actions."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment.custom_env import SavannaAcousticEnv

    env = SavannaAcousticEnv(render_mode="human", seed=seed)
    obs, info = env.reset(seed=seed)

    renderer = SavannaRenderer(env.grid_size)
    renderer._init()

    print("=" * 60)
    print("  SAVANNA ACOUSTIC THREAT DETECTION — RANDOM AGENT DEMO")
    print("  OpenGL 3D Rendering")
    print("=" * 60)
    print(f"  Grid size    : {env.grid_size}x{env.grid_size}")
    print(f"  Action space : {env.action_space.n}")
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
