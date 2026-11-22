"""
FACE CATCHER - a face-controlled AR game.

Glowing items fall down the screen. Move your head so your real mouth lines
up with them:

    CYAN ORB   - open your mouth to eat it      (+10)
    GOLD STAR  - open your mouth to eat it      (+30, falls faster)
    RED BOMB   - DODGE IT! touching your mouth costs a life

Eat items back-to-back to build a combo multiplier (up to x5). Letting an
orb or star fall past you resets the combo. You have 3 lives; the game gets
faster as your score grows.

Every 10th level a BOSS appears. It floats around and throws fast homing
bombs at your mouth. Pinch a bomb (thumb + index) to grab it, then bring
it to the boss to deal damage. Boss HP scales with progress.

When the game ends, a snapshot of your face at the moment of your last
life being lost is shown on the game-over screen.

Controls:
    SPACE - start / restart (and pause / resume while playing)
    M - toggle face mesh overlay
    C - cycle HUD color theme
    S - save a screenshot
    F - re-focus the window (use this if keypresses stop responding)
    ESC - back to main menu
    Q - quit
"""

import ctypes
import math
import random
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

WINDOW_NAME = "FACE CATCHER"

CAM_SOURCE = 0
# Laptop with a built-in webcam: leave as the device index above (0).
# Lab PC with no webcam: set to your phone's IP Webcam stream tunneled over
# USB via `adb forward tcp:8080 tcp:8080`, e.g.:
#   CAM_SOURCE = "http://localhost:8080/video"

THEMES = [
    {"name": "CYAN", "primary": (255, 255, 0), "accent": (0, 255, 255), "mesh": (255, 200, 0)},
    {"name": "MAGENTA", "primary": (255, 0, 255), "accent": (180, 0, 255), "mesh": (255, 100, 255)},
    {"name": "GREEN", "primary": (0, 255, 100), "accent": (0, 255, 0), "mesh": (0, 255, 150)},
    {"name": "AMBER", "primary": (0, 200, 255), "accent": (0, 165, 255), "mesh": (0, 220, 255)},
]

# MediaPipe Face Mesh landmark indices.
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263
LM_MOUTH_TOP = 13
LM_MOUTH_BOTTOM = 14

# Mouth openness = inner-lip gap / inter-eye distance. Two thresholds
# (hysteresis) stop the "open" state from flickering. If the game feels too
# hard or too easy to trigger, tweak MOUTH_OPEN_ON first.
MOUTH_OPEN_ON = 0.14
MOUTH_OPEN_OFF = 0.09

START_LIVES = 3

# How long (seconds) the "LEVEL UP" banner stays on screen.
LEVELUP_DURATION = 1.6

# How long (seconds) the "COMBO xN!" pop-up stays on screen.
COMBO_POP_DURATION = 1.0

# Difficulty ramp. Fall speed (px/s on a 720px-tall frame) = SPEED_BASE +
# SPEED_PER_LEVEL * (level - 1). A new level every POINTS_PER_LEVEL points.
# Raise SPEED_PER_LEVEL and lower POINTS_PER_LEVEL to speed up faster.
SPEED_BASE = 170
SPEED_PER_LEVEL = 40     # was 22
POINTS_PER_LEVEL = 100  # was 150

# Game-over face snapshot: output size in pixels and how much padding to keep
# around the detected face (1.0 = tight box, 1.5 = 50% extra).
SNAPSHOT_SIZE = 180
SNAPSHOT_PADDING = 1.5

ORB_COLOR = (255, 230, 0)     # BGR - cyan-ish
STAR_COLOR = (0, 215, 255)    # BGR - gold
BOMB_COLOR = (0, 0, 255)      # BGR - red
BOSS_COLOR = (180, 0, 255)    # BGR - purple

ITEM_TYPES = {
    "orb": {"points": 10, "radius": 22},
    "star": {"points": 30, "radius": 20},
    "bomb": {"points": 0, "radius": 24},
}

# Boss fight
BOSS_EVERY_N_LEVELS = 5
BOSS_BASE_HP = 3
BOSS_HP_PER_TIER = 2          # extra HP each boss (tier = level // 10)
BOSS_RADIUS = 48
BOSS_THROW_INTERVAL = 1.4    # seconds between homing bombs
BOSS_BOMB_SPEED = 320        # px/s on a 720p frame (scaled by h/720)
# Pinch: thumb-tip to index-tip distance / hand size (wrist→middle MCP).
# Real pinches often land around 0.08–0.20 depending on angle/distance to camera.
PINCH_ON = 0.22
PINCH_OFF = 0.42          # was 0.4 – slightly stickier while holding
PINCH_GRACE = 0.22        # seconds of “lost hand / open hand” allowed before releasing
GRAB_RANGE = 50          # px to grab a bomb (must beat bomb speed)
HIT_BOSS_RANGE = 90          # px to deal damage while holding a bomb near boss

# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------


def draw_hud_text(img, text, org, color, scale=0.55, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_center_text(img, text, y, color, scale=0.8, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - tw) // 2
    draw_hud_text(img, text, (x, y), color, scale, thickness)


def force_window_focus(window_title):
    """Windows blocks background processes from silently stealing focus, so a
    freshly-opened cv2 window sometimes never receives keyboard input even
    after being clicked. Explicitly activate it via the Win32 API.
    Windows only: on other systems ctypes.windll doesn't exist and this does nothing."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if hwnd:
            user32.ShowWindow(hwnd, 5)  # SW_SHOW
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def draw_orb(img, x, y, r, color):
    cv2.circle(img, (x, y), r + 8, color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), r + 3, color, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x - r // 4, y - r // 4), max(2, r // 3), (255, 255, 255), -1, cv2.LINE_AA)


def draw_star(img, x, y, r, color, angle):
    # 5-point star: 10 vertices alternating between a long (outer) and short (inner) radius.
    pts = []
    for i in range(10):
        rad = r * 1.35 if i % 2 == 0 else r * 0.6
        a = angle + i * math.pi / 5 - math.pi / 2
        pts.append([int(x + rad * math.cos(a)), int(y + rad * math.sin(a))])
    arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [arr], color, cv2.LINE_AA)
    cv2.polylines(img, [arr], True, (255, 255, 255), 2, cv2.LINE_AA)


def draw_bomb(img, x, y, r, angle, tick):
    # Dark body with 8 rotating spikes and a pulsing core (`tick` is game time in seconds).
    cv2.circle(img, (x, y), r, (30, 30, 30), -1, cv2.LINE_AA)
    for i in range(8):
        a = angle + i * math.pi / 4
        p1 = (int(x + r * math.cos(a)), int(y + r * math.sin(a)))
        p2 = (int(x + (r + 9) * math.cos(a)), int(y + (r + 9) * math.sin(a)))
        cv2.line(img, p1, p2, BOMB_COLOR, 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, BOMB_COLOR, 2, cv2.LINE_AA)
    pulse = 0.5 + 0.5 * math.sin(tick * 10)
    cv2.circle(img, (x, y), max(3, int(r * (0.25 + 0.2 * pulse))), (60, 60, 255), -1, cv2.LINE_AA)


def draw_boss(img, x, y, r, hp, max_hp, tick):
    """Simple floating skull-ish boss with HP bar."""
    pulse = 0.92 + 0.08 * math.sin(tick * 4)
    rr = int(r * pulse)
    cv2.circle(img, (x, y), rr + 10, BOSS_COLOR, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), rr, (40, 0, 60), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), rr, BOSS_COLOR, 3, cv2.LINE_AA)
    # Eyes
    eye_y = y - rr // 4
    for ex in (x - rr // 3, x + rr // 3):
        cv2.circle(img, (ex, eye_y), max(4, rr // 6), (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (ex, eye_y), max(2, rr // 12), (255, 255, 255), -1, cv2.LINE_AA)
    # Mouth
    cv2.ellipse(img, (x, y + rr // 4), (rr // 3, rr // 6), 0, 0, 180, (0, 0, 200), 2, cv2.LINE_AA)
    # HP bar above boss
    bar_w = max(60, rr * 2)
    bar_h = 8
    bx0 = x - bar_w // 2
    by0 = y - rr - 22
    cv2.rectangle(img, (bx0, by0), (bx0 + bar_w, by0 + bar_h), (40, 40, 40), -1)
    fill = int(bar_w * max(0, hp) / max(1, max_hp))
    cv2.rectangle(img, (bx0, by0), (bx0 + fill, by0 + bar_h), BOSS_COLOR, -1)
    cv2.rectangle(img, (bx0, by0), (bx0 + bar_w, by0 + bar_h), (255, 255, 255), 1)


# ----------------------------------------------------------------------
# Face / hand metrics
# ----------------------------------------------------------------------


def mouth_info(landmarks, img_w, img_h):
    """Returns (mouth_center_xy, inter_eye_distance_px, mouth_openness), with the
    center in pixel coordinates. Openness is the inner-lip gap divided by the
    inter-eye distance, so it holds up whether the player is near or far from
    the camera. The inter-eye distance is also used to size the hit zones.
    Returns None if the eyes are degenerately close (shouldn't happen with a real face)."""

    def pt(idx):
        lm = landmarks.landmark[idx]
        return lm.x * img_w, lm.y * img_h

    left_eye = pt(LM_LEFT_EYE_OUTER)
    right_eye = pt(LM_RIGHT_EYE_OUTER)
    iod = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    if iod < 1e-3:
        return None

    top = pt(LM_MOUTH_TOP)
    bottom = pt(LM_MOUTH_BOTTOM)
    center = ((top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2)
    openness = math.hypot(bottom[0] - top[0], bottom[1] - top[1]) / iod
    return center, iod, openness


def face_bbox(landmarks, img_w, img_h):
    """Pixel bounding box (x_min, y_min, x_max, y_max) of all face landmarks."""
    xs = [lm.x * img_w for lm in landmarks.landmark]
    ys = [lm.y * img_h for lm in landmarks.landmark]
    return min(xs), min(ys), max(xs), max(ys)


def hand_pinch_info(hand_landmarks, img_w, img_h):
    """Returns (pinch_center_xy, normalized_pinch_dist) from one hand.
    Pinch = thumb tip close to index tip, normalized by wrist-to-middle-mcp size.
    Also accepts a pure pixel threshold so foreshortened hands still work."""
    lm = hand_landmarks.landmark
    def pt(i):
        return lm[i].x * img_w, lm[i].y * img_h

    thumb = pt(4)   # THUMB_TIP
    index = pt(8)   # INDEX_FINGER_TIP
    wrist = pt(0)
    mid_mcp = pt(9)
    hand_size = math.hypot(mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1])
    if hand_size < 1e-3:
        hand_size = img_h * 0.15
    px_dist = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
    # Prefer normalized dist; also treat a tight pixel gap as a pinch
    # (helps when the hand is close to the camera and hand_size is large).
    norm = px_dist / hand_size
    if px_dist < 35:
        norm = min(norm, PINCH_ON * 0.5)
    center = ((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2)
    return center, norm


def make_snapshot(clean_frame, box, size=SNAPSHOT_SIZE, padding=SNAPSHOT_PADDING):
    """Crop a square around the face from a clean (no overlays) camera frame and
    resize it to size x size. Falls back to a center crop if there's no box."""
    h, w = clean_frame.shape[:2]
    if box is None:
        cx, cy, side = w / 2, h / 2, min(w, h) * 0.6
    else:
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(x1 - x0, y1 - y0) * padding
    side = max(40.0, min(side, min(w, h)))
    # Shift the crop window to stay inside the frame (keeps it square).
    left = int(round(min(max(cx - side / 2, 0), w - side)))
    top = int(round(min(max(cy - side / 2, 0), h - side)))
    s = int(side)
    crop = clean_frame[top:top + s, left:left + s]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def draw_snapshot(img, snapshot, center_x, top_y, color, label="LAST SEEN ALIVE"):
    """Draw the face snapshot with a themed frame and caption."""
    sh, sw = snapshot.shape[:2]
    x0 = center_x - sw // 2
    y0 = top_y
    img[y0:y0 + sh, x0:x0 + sw] = snapshot
    # Corner-bracket frame + thin border, to match the HUD look.
    cv2.rectangle(img, (x0 - 3, y0 - 3), (x0 + sw + 3, y0 + sh + 3), color, 1, cv2.LINE_AA)
    L = 22
    for (px, py, dx, dy) in (
        (x0 - 8, y0 - 8, 1, 1),
        (x0 + sw + 8, y0 - 8, -1, 1),
        (x0 - 8, y0 + sh + 8, 1, -1),
        (x0 + sw + 8, y0 + sh + 8, -1, -1),
    ):
        cv2.line(img, (px, py), (px + dx * L, py), color, 3, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, 3, cv2.LINE_AA)
    draw_center_text(img, label, y0 + sh + 28, color, 0.55, 1)


# ----------------------------------------------------------------------
# Game logic
# ----------------------------------------------------------------------


@dataclass
class Item:
    kind: str
    x: float
    y: float
    vy: float
    radius: int
    spin: float
    angle: float = 0.0
    dead: bool = False
    # Homing boss bombs
    vx: float = 0.0
    homing: bool = False
    grabbed: bool = False
    born: float = 0.0       # game clock when spawned (can't grab instantly)
    held_for: float = 0.0   # seconds held while pinched


class Game:
    def __init__(self):
        self.best = 0
        self.state = "START"  # START | PLAYING | GAME_OVER
        self.reset()

    def reset(self):
        self.score = 0
        self.lives = START_LIVES
        self.combo = 0
        self.max_combo = 0
        self.level = 1
        self.items = []
        self.particles = []  # [x, y, vx, vy, life, max_life, color]
        self.texts = []      # [text, x, y, life, color]
        self.spawn_timer = 1.0
        self.flash = 0.0
        self.clock = 0.0
        self.levelup_timer = 0.0  # > 0 while the LEVEL UP banner is showing
        self.combo_pop_timer = 0.0  # > 0 while the COMBO pop-up is showing
        self.combo_pop_mult = 1     # multiplier shown in that pop-up
        # Boss
        self.boss_active = False
        self.boss_x = 0.0
        self.boss_y = 0.0
        self.boss_vx = 0.0
        self.boss_vy = 0.0
        self.boss_hp = 0
        self.boss_max_hp = 0
        self.boss_throw_timer = 0.0
        self.boss_tier = 0
        self.grabbed_item = None  # Item currently held by pinch

    def start(self):
        self.reset()
        self.state = "PLAYING"

    def to_menu(self):
        """Abandon the current run and return to the main menu (best score is kept)."""
        self.reset()
        self.state = "START"

    @property
    def multiplier(self):
        return min(5, 1 + self.combo // 3)

    def _spawn(self, w, h):
        if self.boss_active:
            return
        scale = h / 720
        roll = random.random()
        bomb_chance = min(0.35, 0.2 + 0.02 * (self.level - 1))
        if roll < bomb_chance:
            kind = "bomb"
        elif roll < bomb_chance + 0.10:
            kind = "star"
        else:
            kind = "orb"

        speed = (SPEED_BASE + SPEED_PER_LEVEL * (self.level - 1) + random.uniform(-20, 40)) * scale
        if kind == "star":
            speed *= 1.3
        self.items.append(
            Item(
                kind=kind,
                x=random.uniform(0.12 * w, 0.88 * w),
                y=-30.0,
                vy=speed,
                radius=int(ITEM_TYPES[kind]["radius"] * scale),
                spin=random.uniform(-3, 3),
            )
        )

    def _start_boss(self, w, h):
        self.boss_active = True
        self.boss_tier = max(1, self.level // BOSS_EVERY_N_LEVELS)
        self.boss_max_hp = BOSS_BASE_HP + BOSS_HP_PER_TIER * (self.boss_tier - 1)
        self.boss_hp = self.boss_max_hp
        self.boss_x = w * 0.5
        self.boss_y = h * 0.22
        self.boss_vx = random.choice([-1, 1]) * 120 * (h / 720)
        self.boss_vy = 40 * (h / 720)
        self.boss_throw_timer = 0.8
        self.items = []  # clear falling items during boss
        self.grabbed_item = None
        self.levelup_timer = LEVELUP_DURATION
        self._burst(self.boss_x, self.boss_y, BOSS_COLOR, 40)
        self.texts.append(["BOSS FIGHT!", w * 0.5, h * 0.35, 1.5, BOSS_COLOR])

    def _end_boss(self, w, h):
        self.boss_active = False
        bonus = 100 * self.boss_tier
        self.score += bonus
        # Restore one life after defeating the boss (never above START_LIVES)
        if self.lives < START_LIVES:
            self.lives += 1
            self.texts.append(["+1 LIFE", self.boss_x, self.boss_y - 50, 1.4, (0, 255, 100)])
        self.items = [it for it in self.items if not it.homing]
        self.grabbed_item = None
        self._burst(self.boss_x, self.boss_y, STAR_COLOR, 50)
        self.texts.append([f"+{bonus} BOSS CLEAR", self.boss_x, self.boss_y, 1.4, STAR_COLOR])

    def _throw_homing_bomb(self, w, h, mouth):
        scale = h / 720
        speed = BOSS_BOMB_SPEED * scale
        mx, my = (mouth[0], mouth[1]) if mouth else (w * 0.5, h * 0.7)
        dx, dy = mx - self.boss_x, my - self.boss_y
        dist = math.hypot(dx, dy) or 1.0
        vx = speed * dx / dist
        vy = speed * dy / dist
        # Spawn slightly in front of the boss, toward the player — not on top of it
        sx = self.boss_x + 0.35 * BOSS_RADIUS * dx / dist
        sy = self.boss_y + 0.35 * BOSS_RADIUS * dy / dist
        self.items.append(
            Item(
                kind="bomb",
                x=sx,
                y=sy,
                vy=vy,
                vx=vx,
                radius=int(ITEM_TYPES["bomb"]["radius"] * scale),
                spin=random.uniform(-4, 4),
                homing=True,
                born=self.clock,
            )
        )

    def _burst(self, x, y, color, n=18):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            sp = random.uniform(80, 260)
            life = random.uniform(0.4, 0.9)
            self.particles.append([x, y, math.cos(a) * sp, math.sin(a) * sp, life, life, color])

    def update(self, dt, mouth, iod, mouth_open, w, h, pinch_pos=None, is_pinching=False):
        """Advance the game by dt seconds.

        mouth       - (x, y) mouth center in pixels, or None
        iod         - inter-eye distance in pixels (scales the hit zones)
        mouth_open  - True while the mouth counts as open
        w, h        - frame size in pixels
        pinch_pos   - (x, y) of pinch center, or None
        is_pinching - True while thumb+index are pinched
        """
        self.clock += dt
        prev_level = self.level
        self.level = 1 + self.score // POINTS_PER_LEVEL
        if self.level > prev_level:
            self.levelup_timer = LEVELUP_DURATION
            self._burst(0.5 * w, 0.3 * h, STAR_COLOR, 36)
            # Enter boss fight every 10th level
            if self.level % BOSS_EVERY_N_LEVELS == 0 and not self.boss_active:
                self._start_boss(w, h)
        # Only counts down while the game is actually running, so pausing
        # (or losing your face) doesn't eat the banner.
        self.levelup_timer = max(0.0, self.levelup_timer - dt)
        self.combo_pop_timer = max(0.0, self.combo_pop_timer - dt)

        # ---- Boss movement + throws ----
        if self.boss_active:
            self.boss_x += self.boss_vx * dt
            self.boss_y += self.boss_vy * dt
            # Bounce in upper area
            if self.boss_x < w * 0.15 or self.boss_x > w * 0.85:
                self.boss_vx *= -1
            if self.boss_y < h * 0.12 or self.boss_y > h * 0.38:
                self.boss_vy *= -1
            self.boss_x = max(w * 0.15, min(w * 0.85, self.boss_x))
            self.boss_y = max(h * 0.12, min(h * 0.38, self.boss_y))

            self.boss_throw_timer -= dt
            if self.boss_throw_timer <= 0:
                self._throw_homing_bomb(w, h, mouth)
                # Slightly faster throws on later bosses
                self.boss_throw_timer = max(0.7, BOSS_THROW_INTERVAL - 0.1 * (self.boss_tier - 1))

            # Pinch grab / deliver — boss ONLY takes damage from a held bomb
            # that was carried onto it (not from pinching the boss itself).
            if is_pinching and pinch_pos is not None:
                if self.grabbed_item is None:
                    # Grab nearest free bomb that has been in flight long enough
                    best, best_d = None, GRAB_RANGE
                    for it in self.items:
                        if (
                            it.kind == "bomb"
                            and not it.dead
                            and not it.grabbed
                            and (self.clock - it.born) > 0.35
                        ):
                            d = math.hypot(it.x - pinch_pos[0], it.y - pinch_pos[1])
                            if d < best_d:
                                best, best_d = it, d
                    if best is not None:
                        # Must grab away from the boss so you can't one-tap spawn-point bombs
                        d_boss = math.hypot(best.x - self.boss_x, best.y - self.boss_y)
                        if d_boss > BOSS_RADIUS + 30:
                            best.grabbed = True
                            best.homing = False
                            best.vx = 0
                            best.vy = 0
                            best.held_for = 0.0
                            self.grabbed_item = best
                else:
                    # Hold bomb at pinch
                    self.grabbed_item.x = pinch_pos[0]
                    self.grabbed_item.y = pinch_pos[1]
                    self.grabbed_item.vx = 0
                    self.grabbed_item.vy = 0
                    self.grabbed_item.held_for += dt
                    # Deliver only after carrying the bomb briefly onto the boss
                    d_boss = math.hypot(
                        self.grabbed_item.x - self.boss_x,
                        self.grabbed_item.y - self.boss_y,
                    )
                    if (
                        self.grabbed_item.held_for > 0.2
                        and d_boss < HIT_BOSS_RANGE + BOSS_RADIUS * 0.3
                    ):
                        self.grabbed_item.dead = True
                        self.grabbed_item = None
                        self.boss_hp -= 1
                        self._burst(self.boss_x, self.boss_y, STAR_COLOR, 24)
                        self.texts.append(["HIT!", self.boss_x, self.boss_y - 40, 0.8, STAR_COLOR])
                        if self.boss_hp <= 0:
                            self._end_boss(w, h)
            else:
                # Released pinch — drop bomb (falls downward)
                if self.grabbed_item is not None:
                    self.grabbed_item.grabbed = False
                    self.grabbed_item.homing = False
                    self.grabbed_item.vx = 0
                    self.grabbed_item.vy = 200 * (h / 720)
                    self.grabbed_item.held_for = 0.0
                    self.grabbed_item = None
        else:
            self.spawn_timer -= dt
            if self.spawn_timer <= 0:
                self._spawn(w, h)
                self.spawn_timer = max(0.4, 1.1 - 0.07 * (self.level - 1)) * random.uniform(0.8, 1.2)

        # ---- Item movement + collisions ----
        for it in self.items:
            if it.dead or it.grabbed:
                continue
            if it.homing and mouth is not None:
                # Mild course correction toward mouth (keeps them fast + chasey)
                scale = h / 720
                speed = BOSS_BOMB_SPEED * scale
                dx, dy = mouth[0] - it.x, mouth[1] - it.y
                dist = math.hypot(dx, dy) or 1.0
                # Blend current velocity toward target direction
                tx, ty = speed * dx / dist, speed * dy / dist
                it.vx = 0.85 * it.vx + 0.15 * tx
                it.vy = 0.85 * it.vy + 0.15 * ty
                it.x += it.vx * dt
                it.y += it.vy * dt
            else:
                it.y += it.vy * dt
                it.x += it.vx * dt
            it.angle += it.spin * dt

            if mouth is None:
                continue
            d = math.hypot(it.x - mouth[0], it.y - mouth[1])
            if it.kind == "bomb":
                # Bombs hurt whether or not the mouth is open. Their zone is a bit
                # smaller than the eating zone below, so near misses feel fair.
                hit_r = 0.40 * iod + it.radius * 0.6
                if it.homing:
                    hit_r = 0.35 * iod + it.radius * 0.5
                if d < hit_r:
                    it.dead = True
                    if self.grabbed_item is it:
                        self.grabbed_item = None
                    self.lives -= 1
                    self.combo = 0
                    self.flash = 0.5
                    self._burst(it.x, it.y, BOMB_COLOR, 28)
                    self.texts.append(["-1 LIFE", it.x, it.y, 1.0, (0, 0, 255)])
            elif not self.boss_active and mouth_open and d < 0.55 * iod + it.radius * 0.5:
                # Orbs and stars only count when the mouth is open (normal play).
                it.dead = True
                prev_mult = self.multiplier
                self.combo += 1
                self.max_combo = max(self.max_combo, self.combo)
                if self.multiplier > prev_mult:
                    self.combo_pop_timer = COMBO_POP_DURATION
                    self.combo_pop_mult = self.multiplier
                gained = ITEM_TYPES[it.kind]["points"] * self.multiplier
                self.score += gained
                color = STAR_COLOR if it.kind == "star" else ORB_COLOR
                self._burst(it.x, it.y, color)
                label = f"+{gained}" + (f"  x{self.multiplier}" if self.multiplier > 1 else "")
                self.texts.append([label, it.x, it.y, 1.0, color])

        survivors = []
        for it in self.items:
            if it.dead:
                continue
            if it.grabbed:
                survivors.append(it)
                continue
            if it.y > h + 50 or it.x < -80 or it.x > w + 80:
                if it.kind != "bomb":
                    self.combo = 0  # missed an orb/star
                if self.grabbed_item is it:
                    self.grabbed_item = None
                continue
            survivors.append(it)
        self.items = survivors

        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 300 * dt
            p[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]

        for t in self.texts:
            t[2] -= 60 * dt
            t[3] -= dt
        self.texts = [t for t in self.texts if t[3] > 0]

        self.flash = max(0.0, self.flash - dt)

        if self.lives <= 0:
            self.state = "GAME_OVER"
            self.best = max(self.best, self.score)

    def tick_effects(self, dt):
        """Keep particles, floating text and the red flash animating while the
        game itself isn't advancing (paused, face lost, start or game-over screen).
        Items and spawning stay frozen."""
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 300 * dt
            p[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]
        for t in self.texts:
            t[2] -= 60 * dt
            t[3] -= dt
        self.texts = [t for t in self.texts if t[3] > 0]
        self.flash = max(0.0, self.flash - dt)

    def draw(self, img, theme):
        for it in self.items:
            x, y = int(it.x), int(it.y)
            if it.kind == "orb":
                draw_orb(img, x, y, it.radius, ORB_COLOR)
            elif it.kind == "star":
                draw_star(img, x, y, it.radius, STAR_COLOR, it.angle)
            else:
                draw_bomb(img, x, y, it.radius, it.angle, self.clock)
                if it.grabbed:
                    cv2.circle(img, (x, y), it.radius + 12, STAR_COLOR, 2, cv2.LINE_AA)

        if self.boss_active:
            draw_boss(
                img, int(self.boss_x), int(self.boss_y),
                BOSS_RADIUS, self.boss_hp, self.boss_max_hp, self.clock,
            )

        for x, y, _vx, _vy, life, max_life, color in self.particles:
            r = max(1, int(4 * life / max_life))
            cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)

        for text, x, y, life, color in self.texts:
            draw_hud_text(img, text, (int(x) - 30, int(y)), color, 0.7, 2)

    def draw_levelup(self, img, theme):
        """Pop-up banner on level-up: the text grows in, holds, then fades out.
        It blends into the image, so the returned frame must be reassigned."""
        if self.levelup_timer <= 0:
            return img
        h, w = img.shape[:2]
        elapsed = LEVELUP_DURATION - self.levelup_timer
        pop = min(1.0, elapsed / 0.25)               # 0 -> 1 over the first 0.25 s
        alpha = min(1.0, self.levelup_timer / 0.5)   # fades out over the last 0.5 s
        scale = 0.9 + 0.7 * pop
        band_y = int(h * 0.30)

        # Translucent dark band behind the text.
        band = img.copy()
        cv2.rectangle(band, (0, band_y - 70), (w, band_y + 60), (0, 0, 0), -1)
        img = cv2.addWeighted(band, 0.55 * alpha, img, 1 - 0.55 * alpha, 0)

        # The text itself, faded in/out as a whole.
        overlay = img.copy()
        if self.boss_active and self.boss_hp == self.boss_max_hp and elapsed < 1.0:
            draw_center_text(overlay, f"BOSS  TIER {self.boss_tier}", band_y, BOSS_COLOR, scale, 3)
            draw_center_text(overlay, "PINCH BOMBS → DRAG TO BOSS", band_y + 40, theme["accent"], 0.7, 2)
        else:
            draw_center_text(overlay, f"LEVEL {self.level}", band_y, theme["primary"], scale, 3)
            draw_center_text(overlay, "SPEED UP!", band_y + 40, theme["accent"], 0.8, 2)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def draw_combo_pop(self, img, theme):
        """Pop-up when the combo multiplier goes up: quick pop-in, then fades.
        Sits below the level-up banner so the two don't overlap. Returns the
        blended frame, so the result must be reassigned."""
        if self.combo_pop_timer <= 0:
            return img
        h, w = img.shape[:2]
        elapsed = COMBO_POP_DURATION - self.combo_pop_timer
        pop = min(1.0, elapsed / 0.15)                # 0 -> 1 over the first 0.15 s
        alpha = min(1.0, self.combo_pop_timer / 0.35) # fades out over the last 0.35 s
        scale = 0.8 + 0.5 * pop
        label = f"COMBO x{self.combo_pop_mult}!"
        if self.combo_pop_mult >= 5:
            label = "MAX COMBO x5!"
        overlay = img.copy()
        draw_center_text(overlay, label, int(h * 0.48), STAR_COLOR, scale, 3)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    if isinstance(CAM_SOURCE, int):
        cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAM_SOURCE)
    else:
        cap = cv2.VideoCapture(CAM_SOURCE)  # URL source: let OpenCV pick the FFmpeg backend
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAM_SOURCE / camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    game = Game()
    show_mesh = True
    paused = False
    theme_idx = 0
    frame_count = 0
    shot_count = 0
    fps = 0.0
    prev_time = time.perf_counter()

    smoothed_mouth = None
    mouth_open = False
    face_snapshot = None  # face crop captured at the moment of game over
    pinching = False
    last_pinch_pos = None
    pinch_lost_timer = 0.0          # how long we’ve been without a solid pinch

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()  # untouched camera image, used for the game-over snapshot
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)

            now = time.perf_counter()
            dt = min(0.05, now - prev_time)  # clamp so a hiccup doesn't teleport items
            if now - prev_time > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time))
            prev_time = now

            theme = THEMES[theme_idx]

            # ---- face -> game inputs ----
            face_present = False
            iod = 0.0
            current_face_box = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                info = mouth_info(landmarks, w, h)
                if info is not None:
                    face_present = True
                    current_face_box = face_bbox(landmarks, w, h)
                    center, iod, openness = info
                    if smoothed_mouth is None:
                        smoothed_mouth = center
                    else:
                        # Average with the previous position to reduce landmark jitter
                        # (costs about one frame of lag).
                        smoothed_mouth = (
                            0.5 * smoothed_mouth[0] + 0.5 * center[0],
                            0.5 * smoothed_mouth[1] + 0.5 * center[1],
                        )
                    # Hysteresis: it takes a wider opening to turn "open" on than
                    # to keep it on, so the state doesn't flicker near the threshold.
                    if mouth_open:
                        mouth_open = openness > MOUTH_OPEN_OFF
                    else:
                        mouth_open = openness > MOUTH_OPEN_ON

            if not face_present:
                # No face this frame: forget the old position so the next
                # detection starts fresh instead of smoothing from a stale spot.
                smoothed_mouth = None
                mouth_open = False

            # ---- hand pinch (with short grace so bombs don’t drop on tracking flicker) ----
            pinch_pos = None
            hand_seen = False
            if hand_results.multi_hand_landmarks:
                info = hand_pinch_info(hand_results.multi_hand_landmarks[0], w, h)
                if info is not None:
                    hand_seen = True
                    center, dist = info
                    pinch_pos = center
                    last_pinch_pos = center
                    if pinching:
                        # Stay pinched until clearly open
                        pinching = dist < PINCH_OFF
                    else:
                        pinching = dist < PINCH_ON

            if hand_seen and pinching:
                pinch_lost_timer = 0.0
            else:
                # Counting time spent without a solid pinch (tracking lost OR fingers open)
                pinch_lost_timer += dt
                if pinch_lost_timer < PINCH_GRACE and last_pinch_pos is not None:
                    # Still inside the grace window – pretend we are still pinching
                    pinch_pos = last_pinch_pos
                    pinching = True
                else:
                    pinching = False
                    # After grace expires we keep last_pinch_pos for a while so the
                    # next successful detection can re-attach smoothly, but the bomb
                    # will already have been dropped by the game logic.

            # ---- face mesh overlay (semi-transparent) ----
            if show_mesh and results.multi_face_landmarks:
                overlay = frame.copy()
                for landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=theme["mesh"], thickness=1, circle_radius=1
                        ),
                    )
                    mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=theme["primary"], thickness=1, circle_radius=1
                        ),
                    )
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Hand outline during boss so the player can see the pinch point
            if game.boss_active and hand_results.multi_hand_landmarks:
                for hlm in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=theme["accent"], thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=theme["primary"], thickness=2),
                    )
                if pinch_pos is not None:
                    col = STAR_COLOR if pinching else theme["accent"]
                    cv2.circle(frame, (int(pinch_pos[0]), int(pinch_pos[1])), 18, col, 2, cv2.LINE_AA)
                    # Show grab radius so you can aim
                    cv2.circle(frame, (int(pinch_pos[0]), int(pinch_pos[1])), GRAB_RANGE, col, 1, cv2.LINE_AA)
                    label = "PINCH OK" if pinching else "OPEN HAND"
                    draw_hud_text(frame, label, (int(pinch_pos[0]) - 36, int(pinch_pos[1]) - 24), col, 0.5, 1)

            # ---- game update (only while playing, unpaused, with a face in view) ----
            was_playing = game.state == "PLAYING"
            if game.state == "PLAYING" and not paused and face_present:
                game.update(dt, smoothed_mouth, iod, mouth_open, w, h, pinch_pos, pinching)
            else:
                game.tick_effects(dt)

            # The game just ended on this frame: grab the player's face.
            if was_playing and game.state == "GAME_OVER":
                face_snapshot = make_snapshot(clean_frame, current_face_box)

            # ---- game draw ----
            game.draw(frame, theme)

            # ---- mouth reticle: the ring is the orb/star eating zone (0.55 * iod) ----
            if smoothed_mouth is not None and not game.boss_active:
                mx, my = int(smoothed_mouth[0]), int(smoothed_mouth[1])
                ring_color = theme["accent"] if mouth_open else theme["primary"]
                catch_r = int(0.55 * iod)
                cv2.circle(frame, (mx, my), catch_r, ring_color, 2, cv2.LINE_AA)
                cv2.circle(frame, (mx, my), 4, ring_color, -1, cv2.LINE_AA)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    cv2.line(
                        frame,
                        (mx + dx * (catch_r - 6), my + dy * (catch_r - 6)),
                        (mx + dx * (catch_r + 8), my + dy * (catch_r + 8)),
                        ring_color, 2, cv2.LINE_AA,
                    )
                draw_hud_text(
                    frame, "OPEN" if mouth_open else "CLOSED",
                    (mx - 28, my + catch_r + 24), ring_color, 0.5, 1,
                )
            elif smoothed_mouth is not None and game.boss_active:
                # Smaller dodge hint during boss
                mx, my = int(smoothed_mouth[0]), int(smoothed_mouth[1])
                cv2.circle(frame, (mx, my), 10, BOMB_COLOR, 2, cv2.LINE_AA)

            # ---- level-up banner ----
            if game.state == "PLAYING":
                frame = game.draw_levelup(frame, theme)
                frame = game.draw_combo_pop(frame, theme)

            # ---- red screen flash after a bomb hit (fades out over 0.5 s) ----
            if game.flash > 0:
                red = frame.copy()
                cv2.rectangle(red, (0, 0), (w, h), (0, 0, 255), -1)
                frame = cv2.addWeighted(red, min(0.45, game.flash), frame, 1 - min(0.45, game.flash), 0)

            # ---- top HUD bar ----
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
            draw_hud_text(frame, "FACE CATCHER", (12, 26), theme["primary"], 0.65, 2)
            draw_hud_text(frame, f"SCORE {game.score:05d}", (int(w * 0.24), 26), theme["accent"], 0.6, 2)
            if game.boss_active:
                draw_hud_text(
                    frame, f"BOSS HP {game.boss_hp}/{game.boss_max_hp}",
                    (int(w * 0.40), 26), BOSS_COLOR, 0.55, 2,
                )
            else:
                combo_text = f"COMBO x{game.multiplier}" if game.combo >= 3 else f"STREAK {game.combo}"
                draw_hud_text(frame, combo_text, (int(w * 0.40), 26), theme["accent"], 0.55)
            draw_hud_text(frame, f"LEVEL {game.level}", (int(w * 0.54), 26), theme["accent"], 0.55)
            for i in range(START_LIVES):
                cx = int(w * 0.66) + i * 30
                if i < game.lives:
                    cv2.circle(frame, (cx, 20), 9, BOMB_COLOR, -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, 20), 9, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (cx, 20), 9, (90, 90, 90), 1, cv2.LINE_AA)
            draw_hud_text(frame, f"BEST {game.best:05d}", (w - 170, 26), theme["primary"], 0.55)

            # ---- bottom HUD bar ----
            cv2.rectangle(frame, (0, h - 34), (w, h), (0, 0, 0), -1)
            draw_hud_text(frame, f"FPS: {fps:4.1f}", (12, h - 12), theme["accent"])
            draw_hud_text(frame, f"THEME: {theme['name']}", (150, h - 12), theme["accent"])
            hint = (
                "PINCH bomb → drag to boss   [SPACE] pause  [ESC] menu  [Q] quit"
                if game.boss_active
                else "[SPACE] start/pause [M] mesh [C] color [S] shot [F] focus [ESC] menu [Q] quit"
            )
            draw_hud_text(frame, hint, (w - 670, h - 12), (200, 200, 200), 0.45)

            # ---- state screens ----
            if game.state == "START":
                dim = frame.copy()
                cv2.rectangle(dim, (0, 40), (w, h - 34), (0, 0, 0), -1)
                frame = cv2.addWeighted(dim, 0.55, frame, 0.45, 0)
                draw_center_text(frame, "FACE CATCHER", h // 2 - 110, theme["primary"], 1.6, 3)
                draw_center_text(frame, "MOVE YOUR HEAD to line your mouth up with the falling items", h // 2 - 50, (255, 255, 255), 0.65, 1)
                draw_center_text(frame, "OPEN YOUR MOUTH to eat ORBS (+10) and STARS (+30)", h // 2 - 15, ORB_COLOR, 0.65, 1)
                draw_center_text(frame, "DODGE THE BOMBS - they cost a life", h // 2 + 20, BOMB_COLOR, 0.65, 1)
                draw_center_text(frame, "Every 10 levels: BOSS — pinch bombs and drag them onto it", h // 2 + 55, BOSS_COLOR, 0.6, 1)
                if int(time.time() * 2) % 2 == 0:
                    draw_center_text(frame, "PRESS SPACE TO START", h // 2 + 110, theme["accent"], 0.9, 2)
                draw_center_text(frame, "PRESS Q TO QUIT", h // 2 + 150, (200, 200, 200), 0.55, 1)
            elif game.state == "GAME_OVER":
                dim = frame.copy()
                cv2.rectangle(dim, (0, 40), (w, h - 34), (0, 0, 0), -1)
                frame = cv2.addWeighted(dim, 0.55, frame, 0.45, 0)
                # With a snapshot, push the text block down to make room for it.
                off = 130 if face_snapshot is not None else 0
                if face_snapshot is not None:
                    draw_snapshot(frame, face_snapshot, w // 2, h // 2 - 200, BOMB_COLOR)
                draw_center_text(frame, "SYSTEM FAILURE", h // 2 - 70 + off, BOMB_COLOR, 1.6, 3)
                draw_center_text(frame, f"SCORE {game.score}    BEST {game.best}    MAX COMBO {game.max_combo}", h // 2 - 15 + off, (255, 255, 255), 0.8, 2)
                if int(time.time() * 2) % 2 == 0:
                    draw_center_text(frame, "PRESS SPACE TO RETRY", h // 2 + 50 + off, theme["accent"], 0.9, 2)
                draw_center_text(frame, "ESC FOR MAIN MENU", h // 2 + 85 + off, (200, 200, 200), 0.55, 1)
            elif paused:
                draw_center_text(frame, "PAUSED", h // 2, theme["primary"], 1.4, 3)
                draw_center_text(frame, "PRESS SPACE TO RESUME", h // 2 + 40, (200, 200, 200), 0.6, 1)
            elif not face_present:
                draw_center_text(frame, "FACE LOST - LOOK AT THE CAMERA", h // 2, (0, 165, 255), 0.9, 2)

            cv2.imshow(WINDOW_NAME, frame)

            if frame_count == 5:
                # Give the window a few frames to exist before grabbing focus.
                force_window_focus(WINDOW_NAME)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 27:  # ESC -> back to the main menu (Q is the only way to quit)
                if game.state != "START":
                    game.to_menu()  # keeps the BEST score
                    paused = False
                    face_snapshot = None
            elif key == 32:  # SPACE: start / restart, or pause / resume while playing
                if game.state in ("START", "GAME_OVER"):
                    game.start()
                    paused = False
                    face_snapshot = None
                elif game.state == "PLAYING":
                    paused = not paused
            elif key == ord("m"):
                show_mesh = not show_mesh
            elif key == ord("c"):
                theme_idx = (theme_idx + 1) % len(THEMES)
            elif key == ord("s"):
                shot_count += 1
                filename = f"capture_{shot_count:02d}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
            elif key == ord("f"):
                force_window_focus(WINDOW_NAME)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()