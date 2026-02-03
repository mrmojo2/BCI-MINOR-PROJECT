#!/usr/bin/env python3
import time
import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =======================
# Serial / Sampling config
# =======================
PORT = "/dev/ttyACM0"        # Ubuntu: /dev/ttyACM0 or /dev/ttyUSB0
BAUD = 230400

FS = 256.0                  # MUST match your Arduino sampling rate
N = 512                    # FFT window length
HOP = 128                   # update step (samples)
READ_CHUNK = 4096           # bytes per read burst (ASCII is line-based so this is just a hint)

# =======================
# Detection tuning knobs
# =======================
# Blink: detect short transient using recent absolute deviation.
# Increase BLINK_THRESH if you get false blinks; decrease if it misses blinks.
BLINK_WIN_SEC = 0.18
BLINK_THRESH = 6.0          # in "robust z-score" units (6-10 typical)
BLINK_REFRACT_SEC = 0.40

# Focus: detect sustained rise in 10–35 Hz band power relative to baseline EMA.
FOCUS_BAND = (10.0, 35.0)
FOCUS_RATIO = 1.8           # focus when bandpower > baseline * ratio (1.4–2.5 typical)
FOCUS_HOLD_SEC = 0.45       # must stay high this long
FOCUS_REFRACT_SEC = 0.70
BASELINE_ALPHA = 0.02       # EMA smoothing for baseline (smaller = slower baseline)

# =======================
# Binary typing layout
# =======================
SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ _<"  # "_"=space, "<"=backspace

# =======================
# Helpers
# =======================
def robust_zscore(x: np.ndarray):
    """Robust z-score using median and MAD (less sensitive to outliers)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)

class BinaryTyper:
    def __init__(self, symbols: str):
        self.all_symbols = list(symbols)
        self.reset_group()
        self.highlight = 0  # 0 left, 1 right
        self.typed = ""

    def reset_group(self):
        self.group = self.all_symbols[:]
        self.path = []

    def split(self):
        mid = (len(self.group) + 1) // 2
        left = self.group[:mid]
        right = self.group[mid:]
        if not right and len(left) > 1:
            right = left[-1:]
            left = left[:-1]
        return left, right

    def toggle(self):
        self.highlight = 1 - self.highlight

    def select(self):
        left, right = self.split()
        chosen = left if self.highlight == 0 else right
        self.group = chosen
        self.path.append("L" if self.highlight == 0 else "R")
        self.highlight = 0

        if len(self.group) == 1:
            sym = self.group[0]
            if sym == "_":
                self.typed += " "
            elif sym == "<":
                self.typed = self.typed[:-1]
            else:
                self.typed += sym
            self.reset_group()

    def backspace(self):
        self.typed = self.typed[:-1]

    def previews(self, maxlen=20):
        left, right = self.split()
        def pv(lst):
            s = "".join(lst)
            return s if len(s) <= maxlen else (s[:maxlen] + "…")
        return pv(left), pv(right), len(self.group), "".join(self.path) if self.path else "(root)"

# =======================
# Matplotlib UI setup
# =======================
plt.ion()
fig = plt.figure(figsize=(12, 8))

ax_time = fig.add_axes([0.07, 0.58, 0.90, 0.34])
ax_freq = fig.add_axes([0.07, 0.26, 0.90, 0.26])
ax_ui   = fig.add_axes([0.07, 0.04, 0.90, 0.18])
ax_ui.axis("off")

# Buffers
buf = np.zeros(N, dtype=np.float64)
hann = np.hanning(N)

t_axis = np.arange(N) / FS
line_t, = ax_time.plot(t_axis, buf)
ax_time.set_title("Time domain (stream)")
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Amplitude (ADC units or filtered units)")
ax_time.grid(True)

f_axis = np.fft.rfftfreq(N, d=1/FS)
line_f, = ax_freq.plot(f_axis, np.zeros_like(f_axis))
ax_freq.set_title("Frequency domain (Hann window)")
ax_freq.set_xlabel("Frequency (Hz)")
ax_freq.set_ylabel("Magnitude")
ax_freq.grid(True)
ax_freq.set_xlim(0, FS/2)

# Typing UI patches/text
typer = BinaryTyper(SYMBOLS)

box_left  = Rectangle((0.02, 0.10), 0.46, 0.72, transform=ax_ui.transAxes, facecolor="#2d2d3c", edgecolor="black", linewidth=2)
box_right = Rectangle((0.52, 0.10), 0.46, 0.72, transform=ax_ui.transAxes, facecolor="#2d2d3c", edgecolor="black", linewidth=2)
ax_ui.add_patch(box_left)
ax_ui.add_patch(box_right)

txt_header = ax_ui.text(0.02, 0.92, "Binary Typing: Blink=toggle, Focus=select | Demo: ←/→ and Enter | Esc quit",
                        transform=ax_ui.transAxes, fontsize=11)
txt_typed  = ax_ui.text(0.02, 0.02, "", transform=ax_ui.transAxes, fontsize=16)
txt_left   = ax_ui.text(0.04, 0.38, "", transform=ax_ui.transAxes, fontsize=20)
txt_right  = ax_ui.text(0.54, 0.38, "", transform=ax_ui.transAxes, fontsize=20)
txt_status = ax_ui.text(0.52, 0.92, "", transform=ax_ui.transAxes, fontsize=11)

def refresh_ui(status=""):
    left_pv, right_pv, remaining, path = typer.previews()
    # Highlight colors
    if typer.highlight == 0:
        box_left.set_facecolor("#5a78ff")
        box_right.set_facecolor("#2d2d3c")
    else:
        box_left.set_facecolor("#2d2d3c")
        box_right.set_facecolor("#ffb24a")

    txt_left.set_text(f"LEFT:\n{left_pv}")
    txt_right.set_text(f"RIGHT:\n{right_pv}")
    txt_typed.set_text(f"Typed: {typer.typed[-60:]}")
    txt_status.set_text(f"Path: {path} | Remaining: {remaining} | {status}")

refresh_ui("waiting...")

# Keyboard demo controls
def on_key(event):
    if event.key == "escape":
        plt.close(fig)
        return
    if event.key in ("left", "right"):
        typer.toggle()
        refresh_ui("KEY toggle")
    elif event.key == "enter":
        typer.select()
        refresh_ui("KEY select")
    elif event.key == "backspace":
        typer.backspace()
        refresh_ui("KEY backspace")

fig.canvas.mpl_connect("key_press_event", on_key)

# =======================
# Serial open + sync
# =======================
ser = serial.Serial(PORT, BAUD, timeout=0)
time.sleep(0.15)
ser.reset_input_buffer()

# If your Arduino prints "READY", wait briefly but don't block forever
t0 = time.time()
while time.time() - t0 < 1.0:
    line = ser.readline().decode(errors="ignore").strip()
    if line == "READY":
        break

# =======================
# Detection state
# =======================
blink_last_t = 0.0
focus_last_t = 0.0
focus_high_start = None

baseline_bp = 1e-6  # EMA baseline of band power
blink_win = int(max(8, round(BLINK_WIN_SEC * FS)))

# =======================
# Main loop
# =======================
new = 0
status_msg = ""

while plt.fignum_exists(fig.number):
    # Read one line (ASCII sample)
    raw = ser.readline().decode("utf-8", errors="ignore").strip()
    if not raw:
        continue

    try:
        sample = float(raw)   # from your Arduino Serial.println(y,3)
    except ValueError:
        continue

    # Update circular-ish buffer (simple shift)
    buf[:-1] = buf[1:]
    buf[-1] = sample
    new += 1

    if new < HOP:
        continue
    new = 0

    now = time.time()

    # ========= Time plot update =========
    line_t.set_ydata(buf)
    ax_time.relim()
    ax_time.autoscale_view(scalex=False, scaley=True)

    # ========= FFT update =========
    x = buf - np.mean(buf)
    xw = x * hann
    X = np.fft.rfft(xw)
    coherent_gain = np.mean(hann)
    mag = (np.abs(X) / (N * coherent_gain)) * 2.0
    mag[0] *= 0.5

    line_f.set_ydata(mag)
    ax_freq.relim()
    ax_freq.autoscale_view(scalex=False, scaley=True)

    # ========= Feature extraction =========
    freqs = f_axis  # fixed
    f1, f2 = FOCUS_BAND
    band_mask = (freqs >= f1) & (freqs <= f2)

    # Use power (magnitude^2) for a more stable band-power measure
    band_power = float(np.sum((np.abs(X)[band_mask])**2))

    # Baseline EMA
    baseline_bp = (1 - BASELINE_ALPHA) * baseline_bp + BASELINE_ALPHA * band_power

    # -------- Blink detection (time-domain transient) --------
    # robust z-score over a short window near the end
    w = buf[-blink_win:]
    z = robust_zscore(w)
    blink_score = float(np.max(np.abs(z)))

    blink_evt = (blink_score >= BLINK_THRESH) and (now - blink_last_t >= BLINK_REFRACT_SEC)

    # -------- Focus detection (sustained band power rise) --------
    focus_ratio = band_power / (baseline_bp + 1e-12)
    focus_now_high = focus_ratio >= FOCUS_RATIO

    focus_evt = False
    if focus_now_high:
        if focus_high_start is None:
            focus_high_start = now
        elif (now - focus_high_start) >= FOCUS_HOLD_SEC and (now - focus_last_t) >= FOCUS_REFRACT_SEC:
            focus_evt = True
            focus_last_t = now
            focus_high_start = None
    else:
        focus_high_start = None

    # ========= Drive typing UI =========
    status_parts = []
    if blink_evt:
        typer.toggle()
        blink_last_t = now
        status_parts.append("BLINK toggle")
    if focus_evt:
        typer.select()
        status_parts.append("FOCUS select")

    status_msg = " | ".join(status_parts) if status_parts else f"blinkZ={blink_score:.1f} focusRatio={focus_ratio:.2f}"
    refresh_ui(status_msg)

    fig.canvas.draw()
    fig.canvas.flush_events()

try:
    ser.close()
except:
    pass

