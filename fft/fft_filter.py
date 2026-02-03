#!/usr/bin/env python3
import numpy as np
import serial
import matplotlib.pyplot as plt

# ------------------- SETTINGS -------------------
PORT = "/dev/ttyACM0"     # or /dev/ttyUSB0
BAUD = 230400

FS = 250.0               # MUST match Arduino FS
N = 2048                 # FFT / buffer length
HOP = 256                # plot update step

# ------------------------------------------------
ser = serial.Serial(PORT, BAUD, timeout=1)

# wait for Arduino READY line (recommended)
while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line == "READY":
        break

buf = np.zeros(N, dtype=np.float64)
hann = np.hanning(N)

# ------------------- PLOT SETUP -------------------
plt.ion()
fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 7))

t_axis = np.arange(N) / FS
line_t, = ax_t.plot(t_axis, buf)
ax_t.set_title("Time Domain (Filtered)")
ax_t.set_xlabel("Time (s)")
ax_t.set_ylabel("Amplitude")
ax_t.grid(True)

f_axis = np.fft.rfftfreq(N, d=1/FS)
line_f, = ax_f.plot(f_axis, np.zeros_like(f_axis))
ax_f.set_title("Frequency Domain (Hann window)")
ax_f.set_xlabel("Frequency (Hz)")
ax_f.set_ylabel("Magnitude")
ax_f.grid(True)

fig.tight_layout()

# ------------------- STREAM LOOP -------------------
new = 0
while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line:
        continue

    try:
        sample = float(line)
    except ValueError:
        continue

    # shift buffer and append
    buf[:-1] = buf[1:]
    buf[-1] = sample
    new += 1

    if new >= HOP:
        new = 0

        # ----- TIME DOMAIN -----
        line_t.set_ydata(buf)
        ax_t.relim()
        ax_t.autoscale_view(scalex=False, scaley=True)

        # ----- FREQUENCY DOMAIN -----
        x = buf - np.mean(buf)     # remove residual DC
        xw = x * hann

        X = np.fft.rfft(xw)
        coherent_gain = np.mean(hann)
        mag = (np.abs(X) / (N * coherent_gain)) * 2.0
        mag[0] *= 0.5

        line_f.set_ydata(mag)
        ax_f.set_xlim(0, FS / 2)
        ax_f.relim()
        ax_f.autoscale_view(scalex=False, scaley=True)

        fig.canvas.draw()
        fig.canvas.flush_events()

