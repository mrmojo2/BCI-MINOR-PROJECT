#!/usr/bin/env python3
"""
Real-time time + frequency-domain plot of Arduino ADC over USB serial.
Expected serial line format (RECOMMENDED):
    millis,adc
Example:
    12345,512

If you only send "adc" (one integer per line), the script will fall back to an
assumed sampling rate (FS_FALLBACK).
"""

import time
import numpy as np
import serial
import matplotlib.pyplot as plt

# --------------------------
# Ubuntu / Linux serial port
# --------------------------
# Common Arduino ports on Ubuntu:
#   /dev/ttyACM0  (Uno, Mega often)
#   /dev/ttyUSB0  (some USB-serial adapters)
PORT = "/dev/ttyACM0"
BAUD = 115200

# If your Arduino does NOT send timestamps, we must assume fs.
# If your loop has delay(1), fs is ~1000 Hz but not exact.
FS_FALLBACK = 1000.0  # Hz

# Window/FFT settings
N = 2048                 # samples per FFT window (power of 2 recommended)
HOP = 128                # how often to refresh plots (in samples)
VREF = 5.0               # volts, adjust if using 3.3V board
ADC_BITS = 10            # Uno = 10-bit (0..1023). If different, adjust.

# --------------------------
# Helpers
# --------------------------
def adc_to_volts(adc):
    return (adc / ((2**ADC_BITS) - 1)) * VREF

def parse_line(line: str):
    """
    Returns (t_seconds or None, adc_int) from a line.
    Accepts either:
      - "millis,adc"
      - "adc"
    """
    line = line.strip()
    if not line:
        return None, None

    if "," in line:
        a, b = line.split(",", 1)
        try:
            t_ms = int(a.strip())
            adc = int(b.strip())
            return t_ms / 1000.0, adc
        except ValueError:
            return None, None
    else:
        try:
            adc = int(line)
            return None, adc
        except ValueError:
            return None, None

# --------------------------
# Serial setup
# --------------------------
ser = serial.Serial(PORT, BAUD, timeout=1)
# Give Arduino time to reset when serial opens
time.sleep(2.0)
ser.reset_input_buffer()

# --------------------------
# Buffers
# --------------------------
x = np.zeros(N, dtype=np.float64)          # signal (volts)
t = np.zeros(N, dtype=np.float64)          # time (seconds), only used if timestamps arrive
have_timestamps = False

# For fallback timing if no timestamps are sent:
t0 = time.time()
sample_count = 0

# Precompute Hann window (prevents spectral leakage)
hann = np.hanning(N)

# --------------------------
# Plot setup
# --------------------------
plt.ion()
fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 7))

line_time, = ax_time.plot(np.arange(N) / FS_FALLBACK, x)
ax_time.set_title("Time domain")
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Voltage (V)")
ax_time.grid(True)

freqs = np.fft.rfftfreq(N, d=1.0 / FS_FALLBACK)
mag = np.zeros_like(freqs)
line_freq, = ax_freq.plot(freqs, mag)
ax_freq.set_title("Frequency domain (Hann window + rFFT)")
ax_freq.set_xlabel("Frequency (Hz)")
ax_freq.set_ylabel("Magnitude (V)")
ax_freq.grid(True)

fig.tight_layout()

# --------------------------
# Main loop
# --------------------------
new_samples = 0
while True:
    raw = ser.readline().decode("utf-8", errors="ignore")
    t_s, adc = parse_line(raw)
    if adc is None:
        continue

    # Shift left by 1 and append new sample (simple ring behavior)
    x[:-1] = x[1:]
    x[-1] = adc_to_volts(adc)

    if t_s is not None:
        have_timestamps = True
        t[:-1] = t[1:]
        t[-1] = t_s
    else:
        # Synthesize time if no timestamps
        sample_count += 1

    new_samples += 1

    if new_samples >= HOP:
        new_samples = 0

        # Determine sampling rate
        if have_timestamps:
            # Estimate fs from timestamp differences (more accurate on Arduino)
            dt = np.diff(t)
            dt = dt[(dt > 0) & (dt < 1.0)]  # keep sane deltas
            fs = 1.0 / np.median(dt) if dt.size > 10 else FS_FALLBACK

            # Time axis relative to most recent window
            t_rel = t - t[-1]
            # If timestamps are monotonic, t_rel will be negative..0
            line_time.set_xdata(t_rel)
            ax_time.set_xlim(t_rel[0], t_rel[-1])
        else:
            fs = FS_FALLBACK
            time_axis = np.arange(N) / fs
            line_time.set_xdata(time_axis)
            ax_time.set_xlim(time_axis[0], time_axis[-1])

        line_time.set_ydata(x)
        ax_time.relim()
        ax_time.autoscale_view(scalex=False, scaley=True)

        # FFT with Hann window
        xw = x * hann
        X = np.fft.rfft(xw)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)

        # Amplitude scaling:
        # For a Hann window, coherent gain ~ 0.5 (mean(hann) ~ 0.5).
        # This helps make magnitudes more interpretable.
        coherent_gain = np.mean(hann)
        mag = (np.abs(X) / (N * coherent_gain)) * 2.0
        mag[0] *= 0.5  # DC shouldn't be doubled

        line_freq.set_xdata(freqs)
        line_freq.set_ydata(mag)
        ax_freq.set_xlim(0, fs / 2.0)
        ax_freq.relim()
        ax_freq.autoscale_view(scalex=False, scaley=True)

        fig.canvas.draw()
        fig.canvas.flush_events()

