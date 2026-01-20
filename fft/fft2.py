import numpy as np
import serial
import matplotlib.pyplot as plt

PORT = "/dev/ttyACM0"
BAUD = 230400

FS = 1000.0
N = 2048
HOP = 256

ser = serial.Serial(PORT, BAUD, timeout=1)
ser.reset_input_buffer()

buf = np.zeros(N, dtype=np.float64)
hann = np.hanning(N)

plt.ion()
fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 7))

t_axis = np.arange(N) / FS
line_t, = ax_t.plot(t_axis, buf)
ax_t.set_title("Time domain")
ax_t.set_xlabel("Time (s)")
ax_t.set_ylabel("ADC (counts)")
ax_t.grid(True)

f_axis = np.fft.rfftfreq(N, d=1/FS)
line_f, = ax_f.plot(f_axis, np.zeros_like(f_axis))
ax_f.set_title("Frequency domain (Hann window)")
ax_f.set_xlabel("Frequency (Hz)")
ax_f.set_ylabel("Magnitude (counts)")
ax_f.grid(True)

fig.tight_layout()

pending = bytearray()
new = 0

while True:
    chunk = ser.read(4096)
    if not chunk:
        continue
    pending += chunk

    # parse complete 2-byte samples
    n_samples = len(pending) // 2
    if n_samples == 0:
        continue

    data = np.frombuffer(pending[:n_samples*2], dtype=np.uint16)
    pending = pending[n_samples*2:]

    for s in data:
        buf[:-1] = buf[1:]
        buf[-1] = float(s)
        new += 1

        if new >= HOP:
            new = 0

            # Time plot
            line_t.set_ydata(buf)
            ax_t.relim()
            ax_t.autoscale_view(scalex=False, scaley=True)

            # FFT (Hann)
            xw = (buf - np.mean(buf)) * hann
            X = np.fft.rfft(xw)
            coherent_gain = np.mean(hann)
            mag = (np.abs(X) / (N * coherent_gain)) * 2.0
            mag[0] *= 0.5

            line_f.set_ydata(mag)
            ax_f.set_xlim(0, FS/2)
            ax_f.relim()
            ax_f.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw()
            fig.canvas.flush_events()

