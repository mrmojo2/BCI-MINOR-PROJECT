% Rough measured frequency response (from your table)
% Vin = 40 mV (0.04 V). Gain in your table already uses this.

clear; close all; clc;

Fin  = [3 5 7 9 11 13 15 20 22 24 26 28 30];              % Hz
Vout = [1.36E-01 2.88E-01 4.24E-01 1.00 2.52 3.40 3.68 ...
        3.28 2.48 1.60 1.04 7.20E-01 5.00E-01];          % V

Vin = 0.04;                                               % V (40 mV)
Gain = Vout ./ Vin;                                       % linear gain
Gain_dB = 20*log10(Gain);                                 % magnitude in dB

% ---- Plot: linear magnitude ----
figure;
semilogx(Fin, Gain, "o-", "linewidth", 1.5, "markersize", 6);
grid on;
xlabel("Frequency (Hz)");
ylabel("Gain (V/V)");
title("Measured (Rough) Frequency Response - Linear Gain");

% ---- Plot: dB magnitude ----
figure;
semilogx(Fin, Gain_dB, "o-", "linewidth", 1.5, "markersize", 6);
grid on;
xlabel("Frequency (Hz)");
ylabel("Magnitude (dB)");
title("Measured (Rough) Frequency Response - Magnitude (dB)");

% Optional: smooth curve for a "rough" response shape (interpolation)
f_dense = logspace(log10(min(Fin)), log10(max(Fin)), 300);
gain_dense = interp1(Fin, Gain, f_dense, "pchip");

figure;
semilogx(Fin, Gain, "o", "markersize", 6); hold on;
semilogx(f_dense, gain_dense, "-", "linewidth", 1.5);
grid on;
xlabel("Frequency (Hz)");
ylabel("Gain (V/V)");
title("Measured Points + Smoothed (PCHIP) Rough Response");
legend("Measured", "Smoothed", "location", "best");
