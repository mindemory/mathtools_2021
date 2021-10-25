clear; clc; close all;

%%
load('myMeasurements.mat')
figure();
plot(time, sig, 'ko-')
hold on;
time_subsampled = time(4:4:end);
sig_subsampled = sig(4:4:end);
plot(time_subsampled, sig_subsampled, 'r*-')
xlabel('Time (s)')
ylabel('Voltage (V)')
title('EEG Signal')

%%
N = length(sig);
N_space = linspace(-N/2, (N/2)-1, N); 
F_sig = fft(sig);
F_shift_sig = fftshift(sig);
size(F_sig)
size(F_shift_sig)
figure();
plot(N_space, abs(F_sig))
figure();
plot(N_space, abs(F_shift_sig))