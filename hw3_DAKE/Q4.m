clear; clc; close all;

%%
load('myMeasurements.mat')
figure();
plot(time, sig, 'ko-')
hold on;
time_subsampled = time(4:4:end);
sig_subsampled = sig(4:4:end);
plot(time_subsampled, sig_subsampled, 'r*-')

%%
N = length(sig);
N_space = linspace(-N/2, (N/2)-1, N); 
F_sig = fftshift(sig, 1);
figure();
plot(abs(F_sig))