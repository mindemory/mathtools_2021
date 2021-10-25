clear; clc; close all;

%%
sig1 = 1.5;
sig2 = 3.5;
n = linspace(-max(sig1, sig2)*2, max(sig1, sig2)*2, 15);

G1 = exp(-(n.^2)/(2*sig1^2));
G2 = exp(-(n.^2)/(2*sig2^2));

G1 = G1/sqrt(sum(G1.^2));
G2 = G2/sqrt(sum(G2.^2));
dog = G1-G2;
%norm(G2, 2)
plot(n, G1, 'DisplayName', 'gaussian 1');
hold on;
plot(n, G2, 'DisplayName', 'gaussian 2');
plot(n, dog, 'DisplayName', 'dog');
legend()

%%
F_dog = fft(dog, 64);
%F_G1 = fft(G1, 64);
%F_G2 = fft(G2, 64);
F_dog_amp = abs(F_dog);
figure();
%plot(abs(F_G1 - F_G2));
%plot(abs(F_G1)); hold on; plot(abs(F_G2));
%figure()
plot(F_dog_amp);
title('Amplitude of fft')

%% b)
[amp, freq] = max(F_dog_amp)