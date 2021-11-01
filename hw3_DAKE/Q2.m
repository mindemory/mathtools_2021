clear; clc; close all;

%%
sig1 = 1.5;
sig2 = 3.5;
n = linspace(-max(sig1, sig2)*2, max(sig1, sig2)*2, 15);

G1 = exp(-(n.^2)/(2*sig1^2));
G2 = exp(-(n.^2)/(2*sig2^2));

G1 = G1/sum(G1);
G2 = G2/sum(G2);
dog = G1-G2;

plot(n, G1, 'DisplayName', 'gaussian 1');
hold on;
plot(n, G2, 'DisplayName', 'gaussian 2');
plot(n, dog, 'DisplayName', 'dog');
xlabel('n')
ylabel('y(n)')
title('Guassians and DOG filter')
legend()

S1 = 0: .1: 5;
S2 = 0: .1: 5;

%%
F_dog = fft(dog, 64);
F_dog_amp = abs(F_dog);
figure();
plot(F_dog_amp);
title(sprintf('FFT with omega = %d', 64))

%% b)
[amp, freq] = max(F_dog_amp);
period = 1/freq;

%sine_wave = sin(