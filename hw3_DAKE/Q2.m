clear; clc; close all;

%%
n = linspace(-7, 7, 15);
sig1 = 1.5;
sig2 = 3.5;
G1 = exp(-(n.^2)/(2*sig1^2));
G2 = exp(-(n.^2)/(2*sig2^2));

G1 = G1/sqrt(sum(G1.^2));
G2 = G2/sqrt(sum(G2.^2));
dog = G1-G2;
norm(G2, 2)
plot(n, G1, 'DisplayName', 'gaussian 1');
hold on;
plot(n, G2, 'DisplayName', 'gaussian 2');
plot(n, dog, 'DisplayName', 'dog');
legend()

%%
fft(dog, 64)