clear; close all; clc;

%% a)
I = 1:10;
lambda = 0.05;
mu = [5, 4];
sigma = [2, 3];
gauss_cdf1 = normcdf(I, mu(1), sigma(1));
gauss_cdf2 = normcdf(I, mu(2), sigma(2));
p1 = lambda/2 + (1-lambda) * gauss_cdf1;
p2 = lambda/2 + (1-lambda) * gauss_cdf2;

fig1 = figure();
plot(I, p1, 'DisplayName', 'p1', 'LineWidth', 2)
hold on;
plot(I, p2, 'DisplayName', 'p2', 'LineWidth', 2)
xlabel('Brightness (I)')
ylabel('P(red is brighter)')
title('Psychometric functions')
legend('Location', 'southeast')
