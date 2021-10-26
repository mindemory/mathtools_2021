clear; clc; close all;

%%
load('hrfDeconv.mat')
subplot(2, 1, 1);
stem(x, 'filled')
subplot(2, 1, 2);
plot(r)