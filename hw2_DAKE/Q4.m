clear; clc; close all;

%% a)
load('windowedSpikes.mat')

plot(data')
xlabel('Time (ms)')
ylabel('Voltage (V)')
title('Windowed Spikes Visualization')
%%
mean_data = mean(data, 1);
centered_data = ones(size(data, 1), 1) * data
C = data'*data;
[V, D] = eig(C);
[d,ind] = sort(diag(D), 'descend'); 
D = D(ind,ind);
V = V(:,ind);