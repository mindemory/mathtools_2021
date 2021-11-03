clear; close all; clc;

%% a)
load('experimentData.mat')
figure();
histogram(trialConds, 2);

figure();
scatter3(data(:, 1), data(:, 2), data(:, 3), [], [.5,.5,.5], 'filled')
xlabel('Voxel 1'); ylabel('Voxel 2'); zlabel('Voxel 3');
colormap jet