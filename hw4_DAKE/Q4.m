clear; close all; clc;

%% a)
load('experimentData.mat')
figure();
histogram(trialConds, 2);
cond1_index = find(trialConds == 1);
cond2_index = find(trialConds == 2);
cond1_data = data(cond1_index, :);
cond2_data = data(cond2_index, :);

figure();
scatter3(cond1_data(:, 1), cond1_data(:, 2), cond1_data(:, 3), 'r', ...
    'filled', 'DisplayName', 'Condition 1')
hold on;
scatter3(cond2_data(:, 1), cond2_data(:, 2), cond2_data(:, 3), 'k', ...
    'filled', 'DisplayName', 'Condition 2')
legend();
xlabel('Voxel 1'); ylabel('Voxel 2'); zlabel('Voxel 3');
colormap jet