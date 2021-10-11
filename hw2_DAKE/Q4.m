clear; clc; close all;

%% a)
load('windowedSpikes.mat')

%%
% Plotting the data superimposed we see that there are spikes present in
% some waveforms that are absent in other. Overall there appear to be 3 
% unique spike patterns present which might indicate that the source of
% the spikes is likely due to 3 unique neurons.
figure(1);
plot(data')
xlabel('Time (ms)')
ylabel('Voltage (V)')
title('Windowed Spikes Visualization')

%%
% The pattern is much clearer when a random set of 20 waveforms is plotted.
figure(2);
plot(data(randi([1, size(data, 1)], 20, 1), :)')
xlabel('Time (ms)')
ylabel('Voltage (V)')
title('Windowed Spikes Visualization (20 random neurons)')


%% b)
% To perform PCA on the data, first the mean of each row of the data
% matrix is computed. The data is then centered by removing mean from each
% data point. The covariance matrix of the data can then be computed by
% multiplying the centered_data matrix with itself. Running
% eig(covariance_matrix) gives the eigenvectors (stored as columns of V)
% and their eigenvalues (stored as the diagonal elements of D). Sorting
% these eigen values and their corresponding eigen vectors in descending
% order, and plotting logarithms of the eigenvalues, we see that the
% eigenvalues decrease very quickly. The first three eigen values are much
% larger than the other eigenvalues. Hence it can be said that most of the
% variance in the dataset is present in the first three eigenvectors.
mean_data = mean(data, 1);
centered_data = data - ones(size(data, 1), 1) * mean_data;
C = centered_data' * centered_data;
[V, D] = eig(C);
[d,ind] = sort(diag(D), 'descend'); 
D = D(ind, ind);
V = V(:,ind);

figure(3);
plot(log(d), 'r-o', 'MarkerSize', 2)
xlabel('Ranking of the eigenvalue')
ylabel('log(eigenvalue)')
title('Eigenvalues in descending order')

%%
% Visualizing the first two PCs, we see that there appear three clusters in
% the dataset. Hence from this plot, we can say that the source of the
% spikes can be attributed to at least 3 neurons.
PC1 = V(:,1);
PC2 = V(:,2);

projPC1 = centered_data * PC1;
projPC2 = centered_data * PC2;
figure(4);
scatter(projPC1, projPC2, 's', 'filled', ...
    'MarkerEdgeColor','k', 'MarkerFaceColor',[0 .75 .75])
xlabel('PC1')
ylabel('PC2')
title('PCA of spike waveforms')

%%
% Visualizing the first three PCs, we see that there appear four clusters in
% the dataset. Hence from this plot, we can say that the source of the
% spikes can be attributed to at least 4 neurons.
%%
% Through this analysis, we can see that most of the variance in the
% dataset is captured by the first three PCs, plotting which we see four
% clusters. Hence we can inform Drs. Bell and Zell that the source of the
% spikes can be attributed to at least 4 different  neurons.
PC3 = V(:, 3);
projPC3 = centered_data * PC3;
figure(5);
scatter3(projPC1, projPC2, projPC3, 's', 'filled', ...
    'MarkerEdgeColor','k', 'MarkerFaceColor',[0 .75 .75])
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
title('PCA of spike waveforms')