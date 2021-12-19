clear; close all; clc;

%%
load('experimentData.mat')
load('newMeasurements.mat')

cond1_index = find(trialConds == 1);
cond2_index = find(trialConds == 2);
data1 = data(cond1_index, :);
data2 = data(cond2_index, :);

test1_index = find(newConds == 1);
test2_index = find(newConds == 2);
test1 = newMeasurements(test1_index, :);
test2 = newMeasurements(test2_index, :);

%% prototype classifier
mean_data1 = mean(data1);
mean_data2 = mean(data2);
w = mean_data2 - mean_data1;
w_norm = sqrt(sum(w.^2));
w_hat = w./w_norm;
midpoint_data = (mean_data1 + mean_data2)/2;

figure();
scatter3(data1(:, 1), data1(:, 2), data1(:, 3), 10, 'r', ...
    'filled', 'DisplayName', 'control odorant')
hold on;
scatter3(data2(:, 1), data2(:, 2), data2(:, 3), 10, 'k', ...
    'filled', 'DisplayName', 'pumpkin-spice odorant')
legend();
xlabel('Voxel 1'); ylabel('Voxel 2'); zlabel('Voxel 3');
title('Responses in three voxels across 2 trial conditions')

plot3([midpoint_data(1) - w_hat(1)/2, midpoint_data(1) + w_hat(1)/2], ...
    [midpoint_data(2) - w_hat(2)/2, midpoint_data(2) + w_hat(2)/2], ...
    [midpoint_data(3) - w_hat(3)/2, midpoint_data(3) + w_hat(3)/2], ...
    'm-', 'DisplayName', 'w hat', 'LineWidth', 1.5)
axis equal;

x_dec_boundary = xlim;
y_dec_boundary = ylim;
z_dec_boundary = decision_boundary(x_dec_boundary, y_dec_boundary, ...
    midpoint_data, w_hat);

pointsss = 1e2;
x_range = linspace(x_dec_boundary(1), x_dec_boundary(2), pointsss);
y_range = linspace(y_dec_boundary(1), y_dec_boundary(2), pointsss);
z_range = zeros(pointsss, 1);
for zz = 1:pointsss
    z_range(zz) = decision_boundary(x_range(zz), y_range(zz), midpoint_data, w_hat);
end

[x, y] = meshgrid(x_range, y_range);
z = zeros(pointsss, pointsss);
for zz = 1:pointsss
    for ff = 1:pointsss
        z(zz, ff) = decision_boundary(x(zz, ff), y(zz, ff), midpoint_data, w_hat);
    end
end
surf(x, y, z, 'FaceColor', '#EDB120', 'EdgeColor','#EDB120', 'FaceAlpha', ...
    0.5, 'EdgeAlpha', 0.5, 'DisplayName', 'Decision boundary');
set(gca, 'FontSize', 14)
set(gca, 'LineWidth', 2)
title('Prototype Classifier')
legend('Location', 'northeastoutside');

frac_correctly_classified_train = classification_performance(data1, ...
    data2, x_dec_boundary, y_dec_boundary, z_dec_boundary, w_hat)

frac_correctly_classified_test = classification_performance(test1, ...
    test2, x_dec_boundary, y_dec_boundary, z_dec_boundary, w_hat)

%% Fischer Linear Discriminant
lambdas = 0:0.05:1;
frac_correctly_classified_train_cv = zeros(length(lambdas), 1);
frac_correctly_classified_test_cv = zeros(length(lambdas), 1);

for ll = 1:length(lambdas)
    lambda = lambdas(ll);
    
    cov_data1 = cov(data1); cov_data2 = cov(data2);
    cov_combined = (cov_data1 + cov_data2)/2;

    cov_estimated = (1 - lambda) .* cov_combined + lambda .* eye(3);
    w_hat_estim = cov_estimated \ w';% \ cov_estimated;

    z_dec_boundary = decision_boundary(x_dec_boundary, y_dec_boundary, ...
        midpoint_data, w_hat_estim);

    frac_correctly_classified_train_cv(ll) = classification_performance(data1, ...
        data2, x_dec_boundary, y_dec_boundary, z_dec_boundary, w_hat_estim);

    frac_correctly_classified_test_cv(ll) = classification_performance(test1, ...
        test2, x_dec_boundary, y_dec_boundary, z_dec_boundary, w_hat_estim);
end

figure()
plot(lambdas, frac_correctly_classified_train_cv, 'ro-', 'DisplayName', 'train')
hold on;
plot(lambdas, frac_correctly_classified_test_cv, 'b*-', 'DisplayName', 'test')
xlabel('\lambda')
ylabel('proportion correctly classified')
title("Fischer's Linear Discriminant Classifier cross-validation")
legend()

%% QDA
figure()
scatter3(data1(:, 1), data1(:, 2), data1(:, 3), 10, 'r', ...
    'filled', 'DisplayName', 'control odorant')
hold on;
scatter3(data2(:, 1), data2(:, 2), data2(:, 3), 10, 'k', ...
    'filled', 'DisplayName', 'pumpkin-spice odorant')
legend();
xlabel('Voxel 1'); ylabel('Voxel 2'); zlabel('Voxel 3');
title('Responses in three voxels across 2 trial conditions')

xx_ = xlim;
yy_ = ylim;
zz_ = zlim;
pps = 40;
xx = linspace(xx_(1), xx_(2), pps);
yy = linspace(yy_(1), yy_(2), pps);
zz = linspace(zz_(1), zz_(2), pps);
[X, Y, Z] = meshgrid(xx, yy, zz);
XYZ = [X(:) Y(:) Z(:)];
p1 = mvnpdf(XYZ, mean_data1, cov_data1);
p2 = mvnpdf(XYZ, mean_data2, cov_data2);

diff_p = p1 - p2;
aa = abs(diff_p);
thresh = 0.01 * max(aa);
scatter3(XYZ(aa>thresh, 1), XYZ(aa>thresh, 2), XYZ(aa>thresh, 3), [], ...
    aa(aa>thresh), 'MarkerFaceAlpha', 1, ...
    'MarkerEdgeAlpha', 0.4)

p1_Quadratic = reshape(p1, length(zz), length(yy), length(xx));
p2_Quadratic = reshape(p2, length(zz), length(yy), length(xx));
%p3_Quadratic = reshape(p3, length(zz), length(yy), length(xx));
idx_Zgrid = zeros(1,length(zz)); 
%logDiff = (log(p1_Quadratic) - log(p2_Quadratic))<1e-3;
logDiff = (log(p1_Quadratic) - log(p2_Quadratic))<1e-3;
for kk = 1:length(xx)
    idx_Zgrid(kk) = length(yy) - find(flipud(logDiff(:,kk))==0,1);
end
%%
% pointsss = 1e2;
% 
% [x, y, z] = meshgrid(xx, yy, idx_Zgrid);
% xyz = [x(:) y(:) z(:)];
% surf(x, y, z, 'FaceColor', '#EDB120', 'EdgeColor','#EDB120', 'FaceAlpha', ...
%     0.5, 'EdgeAlpha', 0.5, 'DisplayName', 'Decision boundary');


% dec_x = zeros(length(diff_p), 1);
% dec_y = zeros(length(diff_p), 1);
% for i = 1:length(diff_p)
%     if abs(diff_p(i)) <= 1e-4
%         dec_x(i) = XY(i, 1);
%         dec_y(i) = XY(i, 2);
%     end
% end
% plot(dec_x, dec_y, 'k*')

%%
p1_train1 = mvnpdf(data1, mean_data1, cov_data1);
p2_train1 = mvnpdf(data1, mean_data2, cov_data2);
p1_train2 = mvnpdf(data2, mean_data1, cov_data1);
p2_train2 = mvnpdf(data2, mean_data2, cov_data2);

correct_train1 = sum(p1_train1 > p2_train1);
correct_train2 = sum(p2_train2 > p1_train2);

frac_correctly_classified_train = (correct_train1 + correct_train2)./...
    (size(data1, 1) + size(data2, 1))
%%
p1_test1 = mvnpdf(test1, mean_data1, cov_data1);
p2_test1 = mvnpdf(test1, mean_data2, cov_data2);
p1_test2 = mvnpdf(test2, mean_data1, cov_data1);
p2_test2 = mvnpdf(test2, mean_data2, cov_data2);

correct_test1 = sum(p1_test1 > p2_test1);
correct_test2 = sum(p2_test2 > p1_test2);

frac_correctly_classified_test = (correct_test1 + correct_test2)./...
    (size(test1, 1) + size(test2, 1))
%% Functions
function z_dec_boundary = decision_boundary(x_dec_boundary, y_dec_boundary, ...
    midpoint_data, w_hat)
    
    z_dec_boundary = midpoint_data(3) - (w_hat(2)/w_hat(3)) * (y_dec_boundary - ...
        midpoint_data(2)) - (w_hat(1)/w_hat(3)) * (x_dec_boundary - ...
        midpoint_data(2));
end

function frac_correctly_classified = classification_performance(data1, data2, ...
    x_dec_boundary, y_dec_boundary, z_dec_boundary, w_hat)
    
    x1_ones = ones(size(data1(:, 1))) .* x_dec_boundary(1);
    y1_ones = ones(size(data1(:, 1))) .* y_dec_boundary(1);
    z1_ones = ones(size(data1(:, 1))) .* z_dec_boundary(1);
    x2_ones = ones(size(data2(:, 1))) .* x_dec_boundary(1);
    y2_ones = ones(size(data2(:, 1))) .* y_dec_boundary(1);
    z2_ones = ones(size(data2(:, 1))) .* z_dec_boundary(1);
    
    z_data1 = z1_ones - (w_hat(2)/w_hat(3)) * (data1(:, 2) - y1_ones) - ...
        (w_hat(1)/w_hat(3)) * (data1(:, 1) - x1_ones);
    z_data2 = z2_ones - (w_hat(2)/w_hat(3)) * (data2(:, 2) - y2_ones) - ...
        (w_hat(1)/w_hat(3)) * (data2(:, 1) - x2_ones);
    correct_data1 = sum(z_data1 >= data1(:, 3));
    correct_data2 = sum(z_data2 <= data2(:, 3));
    frac_correctly_classified = (correct_data1 + correct_data2)./...
        (2 * size(data1, 1));
end