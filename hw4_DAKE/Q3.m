clear; close all; clc;

%% a)
mu = [4, 5];
cy = [8, -5; -5, 4];
num = 1000;
samples = nRandn(mu, cy, num);
figure();
scatter(samples(1, :), samples(2, :))

%% b)
u_vec = rand(2, 1);
u_vec = u_vec/sqrt(sum(u_vec.^2));
samp_proj = u_vec' * samples;

centered_samp = samples - repmat(mu', 1, num);
cov_samples = (1/num) * (centered_samp * centered_samp');
angs = linspace(0, 2*pi, 48);
mu_proj_pred = zeros(length(angs), 1);
var_proj_pred = zeros(length(angs), 1);
mu_proj_act = zeros(length(angs), 1);
var_proj_act = zeros(length(angs), 1);

for i = 1:length(angs)
    u_vec = [cos(angs(i)), sin(angs(i))];
    samp_proj = u_vec * samples;
    
    mu_proj_pred(i) = u_vec * mu';
    var_proj_pred(i) = u_vec * cov_samples * u_vec';
    
    mu_proj_act(i) = mean(samp_proj);
    var_proj_act(i) = var(samp_proj);    
end

figure();
stem(angs, mu_proj_pred, 'r', 'DisplayName', 'Mean predicted');
hold on;
stem(angs, mu_proj_act, 'k', 'DisplayName', 'Mean actual');
xlabel('Angle (radians)')
ylabel('Mean')
title('Mean projection and actual')
legend();

figure();
stem(angs, var_proj_pred, 'r', 'DisplayName', 'Mean predicted');
hold on;
stem(angs, var_proj_act, 'k', 'DisplayName', 'Mean actual');
xlabel('Angle (radians)')
ylabel('Variance')
title('Variance projection and actual')
legend();

%% c)
samples_new = nRandn(mu, cy, 1000);
figure();
scatter(samples_new(1, :), samples_new(2, :))
hold on;
new_mu_pred = mean(samples_new, 2);
new_cov_pred = (1/num) * (samples_new * samples_new');

[V, D] = eig(cy);
M = V * sqrt(D);
circ_vects = [cos(angs); sin(angs)];
ellip_vects = repmat(mu', 1, length(angs)) + M * circ_vects;
% for i = 1:length(angs)
%     plot([mu(1), ellip_vects(1, i)], [mu(2), ellip_vects(2, i)], ...
%         'LineWidth', 1.5)
% end
plot(ellip_vects(1, :), ellip_vects(2, :), 'r', 'LineWidth', 2)
%%
for k =  1:3
    cov_generator = rand(2);
    cov_k = cov_generator * cov_generator.';
    mu_k = randi(10, 1, 2);
    samples_k = nRandn(mu_k, cov_k, num);
    figure();
    scatter(samples_k(1, :), samples_k(2, :))
    hold on;
    k_mu_pred = mean(samples_k, 2);
    k_cov_pred = (1/num) * (samples_k * samples_k');

    [V, D] = eig(cov_k);
    M = V * sqrt(D);
    circ_vects = [cos(angs); sin(angs)];
    ellip_vects = repmat(mu_k', 1, length(angs)) + M * circ_vects;
%     for i = 1:length(angs)
%         plot([mu_k(1), ellip_vects(1, i)], [mu_k(2), ellip_vects(2, i)], ...
%             'LineWidth', 1.5)
%     end
    plot(ellip_vects(1, :), ellip_vects(2, :), 'r', 'LineWidth', 2)
end

%%
function samples = nRandn(mean, cov, num)
    N = length(mean);
    [V, D] = eig(cov);
    M = V * sqrt(D);
    samp = randn(N, num);
    samples = repmat(mean', 1, num) + M * samp;
end