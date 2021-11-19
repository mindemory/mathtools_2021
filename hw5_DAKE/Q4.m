clear; close all; clc;

%% a)
mean_no_coher = 5;
var_no_coher = 1;
mean_10_coher = 8;
var_10_coher = 1;
samp_size = 1000;

resp_no_coher = mean_no_coher + var_no_coher * randn(samp_size, 1);
resp_no_coher_pos_setter = resp_no_coher > 0;
resp_no_coher_pos = resp_no_coher .* resp_no_coher_pos_setter;

resp_10_coher = mean_10_coher + var_10_coher * randn(samp_size, 1);
resp_10_coher_pos_setter = resp_10_coher > 0;
resp_10_coher_pos = resp_10_coher .* resp_10_coher_pos_setter;

fig1 = figure();
histogram(resp_no_coher, 'DisplayName', 'No coherence');
hold on;
histogram(resp_10_coher, 'DisplayName', '10% coherence');
legend('Location', 'northwest');
xlabel('Firing rate (spikes/s)')
ylabel('Frequency in sample')
title('Histogram of firing rates for different stimulus types')

%% b)
d_prime = abs(mean_10_coher - mean_no_coher)/sqrt(var_no_coher);
fprintf("The d' for the task and the pair of stimuli is %d\n", d_prime)

%% c)
thresholds = 0:15;
HR = zeros(length(thresholds), 1);
FAR = zeros(length(thresholds), 1);
for i = 1:length(thresholds)
    threshold = thresholds(i);
    hits = sum(resp_10_coher > threshold);
    misses = sum(resp_10_coher < threshold);
    false_alarms = sum(resp_no_coher > threshold);
    correct_rejs = sum(resp_no_coher < threshold);
    HR(i) = hits/(hits + misses);
    FAR(i) = false_alarms/(false_alarms + correct_rejs);
end
fig2 = figure();
plot(FAR, HR, 'ro-', 'LineWidth', 2)
xlabel('False alarm rate')
ylabel('Hit rate')
title('ROC Curve')