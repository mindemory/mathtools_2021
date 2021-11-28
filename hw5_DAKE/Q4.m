clear; close all; clc;

%% a)
% The data can be simulated by drawing samples from a standard normal
% distribution and then scaling the distribution by the variance of the
% distribution of interest and finally translating it by the mean of
% interest. The resulting data will have been drawn from a Gaussian
% distribution of desired mean and variance.
%%
% Since the spikes of neurons cannot be negative in number, we can
% eliminate the values in the sample that are negative by first creating a
% Boolean array called 'setter' that has 0s if values are negative and 1s
% if values are positive. We can then compute an element-wise product of
% the actual sample data with this Boolean array to create a positive
% sample data.
%%
mean_no_coher = 5; % Mean of noise
std_no_coher = 1; % Standard deviation of noise
mean_10_coher = 8; % Mean of signal + noise
std_10_coher = 1; % Standard deviation of signal + noise
samp_size = 1000; % Sample size

resp_no_coher = mean_no_coher + std_no_coher^2 * randn(samp_size, 1); % Creating noise sample
resp_no_coher_pos_setter = resp_no_coher > 0; % Setter Boolean array
resp_no_coher = resp_no_coher .* resp_no_coher_pos_setter; % Excluding negative samples from noise

resp_10_coher = mean_10_coher + std_10_coher^2 * randn(samp_size, 1); % Creating signal + noise sample
resp_10_coher_pos_setter = resp_10_coher > 0; % Setter Boolean array
resp_10_coher = resp_10_coher .* resp_10_coher_pos_setter; % Excluding negative samples from signal + noise

fig1 = figure();
histogram(resp_no_coher, 'DisplayName', 'No coherence');
hold on;
histogram(resp_10_coher, 'DisplayName', '10% coherence');
legend('Location', 'northwest');
xlabel('Firing rate (spikes/s)')
ylabel('Frequency in sample')
title('Histogram of firing rates for different stimulus types')

%% b)
% The success of the decoder is given by sensitivity or $$d' $$. For two
% distributions that have same standard deviation, it is given using the
% formula:
%%
% $$d' = \frac{\mu_{S+N} - \mu_{N}}{\sigma^2} $$
%%
d_prime = abs(mean_10_coher - mean_no_coher)/std_no_coher;
fprintf("The d' for the task and the pair of stimuli is %d\n", d_prime)

%% c)
% The decoder over here either observes motion of dots or does not observe
% motion of dots. The probability of making correct guesses and incorrect
% guesses depends on where along the x-axis of the two distributions the
% criterion/threshold is set. Therefore, the maximum likelihood decoder for
% this problem will involve optimizing the threshold/criterion.
%%
thresholds = 0:0.01:10; % range of thresholds
hits = zeros(length(thresholds), 1); % Initializing hits
misses = zeros(length(thresholds), 1); % Initializing misses
false_alarms = zeros(length(thresholds), 1); % Initializing false-alarms
correct_rejs = zeros(length(thresholds), 1); % Initializing correct rejects
for i = 1:length(thresholds)
    threshold = thresholds(i);
    hits(i) = sum(resp_10_coher > threshold); % Computing the number of hits
    misses(i) = sum(resp_10_coher < threshold); % Computing the number of misses
    false_alarms(i) = sum(resp_no_coher > threshold); % Computing the number of false-alarms
    correct_rejs(i) = sum(resp_no_coher < threshold); % Computing the number of correct rejections
    %HR(i) = hits/(hits + misses); % Computing hit rate
    %FAR(i) = false_alarms/(false_alarms + correct_rejs); % Computing false-alarm rate
end
HR = hits./(hits + misses); % Computing hit rate for each threshold
FAR = false_alarms./(false_alarms + correct_rejs); % Computing false-alarm rate for each threshold

accuracy = (hits + correct_rejs)./(misses + false_alarms);
[max_accuracy, optim_threshold_index] = max(accuracy);
optim_threshold = thresholds(optim_threshold_index);

fig2 = figure();
plot(FAR, HR, 'r*-')
hold on;
plot(FAR(optim_threshold_index), HR(optim_threshold_index), 'bo')
xlabel('False alarm rate')
ylabel('Hit rate')
title('ROC Curve')


%%


%% d)
