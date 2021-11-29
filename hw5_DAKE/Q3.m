clear; close all; clc;

%% a)
I = 1:10; % Intensity of the red spot
lambda = 0.05; % Lapse rate
mu = [5, 4]; % Mean of Gaussian
sigma = [2, 3]; % Standard deviation of Gaussian

fig1 = figure();
for i = 1:length(mu)
    gauss_cdf = normcdf(I, mu(i), sigma(i));
    p = lambda/2 + (1-lambda) * gauss_cdf;
    plot(I, p, 'DisplayName', sprintf('p%d',i), 'LineWidth', 2)
    hold on;
end
xlabel('Brightness (I)')
ylabel('P(red is brighter)')
title('Psychometric functions')
legend('Location', 'southeast')

%%
% The two curves are sigmoidal or psychometric functions and saturate to 1
% as the intensity is increased. The two curves differ in their slopes.
% Curve p1 has a steeper slope compared to the curve p2. This implies that 
% the probability of the subject saying that "The red spot is brighter"
% increases rapidly with small changes in intensity of red in case of p1 but is
% rather slow in case of curve p2. We can also see that the curve p1 is in
% general above the curve p1 indicating that for the same intensity values,
% the probability of the subject saying that "The red spot is brighter" is
% higher in the case of p2.

%%
% Changing the mean of the psychometric function shifts the curve to the right
% along the brightness axis as increasing the mean results in increasing
% the threshold for deciding the response. This can be seen clearly in the
% figure below:
%%
I = 1:10;
lambda = 0.05;
mu = [1, 3, 6];
sigma = [2, 2, 2];

fig2 = figure();
for i = 1:length(mu)
    gauss_cdf = normcdf(I, mu(i), sigma(i));
    p = lambda/2 + (1-lambda) * gauss_cdf;
    plot(I, p, 'DisplayName', sprintf('Mean = %d',mu(i)), 'LineWidth', 2)
    hold on;
end
xlabel('Brightness (I)')
ylabel('P(red is brighter)')
title('Psychometric functions (varying mean)')
legend('Location', 'southeast')

%%
% Changing the standard deviation of the psychometric functions does not
% affect the threshold but affects the slope of the curve. Hence increasing
% the standard deviation would flatten the curve. Specifically, it will
% decrease the probability of calling "Red is brighter" at higher
% intensity levels but increase the probability of calling "Red is
% brighter" at lower intensity levels. This can be seen clearly in the
% figure below:

%%
I = 1:10;
lambda = 0.05;
mu = [3, 3, 3];
sigma = [1, 2, 4];

fig3 = figure();
for i = 1:length(mu)
    gauss_cdf = normcdf(I, mu(i), sigma(i));
    p = lambda/2 + (1-lambda) * gauss_cdf;
    plot(I, p, 'DisplayName', sprintf('Stdev = %d',sigma(i)), 'LineWidth', 2)
    hold on;
end
xlabel('Brightness (I)')
ylabel('P(red is brighter)')
title('Psychometric functions (varying standard deviation)')
legend('Location', 'southeast')

%%
% The range of p(I) is [0:1]. This range is appropriate because p(I) is a
% probability of the subject making a decision. The probability makes sense
% only in the range 0 to 1 and thus justifies the range of p(I).

%% b)
% The function simpsych is defined at the end of the file.

%% c)
T100 = ones(1, 7) * 100; % Length of trials for each intensity value
I100 = 1:7; % Intensities of red light
lambda100 = 0.05; % Lapse rate
mu100 = 4; % Mean of the Gaussian
sigma100 = 1; % Standard deviation of the Gaussian
B100 = simpsych(lambda100, mu100, sigma100, I100, T100); % Computing yesses for trials run for each intensity value

% Computing the psychometric curve
gauss_cdf100 = normcdf(I100, mu100, sigma100);
p100 = lambda100/2 + (1 - lambda100) * gauss_cdf100;

fig4 = figure();
plot(I100, p100, 'DisplayName', 'p(I)', 'LineWidth', 2)
hold on;
plot(I100, B100./T100, 'r*', 'DisplayName', 'B./T', 'LineWidth', 2)
xlabel('Brightness (I)')
ylabel('P(red is brighter)')
title(sprintf('Simulating simpsych (trials = %d)', 100))
legend('Location', 'southeast')

%% d)
T10 = ones(1, 7) * 10; % Length of trials for each intensity value
I10 = 1:7; % Intensities of red light
lambda10 = 0.05; % Lapse rate
mu10 = 4; % Mean of the Gaussian
sigma10 = 1; % Standard deviation of the Gaussian
B10 = simpsych(lambda10, mu10, sigma10, I10, T10); % Computing yesses for trials run for each intensity value

% Computing the psychometric curve
gauss_cdf10 = normcdf(I10, mu10, sigma10);
p10 = lambda10/2 + (1 - lambda10) * gauss_cdf10;

fig5 = figure();
plot(I10, p10, 'DisplayName', 'p(I)', 'LineWidth', 2)
hold on;
plot(I10, B10./T10, 'r*', 'DisplayName', 'B./T', 'LineWidth', 2)
xlabel('Brightness (I)')
ylabel('P(red is brighter)')
title(sprintf('Simulating simpsych (trials = %d)', 10))
legend('Location', 'southeast')

%%
% The second plot also samples from the same psychometric distribution but
% since the number of trials are less, there is more noise in sampling. As
% a result, the data points from the sample do not fit the psychometric
% curve as well in the second curve as they do in the first curve.

%% e)
% For each trial, the subject can either say 'yes' or 'no' with probability
% given by the psychometric function. Therefore, each trial follows a
% Bernoulli distribution with probability given by the psychometric
% function. Since the trials are repeated number of times, the sum of all
% these trials follows a Binomial distribution with p = p(I) and the k =
% B(lambda, mu, sigma, I, T). Here, lambda, sigma, I and T are known and 
% hence we are interested in computing the optimal mu given the data.
% Therefore, we are interested in P(data | mu) which as described follows a
% binomial distribution. Therefore, we can compute the likelihood for a
% range of mu using binopdf.
%%
mu_steps = 1:0.1:7; % Range of mu
I = 1:7; % Intensity of red
lambda = 0.05; % Lapse rate
sigma = 1; % Standard deviation of Gaussians
B10_loglikelihood = zeros(1, length(mu_steps));
B100_loglikelihood = zeros(1, length(mu_steps));

for i = 1:length(mu_steps)
    % Psychometric function for each mean
    gauss_cdf = normcdf(I, mu_steps(i), sigma);
    pI = lambda/2 + (1 - lambda) * gauss_cdf;
    
    % Computing log-likelihood for each mean
    B10_likelihood = binopdf(B10, T10, pI);
    B100_likelihood = binopdf(B100, T100, pI);
    B10_loglikelihood(:, i) = sum(log(B10_likelihood));
    B100_loglikelihood(:, i) = sum(log(B100_likelihood));
end

[B10_max_loglikelihood, B10_max_loglikelihood_index] = max(B10_loglikelihood); % maximum log-likelihood for 10 trials per intensity
[B100_max_loglikelihood, B100_max_loglikelihood_index] = max(B100_loglikelihood); % maximum log-likelihood for 100 trials per intensity

max_mu10 = mu_steps(B10_max_loglikelihood_index) % Maximum estimate of mean for 10 trials
max_mu100 = mu_steps(B100_max_loglikelihood_index) % Maximum estimate of mean for 100 trials

%%
% The maximum-likelihood estimate of mean agrees with the actual mean that
% was used to simulate the data. The mean estimate is closer to the actual
% mean when the number of trials is larger.

%% f)
% Using the same rationale as in (e), we can compute the maximum
% log-likelihood estimate for mean and standard deviation by computing
% log-likelihood for each mu and sigma and then computing the index in the
% resulting log-likelihood matrix of the maximum log-likelihood. The
% maximum estimates of mean and standard deviation will be given by the
% values of mean and standard deviation at the corresponding positions in
% their own arrays.
%%
mu_steps = 1:0.1:7; % Range of mu
I = 1:7; % Intensity of red
lambda = 0.05; % Lapse rate
sigma_steps = 0.1:0.1:2.5; % Standard deviation of Gaussians
B10_loglikelihood = zeros(length(sigma_steps), length(mu_steps));
B100_loglikelihood = zeros(length(sigma_steps), length(mu_steps));

for i = 1:length(mu_steps)
    for j = 1:length(sigma_steps)
        % Psychometric function for each mean and sigma
        gauss_cdf = normcdf(I, mu_steps(i), sigma_steps(j));
        pI = lambda/2 + (1 - lambda) * gauss_cdf;

        % Computing log-likelihood for each mean and sigma
        B10_likelihood = binopdf(B10, T10, pI);
        B100_likelihood = binopdf(B100, T100, pI);
        B10_loglikelihood(j, i) = sum(log(B10_likelihood));
        B100_loglikelihood(j, i) = sum(log(B100_likelihood));
    end
end

B10_max_loglikelihood = max(B10_loglikelihood, [], 'all'); % maximum log-likelihood for 10 trials per intensity
B100_max_loglikelihood = max(B100_loglikelihood, [], 'all'); % maximum log-likelihood for 100 trials per intensity

[B10_max_loglikelihood_row, B10_max_loglikelihood_column] = find(B10_loglikelihood == B10_max_loglikelihood); % row and index for maximum log-likelihood for 10 trials per intensity
[B100_max_loglikelihood_row, B100_max_loglikelihood_column] = find(B100_loglikelihood == B100_max_loglikelihood); % row and index for maximum log-likelihood for 100 trials per intensity

max_mu10 = mu_steps(B10_max_loglikelihood_column) % Maximum estimate of mean for 10 trials
max_mu100 = mu_steps(B100_max_loglikelihood_column) % Maximum estimate of mean for 100 trials

max_sigma10 = sigma_steps(B10_max_loglikelihood_row) % Maximum estimate of standard deviation for 10 trials
max_sigma100 = sigma_steps(B100_max_loglikelihood_row) % Maximum estimate of standard deviation for 100 trials

%%
% As can be seen, the maximum log-likelihood estimates of mean and sigma do
% agree with the actual mean = 4 and sigma = 1 used to simulate the data.

%% Function
function B = simpsych(lambda, mu, sigma, I, T)
    loop_size = length(I);
    B = zeros(1, loop_size); % Initializing B
    
    % Computing the psychometric curve for given parameters
    gauss_cdf = normcdf(I, mu, sigma);
    p = lambda/2 + (1 - lambda) * gauss_cdf;
    
    for k = 1:loop_size
        % Drawing uniform random values between 0 and 1 of length given by
        % T(k) and then determining if each of these random values is lower
        % than the probability of yes given by the psychometric curve. If
        % so, then the subject would respond as yes. Computing the number
        % of yesses in a given trial by summing the yesses just computed.
        trial_rand = rand(1, T(k));
        yesses = trial_rand < p(k);
        B(:, k) = sum(yesses);
    end
end

