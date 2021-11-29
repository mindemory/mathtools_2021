clear; close all; clc;

%% a)
% We are given that the conditional probability of activation given
% language and the conditional probabilty of activation given no language
% both follow a Bernoulli distribution. Since each experiment follows a
% Bernoulli distribution, the total set of studies would follow a Binomial
% distribution with N = total and X = activations observed for each
% condition. Using binopdf, we can then compute the likelihood of the range
% of parameters 0 to 1.
%%
lang_total = 869; % Total language studies
lang_activ = 103; % Language studies showing activation in Broca's area
nolang_total = 2353; % Total no language studies
nolang_activ = 199; % No language studies showing activation in Broca's area

xl = 0:0.001:1; % Range of probabilities for language
xnl = 0:0.001:1; % Range of probabilities for no language

lang_likelihood = binopdf(lang_activ, lang_total, xl); % Likelihood of xl for language data
nolang_likelihood = binopdf(nolang_activ, nolang_total, xnl); % Likelihood of xnl for no language data

fig1 = figure();
bar(xl, lang_likelihood, 'DisplayName', 'x_l');
hold on;
bar(xnl, nolang_likelihood, 'DisplayName', 'x_{nl}');
xlabel('x')
ylabel('p(data|x)')
title('Likelihood of parameters')
legend()

%% b)
% The paramter that maximizes each distribution can be computed by finding
% the index for maximum in each likelihood and then checking the index at
% the corresponding location in the parameter set.
%%
lang_max_index = find(lang_likelihood == max(lang_likelihood)); % Index of maximum lang likelihood
nolang_max_index = find(nolang_likelihood == max(nolang_likelihood)); % Index of maximum nolang likelihood
lang_maxlikelihood_pred = xl(lang_max_index) % The parameter value at maximum for language
nolang_maxlikelihood_pred = xnl(nolang_max_index) % The parameter value at maximum for no language

%%
% The pdf of a Bernoulli random variable (X) can be written as:
%%
% $$f(X) = p^X(1-p)^{(1-X)} $$
%%
% where, X = 0 or 1 in this case no language or language.
%%
% The likelihood function then becomes:
%%
% $$L(p) = \prod_{i = 1}^N p^{X_i}(1-p)^{(1-X_i)} $$
%%
% The log-likelihood function can be obtained by taking the logarithm of
% both sides. Thus,
%%
% $$LL(p) = log(\prod_{i = 1}^N p^{X_i}(1-p)^{(1-X_i)}) $$
%%
% Therefore,
%%
% $$LL(p) = \sum_{i = 1}^N log(p^{X_i}(1-p)^{(1-X_i)}) $$
%%
% Therefore,
%%
% $$LL(p) = \sum_{i = 1}^N X_i log p + (1-X_i)log(1-p) $$
%%
% Let $$Y = \sum_{i = 1}^N X_i $$
%%
% Therefore,
%%
% $$LL(p) = Y log p + (N-Y)log(1-p) $$
%%
% The aim is to compute the maximum likelihood estimation. Hence we are
% interested in the value of p for which the derivative of LL(p) equates to
% 0.
%%
% Therefore,
%%
% $$\frac{\partial LL(p)}{\partial p} = \frac{\partial }{\partial p} (Y log p + (N-Y)log(1-p)) = 0 $$
%%
% Therefore,
%%
% $$\frac{Y}{p} + \frac{Y-N}{1-p} = 0 $$
%%
% Therefore,
%%
% $$Y - pY = pN - pY $$
%%
% Therefore,
%%
% $$Y = pN $$
%%
% Therefore,
%%
% $$\hat{p} = \frac{Y}{N} $$
%%
% Therefore,
%%
% $$\hat{p} = \frac{\sum_{i=1}^N X_i}{N} $$
%%
% In our example, Y is the active instances and N is the total instances
% for each condition. Computing the estimated maximum likelihoods:
%%
lang_maxlikelihood_estim = lang_activ/lang_total
nolang_maxlikelihood_estim = nolang_activ/nolang_total

%%
% As can be seen, the exact maximum likelihood estimates given by the
% formula for ML estimator of a Bernoulli probability agrees with the
% estimates obtained from the computed maximum likelihood estimate.

%% c)
% The next step is to compute $$P(x | data) $$
%%
% Using Bayesian formula, we have:
%%
% $$P(x_i | data) = \frac{P(data | x_i) P(x_i)}{\sum_{i = 1}^N P(data | x_i) P(x_i)} $$
%%
% The denominator can be obtained as a dot product between the likelihood
% and px for each condition. The posterior can then be computed
% as an element-wise product between the likelihood and px and the
% dividing the result by the denominator computed. Here, since the prior is
% uniform, we have:
%%
% $$P(x_i) = \frac{1}{N} $$
%%
px = (1/length(xl)) * ones(1, length(xl)); % uniform prior
lang_denom = lang_likelihood * px'; % normalizing denominator for computing posterior for language
nolang_denom = nolang_likelihood * px'; % normalizing denominator for computing posterior for no language
lang_posterior = (lang_likelihood .* px) / lang_denom; % Posterior for language
nolang_posterior = (nolang_likelihood .* px) / nolang_denom; % Posterior for no language

fig2 = figure();
plot(xl, lang_posterior, 'DisplayName', 'x_l');
hold on;
plot(xnl, nolang_posterior, 'DisplayName', 'x_{nl}');
xlabel('x')
ylabel('p(x|data)')
title('Posterior distributions')
legend()

%%
% The cumulative posterior distributions can be obtained by summing all the 
% posterior distributions upto the given parameter index. The cumulative
% posterior distributions will be monotonically increasing functions from 0
% to 1.
%%
lang_cumulative = zeros(1, length(xl)); % Initializing language cumulative
nolang_cumulative = zeros(1, length(xl)); % Initializing no language cumulative

for i = 1:length(xl)
    if i == 1
        lang_cumulative(i) = lang_posterior(i);
        nolang_cumulative(i) = nolang_posterior(i);
    else
        lang_cumulative(i) = lang_cumulative(i-1) + lang_posterior(i);
        nolang_cumulative(i) = nolang_cumulative(i-1) + nolang_posterior(i);
    end
end

fig3 = figure();
plot(xl, lang_cumulative, 'DisplayName', 'x_l');
hold on;
plot(xnl, nolang_cumulative, 'DisplayName', 'x_{nl}');
xlabel('x')
ylabel('sum p(obs|x)')
title('Cumulative Posterior Distributions')
legend()

%%
% Next we are to compute the upper and lower bounds for 95% confidence
% intervals. The upper bound will be set by cdf having a
% value of 0.975 and the lower bound will be set by the cdf having value of
% 0.025. Therefore, we are interested in finding the first value in the
% posterior that will result in cdf having value greater than 0.975. This
% will be the upper bound. Similarly, we are interested in finding the last
% value in the posterior that will result in cdf having value lesser than
% 0.025. This will be the lower bound.
%%
lang_low_index = find(lang_cumulative < 0.025, 1, 'last' ); % Index of lower bound language
lang_up_index = find(lang_cumulative > 0.975, 1 ); % Index of upper bound language
nolang_low_index = find(nolang_cumulative < 0.025, 1, 'last' ); % Index of lower bound no language
nolang_up_index = find(nolang_cumulative > 0.975, 1 ); % Index of upper bound no language

lang_low_limit = lang_posterior(lang_low_index) % Lower limit for language
lang_up_limit = lang_posterior(lang_up_index) % Upper limit for language
nolang_low_limit = nolang_posterior(nolang_low_index) % Lower limit for no language
nolang_up_limit = nolang_posterior(nolang_up_index) % Upper limit for no language

%% d)
% The joint posterior distribution can be computed by taking an outer
% product of the posterior of each distribution.
jdf = lang_posterior' * nolang_posterior;

fig4 = figure();
imagesc(jdf)
xlabel('xnl')
ylabel('xl')
title('Joint posterior distribution')

%%
% The posterior probability of xl > xnl is given by the sum of all the
% values in the lower triangular part of jdf as this corresponds to xl >
% xnl. The posterior probability of xl <= xnl is given as 1 - the
% probability of xl > xnl.
pxlgxnl = sum(tril(jdf), 'all') % xl > xnl
pxnlgxl = 1 - pxlgxnl % xl <= xnl

%% e)
% We can compute P(language | activation) using the Bayes' rule:
%%
% $$P(language | activation) = \frac{P(activation | language)\times P(language)}{P(activation | language)\times P(language) + P(activation | no language)\times P(no language)} $$
%%
% The priors are:
%%
% $$P(language) = P(no language) = 0.5 $$
%%
% And the likelihoods are computed in b. Therefore, in terms of variables,
% we have:
%%
% $$P(activation | language) = lang_maxlikelihood_estim $$
%%
% $$P(activation | no language) = nolang_maxlikelihood_estim $$
%%

p_lang = 0.5; % prior probability of language
p_nolang = 0.5; % prior probability of no language
p_lang_activ = (lang_maxlikelihood_estim * p_lang) / ...
    (lang_maxlikelihood_estim * p_lang + nolang_maxlikelihood_estim * p_nolang) % P(language | activation)
p_nolang_activ = (nolang_maxlikelihood_estim * p_nolang) / ...
    (lang_maxlikelihood_estim * p_lang + nolang_maxlikelihood_estim * p_nolang) % P(language | activation)
%%
% If Poldrack's critique were true, then the Bayes factor would not equate
% the prior odds. If, however, Poldrack's critique were incorrect, then the
% Bayes factor would be the same as the prior odds. Computing the Bayes'
% factor as the ratio of p_lang_activ and p_nolang_activ and comparing it
% to the prior odds:
%%
bayes_factor = p_lang_activ/p_nolang_activ % Bayes' factor
prior_odds = p_lang/p_nolang % Prior odds

%%
% As can be seen, the prior odds are 1. However, the Bayes' factor is not
% 1. Therefore, we can see that Poldrack's critique is correct and can say
% that activation in a given area does not necessarily indicate that a
% cognitive process is engaged in language without computing the posterior
% probability.