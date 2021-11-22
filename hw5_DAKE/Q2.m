clear; close all; clc;

%% a)
lang_total = 869;
lang_activ = 103;
nolang_total = 2353;
nolang_activ = 199;

xl = 0:0.001:1;
xnl = 0:0.001:1;

lang_likelihood = binopdf(lang_activ, lang_total, xl);
nolang_likelihood = binopdf(nolang_activ, nolang_total, xnl);
fig1 = figure();
bar(xl, lang_likelihood, 'DisplayName', 'x_l');
hold on;
bar(xnl, nolang_likelihood, 'DisplayName', 'x_{nl}');
xlabel('x')
ylabel('p(x|obs)')
legend()

%% b)
lang_max_index = find(lang_likelihood == max(lang_likelihood));
nolang_max_index = find(nolang_likelihood == max(nolang_likelihood));
lang_maxlikelihood_pred = xl(lang_max_index);
nolang_maxlikelihood_pred = xnl(nolang_max_index);


lang_maxlikelihood_estim = lang_activ/lang_total;
nolang_maxlikelihood_estim = nolang_activ/nolang_total;

%% c)
lang_denom = lang_likelihood * xl';
nolang_denom = nolang_likelihood * xnl';
lang_posterior = (lang_likelihood .* xl) / lang_denom;
nolang_posterior = (nolang_likelihood .* xl) / nolang_denom;
lang_cumulative = zeros(1, length(xl));
nolang_cumulative = zeros(1, length(xl));
for i = 1:length(xl)
    if i == 1
        lang_cumulative(i) = lang_posterior(i);
        nolang_cumulative(i) = nolang_posterior(i);
    else
        lang_cumulative(i) = lang_cumulative(i-1) + lang_posterior(i);
        nolang_cumulative(i) = nolang_cumulative(i-1) + nolang_posterior(i);
    end
end
lang_low_index = find(lang_cumulative < 0.05, 1, 'last' );
lang_up_index = find(lang_cumulative > 0.95, 1 );
nolang_low_index = find(nolang_cumulative < 0.05, 1, 'last' );
nolang_up_index = find(nolang_cumulative > 0.95, 1 );
lang_low_limit = lang_posterior(lang_low_index);
lang_up_limit = lang_posterior(lang_up_index);
nolang_low_limit = nolang_posterior(nolang_low_index);
nolang_up_limit = nolang_posterior(nolang_up_index);
