clear; clc; close all;

%% a)
load('constrainedLS.mat')

%%
% The original optimization problem can be stated as:
%%
% $min_{\vec{\beta}} \sum_n(\vec{\beta}^T \vec{d_n})^2$, s.t. $\vec{\beta}^T\vec{w} = 1$
%%
% Let D be the data matrix which has the data points as its rows and
% beta is the vector of parameters that we are interested in. Hence
% $$D\vec{\beta} $$ becomes a vector with nth row being $$\vec{\beta}^T
% \vec{d_n} $$ or equivalently $$\vec{d_n}^T \vec{\beta} $$
%%
% Using the definition of the norm, we have:
%%
% $||D\vec{\beta}||^2 = \sum_n (\vec{\beta}^T \vec{d_n})^2$
%%
% Thus, the optimization problem can be re-written in the matrix form as:
%%
% $min_{\vec{\beta}} ||D\vec{\beta}||^2$, s.t. $\vec{\beta}^T\vec{w} = 1$
%%
% Now we can simplify the problem by performing SVD over the data matrix D.
% Doing so, we get:
%%
% $D = USV^T$
%%
% Substituting the SVD back into the matrix form of the optimization
% problem:
%%
% $min_{\vec{\beta}} ||USV^T\vec{\beta}||^2$, s.t. $\vec{\beta}^T\vec{w} = 1$
%%
% Since U is a transformation matrix, it is not relevant to the minimization problem:
%% 
% $min_{\vec{\beta}} ||SV^T\vec{\beta}||^2$, s.t. $\vec{\beta}^T\vec{w} = 1$
%%
% Let $$\vec{\beta^*} = V^T\vec{\beta} $$
%%
% Therefore,
%%
% $\vec{\beta} = V\vec{\beta^*}$
%%
% The minimization problem then becomes:
%%
% $min_{\vec{\beta^*}} ||S\vec{\beta^*}||^2$, s.t. $(V\vec{\beta^*})^T \vec{w} = 1$
%% 
% $min_{\vec{\beta^*}} ||S\vec{\beta^*}||^2$, s.t. $\vec{\beta^*}^T V^T \vec{w} = 1$
%%
% Let $$\vec{w^*} = V^T \vec{w} $$
%%
% Therefore,
%%
% $min_{\vec{\beta^*}} ||S\vec{\beta^*}||^2$, s.t. $\vec{\beta^*}^T \vec{w^*} = 1$
%%
% Now S is a diagonal matrix of shape 300 * 2. Hence it has the first two
% rows with non-zero diagonal elements. All the remaining rows of S are zeros.
% Therefore, the transformation of $$\vec{\beta^*} $$ by S creates a 300
% dimensional vector with only two non-zero entries.
% Hence, the minimization problem can be simplified by considering only the
% first two rows of S. Let this matrix be called S*. The optimization
% problem can hence be restated as:
%%
% $min_{\vec{\beta^*}} ||S^*\vec{\beta^*}||^2$, s.t. $\vec{\beta^*}^{T}\vec{w^*} = 1$
%%
% Let $$\tilde{\beta} = S^{*} \vec{\beta^*} $$
%%
% Therefore,
%%
% $\vec{\beta^*} = S^{*\#}\tilde{\beta}$
%%
% Here $$S^{*\#} $$ is the pseudoinverse of $$S^* $$ and is an orthogonal
% matrix, since $$S^* $$ is a square matrix with non-zero diagonal elements
% and the off-diagonal elements being zero.
%%
% Thus, we have:
%%
% $\vec{\beta^*}^T = (S^{*\#}\tilde{\beta})^T =\tilde{\beta}^T S^{*\#}$
%%
% The optimization problem now becomes:
%%
% $$min_{\tilde{\beta}} ||\tilde{\beta}||^2$$, s.t. $$\tilde{\beta}^{T}S^{*\#}\vec{w^*} = 1$$
%%
% Let $$\tilde{w} = S^{*\#} \vec{w^*} $$
%%
% Therefore,
%%
% $min_{\tilde{\beta}} ||\tilde{\beta}||^2$, s.t. $\tilde{\beta}^{T}\tilde{w} = 1$
%%
% We can re-write $$\tilde{\beta} $$ and $$\tilde{w} $$ in terms of $$\vec{\beta} $$
% and $$\vec{w} $$
%%
% $\tilde{\beta} = S^* V^T \vec{\beta}$
%%
% $\tilde{w} = S^{*\#} V^T \vec{w}$

%% b)
% Performing SVD on the original data:
%%
[U, S, V] = svd(data);
Ss = S(1:2, :);
%%
% The shortest vector $$\tilde{\beta}_{opt} $$ is the one that lies along the
% direction of $$\tilde{w} $$ . Hence the angle between $$\tilde{\beta}_{opt} $$
% and $$\tilde{w} $$ is either $$0^{\circ} $$ or $$180^{\circ} $$
%%
% The constraint is basically a dot product of the vectors and hence can
% also be written as:
%%
% $$\tilde{\beta}^{T} \tilde{w} = ||\tilde{\beta}||.||\tilde{w}||cos \theta
% = 1 $$
%%
% For $$\theta = 0^{\circ} $$, we have $$cos \theta = 1 $$, and 
% for $$\theta = 180^{\circ} $$, we have $$cos \theta = -1 $$
%%
% The constraint then becomes:
%%
% $$||\tilde{\beta}_{opt}||.||\tilde{w}|| = \pm 1 $$
%%
% Therefore,
%%
% $||\tilde{\beta}_{opt}|| = \pm \frac{1}{||\tilde{w}||}$
%%
% This gives the length of $$\tilde{\beta}_{opt} $$. This vector points in
% the same direction as $$\tilde{w} $$ and hence the vector can be
% represented as:
%%
% $\tilde{\beta}_{opt} = \pm ||\tilde{\beta}_{opt}|| \frac{\tilde{w}}{||\tilde{w}||}$
%%
% Substituting the length of $$\tilde{\beta}_{opt} $$ calculated above:
%%
% $\tilde{\beta}_{opt} = \pm \frac{\tilde{w}}{||\tilde{w}||^2}$
%%
% Computing $$\tilde{\beta}_{opt}$$:
w_tilde = pinv(Ss)*V'*w;
w_tilde_hat = w_tilde/sqrt(sum(w_tilde.^2));
beta_tilde_optp = w_tilde_hat * 1/sqrt(sum(w_tilde.^2));
beta_tilde_optn = - w_tilde_hat * 1/sqrt(sum(w_tilde.^2));

%%
% Creating a matrix to plot $$\tilde{\beta}_{opt}$$
beta_tilde_optp_fplot = [zeros(2, 1), beta_tilde_optp];
beta_tilde_optn_fplot = [zeros(2, 1), beta_tilde_optn];
w_tilde_fplot = [zeros(2, 1), w_tilde];

slope_beta_tilde_optp = beta_tilde_optp(2)/beta_tilde_optp(1);
slope_beta_tilde_optn = beta_tilde_optn(2)/beta_tilde_optn(1);

slope_perp_p = -1/slope_beta_tilde_optp;
slope_perp_n = -1/slope_beta_tilde_optn;
intercept_perp_p = beta_tilde_optp(2) - slope_perp_p * beta_tilde_optp(1);
intercept_perp_n = beta_tilde_optn(2) - slope_perp_n * beta_tilde_optn(1);

data_transformed = U' * data;

figure(1);
plot(beta_tilde_optp_fplot(1, :), beta_tilde_optp_fplot(2, :), 'r-o', 'MarkerSize', 5, ...
    'DisplayName', '$\tilde{\beta}_{opt}$')
hold on;
plot(w_tilde_fplot(1, :), w_tilde_fplot(2, :), 'k-o', 'MarkerSize', 5, ...
    'DisplayName', '$\tilde{w}$')
x_perp_p = xlim;
y_perp_p = slope_perp_p * x_perp_p + intercept_perp_p * [1, 1];
plot(x_perp_p, y_perp_p, 'b', 'LineWidth', 2, ...
    'DisplayName', 'perp line')
%plot(data_transformed(:, 1), data_transformed(:, 2), 'o')

title('Space of beta tilde')
hl = legend('show');
set(hl, 'Interpreter', 'latex')

figure(2);
plot(beta_tilde_optn_fplot(1, :), beta_tilde_optn_fplot(2, :), 'r-o', 'MarkerSize', 5, ...
    'DisplayName', '$\tilde{\beta}_{opt}$')
hold on;
plot(w_tilde_fplot(1, :), w_tilde_fplot(2, :), 'k-o', 'MarkerSize', 5, ...
    'DisplayName', '$\tilde{w}$')
x_perp_n = xlim;
y_perp_n = slope_perp_n * x_perp_n + intercept_perp_n * [1, 1];
plot(x_perp_n, y_perp_n, 'b', 'LineWidth', 2, ...
    'DisplayName', 'perp line')
%plot(data_transformed(:, 1), data_transformed(:, 2), 'o')

title('Space of beta tilde')
hl = legend('show');
set(hl, 'Interpreter', 'latex')


%% c)
% From the previous transformations we have:
%%
% $\tilde{\beta} = S^* V^T \vec{\beta}$
%%
% Therefore,
%%
% $\vec{\beta} = (S^* V^T)^{-1} \tilde{\beta}$
%%
% Therefore,
%%
% $\vec{\beta} = (V^T)^{-1} (S^*)^{-1} \tilde{\beta}$
%%
% Therefore,
%%
% $\vec{\beta} = V S^{*\#} \tilde{\beta}$
%%
% Computing $$\vec{\beta} $$ from $$\tilde{\beta} $$ we just computed, we
% get:
%%
beta_opt = V * pinv(Ss)*beta_tilde_optp;
beta_opt_fplot = [zeros(2, 1), beta_opt];
w_fplot = [zeros(2, 1), w];
%%
% The original contraint line is perpendicular to the projection of
% $$\vec{\beta}_{opt} $$ on $$\vec{w} $$
%%
% The projection can be computed as:
%%
% $$\beta_{opt, proj} = \vec{\beta_{opt}}cos \theta $$
%%
% where, $$\phi $$ is the angle between the two vectors, the cosine of which
% can be computed as:
%%
% $$cos \phi = \frac{\vec{\beta}_{opt}'
% \vec{w}}{||\vec{\beta}_{opt}||.||\vec{w}||} $$
scalar_proj = beta_opt' * w / norm(w, 2);
beta_opt_proj = scalar_proj * w / norm(w, 2);
beta_opt_proj_fplot = [zeros(2, 1), beta_opt_proj];


slope_beta_proj = beta_opt_proj(2)/beta_opt_proj(1);
slope_perp_org = -1/slope_beta_proj;
intercept_perp_org = beta_opt_proj(2) - slope_perp_org * beta_opt_proj(1);

figure(2);
plot(beta_opt_fplot(1, :), beta_opt_fplot(2, :), 'r-', 'MarkerSize', 5, ...
    'DisplayName', '$\vec{\beta}_{opt}$')
hold on;
plot(w_fplot(1, :), w_fplot(2, :), 'k-', 'MarkerSize', 5, ...
    'DisplayName', '$\vec{w}$')

plot(data(:, 1), data(:, 2), 'o')
x_perp_org = xlim;
y_perp_org = slope_perp_org * x_perp_org + intercept_perp_org * [1, 1];
plot(x_perp_org, y_perp_org, 'b', 'LineWidth', 2, ...
    'DisplayName', 'perp line')
title('Original space')
hl = legend('show');
set(hl, 'Interpreter', 'latex')
