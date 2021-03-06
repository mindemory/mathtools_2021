clear; clc; close all;

%% a)
load('constrainedLS.mat')

%%
% The original optimization problem can be stated as:
%%
% $$min_{\vec{\beta}} \sum_n(\vec{\beta}^T \vec{d_n})^2 $$, s.t. $$\vec{\beta}^T\vec{w} = 1 $$
%%
% Let D be the data matrix which has the data points as its rows and
% beta is the vector of parameters that we are interested in. Hence
% $$D\vec{\beta} $$ becomes a vector with nth row being $$\vec{\beta}^T
% \vec{d_n} $$ or equivalently $$\vec{d_n}^T \vec{\beta} $$
%%
% Using the definition of the norm, we have:
%%
% $$||D\vec{\beta}||^2 = \sum_n (\vec{\beta}^T \vec{d_n})^2$$
%%
% Thus, the optimization problem can be re-written in the matrix form as:
%%
% $$min_{\vec{\beta}} ||D\vec{\beta}||^2$$, s.t. $$\vec{\beta}^T\vec{w} =
% 1$$
%%
% Now we can simplify the problem by performing SVD over the data matrix D.
% Doing so, we get:
%%
% $$D = USV^T$$
%%
% Substituting the SVD back into the matrix form of the optimization
% problem:
%%
% $$min_{\vec{\beta}} ||USV^T\vec{\beta}||^2$$, s.t. $$\vec{\beta}^T\vec{w}
% = 1$$
%%
% Since U is a transformation matrix, it is not relevant to the minimization problem:
%% 
% $$min_{\vec{\beta}} ||SV^T\vec{\beta}||^2$$, s.t. $$\vec{\beta}^T\vec{w}
% = 1$$
%%
% Let $$\vec{\beta^*} = V^T\vec{\beta} $$
%%
% Therefore,
%%
% $$\vec{\beta} = V\vec{\beta^*}$$
%%
% The minimization problem then becomes:
%%
% $$min_{\vec{\beta^*}} ||S\vec{\beta^*}||^2$$, s.t. $$(V\vec{\beta^*})^T
% \vec{w} = 1$$
%% 
% $$min_{\vec{\beta^*}} ||S\vec{\beta^*}||^2$$, s.t. $$\vec{\beta^*}^T V^T
% \vec{w} = 1$$
%%
% Let $$\vec{w^*} = V^T \vec{w} $$
%%
% Therefore,
%%
% $$min_{\vec{\beta^*}} ||S\vec{\beta^*}||^2$$, s.t. $$\vec{\beta^*}^T
% \vec{w^*} = 1$$
%%
% Now S is a diagonal matrix of shape 300 * 2. Hence it has the first two
% rows with non-zero diagonal elements. All the remaining rows of S are zeros.
% Therefore, the transformation of $$\vec{\beta^*} $$ by S creates a 300
% dimensional vector with only two non-zero entries.
% Hence, the minimization problem can be simplified by considering only the
% first two rows of S. Let this matrix be called S*. The optimization
% problem can hence be restated as:
%%
% $$min_{\vec{\beta^*}} ||S^*\vec{\beta^*}||^2$$, s.t.
% $$\vec{\beta^*}^{T}\vec{w^*} = 1$$
%%
% Let $$\tilde{\beta} = S^{*} \vec{\beta^*} $$
%%
% Here, $$S^{*} $$ is an orthogonal matrix with non-zero entries along the 
% diagonal and zeros off-diagonal. Therefore,
%%
% $$\vec{\beta^*} = S^{*^{-1}}\tilde{\beta}$$
%%
% Here $$S^{*^{-1}} $$ is the inverse of $$S^* $$, and since $$S^* $$ is an
% orthogonal matrix, we have:
%% 
% $$S^{*^{-1}} = S^{*^T} $$
%%
% Therefore,
%%
% $$\vec{\beta^*}^T = (S^{*^{-1}}\tilde{\beta})^T =\tilde{\beta}^T S^{*}$$
%%
% The optimization problem now becomes:
%%
% $$min_{\tilde{\beta}} ||\tilde{\beta}||^2$$, s.t. $$\tilde{\beta}^{T}S^{*^{-1}}\vec{w^*} = 1$$
%%
% Let $$\tilde{w} = S^{*^{-1}} \vec{w^*} $$
%%
% Therefore,
%%
% $$min_{\tilde{\beta}} ||\tilde{\beta}||^2$$, s.t.
% $$\tilde{\beta}^{T}\tilde{w} = 1$$
%%
% We can re-write $$\tilde{\beta} $$ and $$\tilde{w} $$ in terms of $$\vec{\beta} $$
% and $$\vec{w} $$
%%
% $$\tilde{\beta} = S^* V^T \vec{\beta}$$
%%
% $$\tilde{w} = S^{*^{-1}} V^T \vec{w}$$

%% b)
% Performing SVD on the original data:
%%
[U, S, V] = svd(data);
Ss = S(1:2, :);
%%
% The shortest vector $$\tilde{\beta}_{opt} $$ is the one that lies along the
% direction of $$\tilde{w} $$ . Hence the angle between $$\tilde{\beta}_{opt} $$
% and $$\tilde{w} $$ is $$0^{\circ} $$
%%
% The constraint is basically a dot product of the vectors and hence can
% also be written as:
%%
% $$\tilde{\beta}^{T} \tilde{w} = ||\tilde{\beta}||.||\tilde{w}||cos \theta
% = 1 $$
%%
% Since $$\theta = 0^{\circ} $$, we have $$cos \theta = 1 $$
%%
% The constraint then becomes:
%%
% $$||\tilde{\beta}_{opt}||.||\tilde{w}|| = 1 $$
%%
% Therefore,
%%
% $$||\tilde{\beta}_{opt}|| = \frac{1}{||\tilde{w}||}$$
%%
% This gives the length of $$\tilde{\beta}_{opt} $$. This vector points in
% the same direction as $$\tilde{w} $$ and hence the vector can be
% represented as:
%%
% $$\tilde{\beta}_{opt} =||\tilde{\beta}_{opt}||
% \frac{\tilde{w}}{||\tilde{w}||}$$
%%
% Substituting the length of $$\tilde{\beta}_{opt} $$ calculated above:
%%
% $$\tilde{\beta}_{opt} = \frac{\tilde{w}}{||\tilde{w}||^2}$$
%%
% Computing $$\tilde{\beta}_{opt} $$:
w_tilde = inv(Ss)*V'*w;
w_tilde_hat = w_tilde/sqrt(sum(w_tilde.^2));
beta_tilde_opt = w_tilde_hat * 1/sqrt(sum(w_tilde.^2));

%%
% Creating a matrix to plot $$\tilde{\beta}_{opt} $$
beta_tilde_opt_fplot = [zeros(2, 1), beta_tilde_opt];

slope_beta_tilde_opt = beta_tilde_opt(2)/beta_tilde_opt(1);

slope_perp = -1/slope_beta_tilde_opt;
intercept_perp = beta_tilde_opt(2) - slope_perp * beta_tilde_opt(1);

data_transformed = U(:, 1:2);
% data_transformed = data * V * inv(Ss);

figure(1);
plot(beta_tilde_opt_fplot(1, :), beta_tilde_opt_fplot(2, :), 'r-o', 'MarkerSize', 5, ...
    'DisplayName', '$\tilde{\beta}_{opt}$')
hold on;
x_perp_p = xlim;
y_perp_p = slope_perp * x_perp_p + intercept_perp * [1, 1];
plot(x_perp_p, y_perp_p, 'b', 'LineWidth', 2, ...
    'DisplayName', 'constraint line')
plot(data_transformed(:, 1), data_transformed(:, 2), 'o')

title('Space of beta tilde')
hl = legend('show');
set(hl, 'Interpreter', 'latex')


%% c)
% From the previous transformations we have:
%%
% $$\tilde{\beta} = S^* V^T \vec{\beta}$$
%%
% Therefore,
%%
% $$\vec{\beta} = (S^* V^T)^{-1} \tilde{\beta}$$
%%
% Therefore,
%%
% $$\vec{\beta} = (V^T)^{-1} (S^*)^{-1} \tilde{\beta}$$
%%
% Therefore,
%%
% $$\vec{\beta} = V S^{*^{-1}} \tilde{\beta}$$
%%
% Computing $$\vec{\beta} $$ from $$\tilde{\beta} $$ we just computed, we
% get:
%%
beta_opt = V * inv(Ss)*beta_tilde_opt;
beta_opt_fplot = [zeros(2, 1), beta_opt];
w_fplot = [zeros(2, 1), w];
%%
% The original contraint line is perpendicular to the projection of
% $$\vec{\beta}_{opt} $$ on $$\vec{w} $$
%%
% The projection can be computed as:
%%
% $$\beta_{opt, proj} = \frac{\vec{\beta_{opt}} . \vec{w}}{||\vec{w}||} \frac{\vec{w}}{||\vec{w}||} $$
%%
% The slope of the constraint line is then -1 divided by the slope of the
% projection of beta_opt on w. The intercept of the constraint line can
% then be computed by passing the data point beta_opt through the linear
% regression model with slope just computed. This can then be used to plot
% the constraint line within the x limits of the plot.
%%
% For the total least squares problem, the optimization problem becomes:
%%
% $$min_{\hat{u}} ||D\hat{u}||^2$$, s.t. ||\hat{u}||^2 = 1
%% 
% The minimization goal can be simplified by performing SVD:
%%
% Therefore,
%%
% $$||D\hat{u}||^2 = (D\hat{u})^T(D\hat{u}) = \hat{u}^TD^TD\hat{u} $$
%%
% Therefore,
%%
% $$||D\hat{u}||^2 = \hat{u}^T(USV^T)^T(USV^T)\hat{u}$$
%%
% Therefore,
%%
% $$||D\hat{u}||^2 = \hat{u}^TVS^TU^TUSV^T\hat{u} = \hat{u}^TVS^TSV^T\hat{u}$$
%%
% V is an orthogonal matrix involved in the transformation of the vector
% $$\hat{u} $$ and hence we can define:
%%
% $$\hat{v} = V\hat{u} $$
%%
% And the optimization problem becomes:
%%
% $$min_{\hat{v}} \hat{v}^TS^TS\hat{v} $$, s.t. $$||\hat{v}|| = 1 $$
%%
% $$S^T S $$ is a square matrix with zero off-diagonal elements and the
% diagonal elements being $$s_1^2 $$ and $$s_2^2 $$
%%
% Thus, the goal of minimization can be re-written as:
%%
% $$min_{\hat{v}} \hat{v}^TS^TS\hat{v} $$ = $$min_{\hat{v}} \sum_n s_n^2
% v_n^2 $$
%%
% $$ = \sum_n s_2^2 v_n^2 $$, since $$s_2 $$ is the smallest eigen value
%%
% $$ = s_2^2 \sum_n v_n^2 = s_2^2 ||\hat{v}||^2 = s_2^2 $$, since
% $$||\hat{v}||^2 = 1 $$
%%
% Therefore, in the space of $$\hat{u} $$ the goal is to pick the column in
% V that belongs to the smallest eigen value, in this case the second
% column of V.
%%
scalar_proj = beta_opt' * w / norm(w, 2);
beta_opt_proj = scalar_proj * w / norm(w, 2);
beta_opt_proj_fplot = [zeros(2, 1), beta_opt_proj];

slope_beta_proj = beta_opt_proj(2)/beta_opt_proj(1);
slope_perp_org = -1/slope_beta_proj;
intercept_perp_org = beta_opt_proj(2) - slope_perp_org * beta_opt_proj(1);
u_opt = V(:, 2);
u_opt_fplot = [zeros(2, 1), u_opt];
figure(2);
plot(data(:, 1), data(:, 2), 'o')
hold on;
plot(beta_opt_fplot(1, :), beta_opt_fplot(2, :), 'r-', ...
    'DisplayName', '$\vec{\beta}_{opt}$')
plot(u_opt_fplot(1, :), u_opt_fplot(2, :), 'm-', 'LineWidth', 2, ...
    'DisplayName', '$\hat{u}$')

plot(w_fplot(1, :), w_fplot(2, :), 'k-', 'LineWidth', 2, ...
    'DisplayName', '$\vec{w}$')

x_perp_org = xlim;
y_perp_org = slope_perp_org * x_perp_org + intercept_perp_org * [1, 1];
plot(x_perp_org, y_perp_org, 'b', 'LineWidth', 2, ...
    'DisplayName', 'constraint line')
title('Original space')
hl = legend('show');
set(hl, 'Interpreter', 'latex')
