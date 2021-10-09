clear; clc; close all;

%% a)
load('constrainedLS.mat')

%%
% The original optimization problem can be stated as:
%%
% $$min_{\beta} \sum_n(\beta^T d_n)^2$$, s.t. $$\beta^Tw = 1$$
%%
% Let D be the data matrix which has the data points as its rows and the
% beta is the vector of the parameters that we are interested in. Hence
% D*beta becomes a vector with nth row being beta^T * dn
%%
% Using the definition of the norm, we can then see that:
%%
% $$||D\beta||^2 = \sum_n (\beta^T d_n)^2$$
%%
% Thus, the optimization problem can be re-written in the matrix form as:
%%
% $$min_{\beta} ||D\beta||^2$$, s.t. $$\beta^Tw = 1$$
%%
% Now we can simplify the problem by performing SVD over the data matrix D.
% Doing so, we get:
%%
% $$D = USV^T$$
%%
% Substituting the SVD back into the matrix form of the optimization
% problem:
%%
% $$min_{\beta} ||USV^T\beta||^2$$, s.t. $$\beta^Tw = 1$$
%%
% Since U is a transformation matrix, we can ignore it for the minimization
% problem:
%% 
% $$min_{\beta} ||SV^T\beta||^2$$, s.t. $$\beta^Tw = 1$$
%%
% Let $$\beta^* = V^T\beta $$, and $$w^* = V^Tw $$
%%
% The minimization problem then becomes:
%%
% $$min_{\beta^*} ||S\beta^*||^2$$, s.t. $$\beta^{*T}w^* = 1$$
%%
% Now S is a diagonal matrix of shape 300 * 2. Hence it has the first two
% rows that is a matrix of two elements along the diagonals s1, s2 with the
% other two elements being 0. Also all the remaining rows of S are zeros.
% Let $$ S^* $$ be the matrix obtained by taking the first two non-zero rows
% of S and $$ S^{\#} $$ be the inverse of $$ S^* $$, then we have:
%%
% $$min_{\tilde{\beta}} ||\tilde{\beta}||^2$$, s.t. $$\tilde{\beta}^{T}\tilde{w} = 1$$
%%
% where,
%%
% $$\tilde{\beta} = S^{\#}\beta^* $$, and $$\tilde{w} = S^* w^* $$
%%
% This can also be written in terms of the original beta and w:
%%
% $$\tilde{\beta} = S^{\#}V^T\beta $$, and $$\tilde{w} = S^* V^Tw $$

%% b)
% Performing SVD on the original data:
%%
[U, S, V] = svd(data);
w_tilde = S(1:2, :)*V'*w;
w_tilde_hat = w_tilde/sqrt(sum(w_tilde.^2));
beta_tilde = w_tilde_hat * 1/sqrt(sum(w_tilde.^2))