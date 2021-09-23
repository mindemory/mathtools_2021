clear; close all; clc;

%% 
% The components of vector v along the direction of u_hat and along the 
% direction perpendicular to u_hat can be computed by running the
% inner_prod function (for more details, refer to the function).

%%
% Creating random vectors v1 and u1 using the randn:
v1 = randn(2, 1);
u1 = randn(2, 1);

%%
% Since u1_hat is supposed to be a unit vector, we compute u1_hat by taking
% diving u1 by the norm of itself
u1_hat = u1/norm(u1, 2);

%%
% Next we compute the v1_x, v1_y, and v1_perp_dist using the inner_prod
% function
[v1_x, v1_y, v1_perp_dist] = inner_prod(u1_hat, v1)