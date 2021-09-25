clear; close all; clc;
%% Mrugank Dake

%%
% Let S be a system that maps the input vectors to the output vectors. For
% S to be a linear system, we should have:
%%
% S(av1 + bv2) = aS(v1) + bS(v2);
%% 
% 
%% System 1:
% The input vectors are:
%%
% v1 = [3, 2]; v2 = [-1, 1]
%% 
% And the output vectors are:
%%
% w1 = 13; w2 = 3
%%
% Hence the system, if it exists, converts a 2 * 1 vector to a 1 * 1 vector
% (or a scalar). Therefore, the system has to be a matrix of size 1 * 2
%%
% Let the matrix be M = [a b]
%%
% Hence we have:
%%
% [a b]*[3 2]' = 13; and [a b]*[-1 1]' = 3
%%
% Therefore, 3a + 2b = 13; and -a + b = 3
%%
% From the equation on the right, we get:
%%
% b = 3 + a;
%% 
% Substituting in the equation in the left:
%%
% 3a + 2(3 + a) = 13
%%
% Therefore, 3a + 6 + 2a = 13
%%
% Therefore, 5a + 6 = 13
%%
% Therefore, 5a = 7
%%
% Therefore, a = 7/5 = 1.4
%%
% and b = 3 + a = 3 + 1.4 = 4.4
M = [1.4, 4.4];
u1 = [3, 2]'; u2 = [-1, 1]';
v1 = 13; v2 = 3;
%%
% Checking if the system works:
round(v1 - M * u1, 4) == 0
round(v2 - M * u2, 4) == 0

%% System 2:
% The system S here converts the input 0 to a vector [3, -3]. Since the
% transformation here is from a null vector of dimension 1 to an ouptut
% vector of shape (2, 1). No system exists that can multiply 0 to produce
% an output. Therefore, the system S does not exist.

%% System 3:
% The input vectors are:
%%
% v1 = [5, 2.5], v2 = [-1, -0.5]
%%
% The output vectors are:
% w1 = [-5, -10], w2 = [1, 2]
%%
% The input and output vectors are of size 2 * 1 and hence the matrix that 
% act as a representation of the system should be 2 * 2. Let this matrix
% be:
%%
% M = [a b; c d]
%%
% Therefore, we have four parameters that need to be figured out. However,
% a close examination of the input and output vectors shows that:
%%
% v1/-5 = v2 and w1/-5 = w2
%%
% Therefore, in this scenario, the two input/output pairs are linearly
% independent. Hence the problem boils down to solving for 4
% hyperparameters given only one equation with two variables. Therefore,
% there are infinitely many solutions that can solve this system of
% input/output pairs.
%%
% The system might be linear, but the matrix that can be used to solve this
% system of equations is not unique. Two example matrices that can solve
% the pair of equations are:
%%
% M = [-1 0; -2 0]
%%
% M = [-2 2; -3 2]

%% System 4:
% Let S be a system that converts the input vectors v1, v2, v3 to the
% output vectors w1, w2, w3, where:
%%
% v1 = [1, 3], v2 = [1, -1], v3 = [4, 0]
%%
% w1 = [3, 1], w2 = [-2, 2], w3 = [1, 6]
%% 
% In this particular case, v1 + 3 * v2 = v3
%%
% Using the definition of a linear system, if S is indeed linear:
%%
% S(v1 + 3 * v2) = S(v1) + 3 * S(v2)
%%
% Evaluating the LHS, S([1, 3]' + 3 * [1, -1]') = S([4, 0]') = [1, 6]'
%%
% Evaluating the RHS, S([1, 3]') + 3 * S([1, -1]') = [3, 1]' + 3 * [-2, 2]'
%%
% Therefore, RHS = [3, 1]' + [-6, 6]' = [-3, 7]'
%%
% Because LHS != RHS, the system S is not linear