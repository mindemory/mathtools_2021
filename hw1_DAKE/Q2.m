%% Homework 0 - Question 2 - Mrugank Dake
clear; clc; close all;

%% 2a)
% Vectors a and b are unit-length vectors and we can center them at 0. The
% If the two vectors are aligned with the tip of vector b sitting on the
% top of vector a, then the vector c which represents the hypotenuse of the
% resulting triangle is given as the sum of the vectors a and b. The length
% of the hypotenuse is then the length of the vector c, which is defined as
% the positive square-root of the sum of the two co-ordinates of the
% vector.
a = [0 1];
b = [1 0];
c = a + b;
hypotenuse_length = abs(sqrt(c(1)^2 + c(2)^2));
disp(['The length of the hypotenuse i.e. vector c is: ', num2str(hypotenuse_length)]);
% As can be seen, the length of the resulting hypotenuse vector agrees with
% what we computed in question 1

%% 2b)
%

% The length of the hypotenuse using the norm:
norm_length = norm(c, 2);
disp(['The length of the vector c using norm is: ', num2str(norm_length)]);
