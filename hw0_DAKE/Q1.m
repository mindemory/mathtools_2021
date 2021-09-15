%% Homework 0 - Question 1 - Mrugank Dake
clear; clc; close all;

%% 1a)
% The hypotenuse of a right-angle triangle is given as the square-root of
% the sum of the squares of the base and height. For a triangle with base
% of length b, and height of length h, the length of the hypotenuse c is
% given as:
% c = sqrt(b^2 + h^2)
% However, since the length is always positive, we can restrict the results
% by using the abs() function.
b = 1; h = 1;
c = abs(sqrt(b^2 + h^2));
disp(['The length of the hypotenuse is: ', num2str(c)])

%% 1b)
% The perimeter of the circle is defined by its circumference and the
% length of the circumference is given as the product of pi and the
% diameter of the circle. In the given scenario, diameter of the circle is
% the hypotenuse we computed.
d = c;

% Thus, the perimeter of the circle becomes:
perimeter_circle = pi * d;
disp(['The perimeter of the circle is: ', num2str(perimeter_circle)])

%% 1c)
% Adding a semi-colon suppresses printing of the variable perimeter_circle

%% 1d)
% The final script is ready for implementation.