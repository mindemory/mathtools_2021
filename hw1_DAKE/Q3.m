%% Homework 0 - Question 2 - Mrugank Dake
clear; clc; close all;

%% 3a)
% A column vector of 12 elements each of 3000 representing a steady income
% source of 3000 per month
income = 3000 * ones(12, 1);
disp('The column vector is:')
disp(income)

%% 3b)
% Due to the TAship, the income for the 7th month is now 4000. Therefore,
% the resulting income vector becomes:
income(7) = 4000;
disp('The update vector is:')
disp(income)

%% 3c)
% The expenses matrix has four columns: rent, grocery, clothing, fun in the
% order. The expenses for each month are kept constant and also the expense
% across each domain is also kept constant, say 500 each.
expenses = repmat(500, 12, 4);
disp('The expenses matrix is:')
disp(expenses)

%% 3d)
% The saving vector is the difference between the income vector and the
% row-sum vector of the expenses matrix
savings = income - sum(expenses, 2);
disp('The savings vector is:')
disp(savings)

%% 3e)
% The discount_vector is a vector with a linear spacing from 1 to 0.8 with
% 12 equal-spaced intervals
discount_vector = linspace(1, .8, 12);
% Transforming the grocery and clothing columns i.e. columns 2 and 3 of the expenses matrix
% using the discount matrix
expenses(:, 2:3) = expenses(:, 2:3) .* discount_vector';
disp('The resulting expenses matrix after discounting is:')
disp(expenses)

%% 3f)
savings(:, 2) = income - sum(expenses, 2);
h = bar(savings, 'grouped');
set(h, {'DisplayName'}, {'savings_{initial}', 'savings_{discounted}'}')
legend()