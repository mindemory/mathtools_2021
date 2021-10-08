clear; clc; close all;

%% a)
load('regress1.mat')

figure(1)
% Plotting the x and y variables as a scatterplot
plot(x, y, 'r*', 'MarkerSize', 6, 'LineWidth', 2)
xlabel('x')
ylabel('y')
title('Polynomial regression orders 0 to 5')
hold on;

% Initializing vectors: orders is a vector listing orders that need to be
% sampled; colors is a vector of colors to be used for the line plots for
% each order, X is initialized which will be the data matrix, errors is
% initialized with zeros of the size of the orders vector
orders = 0:5;
X = [];
Colors = ['k', 'b', 'g', 'm', 'c', 'y'];
Errors = zeros(size(orders, 2), 1);

% Creating a data matrix with columns x^i in ith loop. The beta optimal is
% computed using SVD. The beta optimal are then used to derive the
% predictions, the prediction errors, and the squared prediction errors.
% The squared-errors are then appended to the Errors vector. Lastly, a plot
% is created at each iteration that plots the fit of order i for the data.
for i = orders
    X = [X x.^i];
    [U, S, V] = svd(X);
    betaOpt = V * pinv(S) * U' * y;
    predictions = X * betaOpt;
    prediction_errors = predictions - y;
    prediction_squared_error = prediction_errors' * prediction_errors;
    Errors(i+1) = prediction_squared_error;
    plot(x, predictions, Colors(i+1), ...
        'DisplayName', ['poly reg order = ', num2str(i), ', sq error = ', num2str(prediction_squared_error)], ...
        'LineWidth', 1.5)
end
legend('location', 'north')

%% b)

figure(2);
plot(orders, Errors, 'k-o', 'LineWidth', 2, 'MarkerSize', 5)
xlabel('Order of Polynomial')
ylabel('Squared errors')
title('Squared errors as a function of degree of polynomial')
%%
% The squared errors decrease monotonically as the order of the polynomial
% increases. However, this results in overfitting of the data. The goal of
% the regression is to model the data as closely as possible with as
% minimal parameters as we can. From the graph of the models, we see that
% the models with degree 0, 1, and 2 are an underfit. Model 3 has a lower error 
% but seems to assume that the data flattens on
% the lower end of x, which might or might not be the case. Models 4 and 5
% are good fit to the data and exhibit similar squared-errors. Therefore,
% the model with the least order that has the best fit but also does not
% end up fitting the noise is a good model. The best model in this scenario
% is the polynomial of order 4.