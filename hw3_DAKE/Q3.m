clear; clc; close all;

%% a)
load('hrfDeconv.mat')
subplot(2, 1, 1);
stem(x, 'filled')
xlabel('Time (s)')
title('Input vector x(n)')
subplot(2, 1, 2);
plot(r)
xlabel('Time (s)')
title('Response vector r(n)')

N = length(x);
M = length(r) + 1 - N;
X = createConvMat(x, M);

% Visualizing the matrix X as an image
figure();
imagesc(X)

for i = 1:10
    % Creating random hemodynamic responses
    h = rand(15, 1);
    % Computing the response vectors for the haemodynamic response obtained
    % from the manually computed X matrix
    r_X = X * h;
    % Computing the response vectors for the haemodynamic response obtained
    % from the conv function
    r_mat = conv(x, h);
    % Checking for the equality of the computed response vectors
    if isequal(r_X, r_mat)
        disp([num2str(i), ': Response through X and conv function are the same']);
    else
        disp('Response through X and conv function are different');
    end
end

%% b)
[U, S, V] = svd(X);
Spinv = 1./S';
Spinv(isinf(Spinv)) = 0;
h_opt = V * Spinv * U * r;
plot(h_opt);

%% c)
figure()
F_h_opt = fftshift(h);
plot(F_h_opt);
%% Functions
function X = createConvMat(x, M)
    dim_X = length(x) + M - 1;
    x(dim_X) = 0;
    X = zeros(dim_X, M);
    for i = 1:M
        X(:, i) = circshift(x, i - 1);
    end
    
end

