clear; clc; close all;

%%
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

figure();
imagesc(X)

for i = 1:10
    h = rand(15, 1);
    r_X = X * h;
    r_mat = conv(x, h);
    %size(r_X)
    %size(r_mat)
    if isequal(r_X, r_mat)
        disp([num2str(i), ': Response through X and conv function are the same']);
    else
        disp('Response through X and conv function are different');
    end
end
    

function X = createConvMat(x, M)
    dim_X = length(x) + M - 1;
    x(dim_X) = 0;
    X = zeros(dim_X, M);
    for i = 1:M
        X(:, i) = circshift(x, i - 1);
    end
    
end

