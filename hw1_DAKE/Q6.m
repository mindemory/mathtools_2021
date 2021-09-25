clear; close all; clc;

%% 
% The system is:
%%
% y = Mv
%%
% where y is the output vector, v is the input vector and M is the linear
% system
%%
% Using SVD, we have:
%%
% M = USV'
%%
% The null space of the matrix M is the rows of V' that limit the
% dimensions of the input vector that matter for the output. It is the
% diagonal elemnts of the S matrix that are 0 and hence the information
% present in the corresponding dimensions of the input vector are lost when
% the input vector is passed through the SVT.

%%
% The range space of the matrix M is the columns of U that map the output
% space. They are literally the "range" of the matrix M. The output vector
% can be thought of as a linear combination of the range-space of M. These
% are the columns of U that matter for the output. The range-space is
% determined by the rows of the S matrix that have non-zero diagonal
% elements.

%%
% The input vector for the animal is a vector of pressure measurements and
% the output vector is a vector of neuronal responses. If the null-space of
% the system is non-zero, it implies that there is a limit on the
% dimensions of the input vector that the creature can perceive. The
% rows of the input vector that belong to the null-space cannot be
% perceived since those pressure measurements are not transformed by the
% system the creature is using to map the input to the output neuronal
% responses.

%%
mtxExamples = load('mtxExamples.mat');
mtx_fields = fieldnames(mtxExamples);
for i = 1:size(mtx_fields, 1)
    mtx = mtxExamples.(mtx_fields{i})
    [U, S, V] = svd(mtx)
    
    null_cols = find(all(S<10^-3, 1));
    if size(null_cols, 2) > 0
        fprintf('There is a nullspace of size %d\n', size(null_cols, 2))
        x =  V(:, null_cols) * randn(size(null_cols, 2), 1);
        x
        y = mtx * x
    else
        fprintf('No nullspace exists for this matrix\n')
    end

    range_rows = find(any(S>10^-3, 2));
    y_range = U(:, range_rows) * randn(size(range_rows, 1), 1)
    x_range = pinv(mtx) * y_range;
    y_range_recomputed = mtx * x_range
end