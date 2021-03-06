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
%% Null space
% The null space of the matrix M is the rows of V' that limit the
% dimensions of the input vector that matter for the output. It is the
% diagonal elements of the S matrix that are 0 and hence the information
% present in the corresponding dimensions of the input vector are lost when
% the input vector is passed through the SV'.

%% Range space
% The range space of the matrix M is the columns of U that span the output
% space. They are literally the "range" of the matrix M. The output vector
% can be thought of as a linear combination of the range space of M. These
% are the columns of U that matter for the output. The range space is
% determined by the rows of the S matrix that have non-zero diagonal
% elements.

%% Linear Tactile System
% The input vector for the animal is a vector of pressure measurements and
% the output vector is a vector of neuronal responses. If the null-space of
% the system is non-zero, it implies that there is a limit on the
% dimensions of the input vector that the creature can perceive. The
% rows of the input vector that belong to the null-space cannot be
% perceived since those pressure measurements are not transformed by the
% system the creature is using to map the input to the output neuronal
% responses.

%% mtxExamples
mtxExamples = load('mtxExamples.mat');
% Creating an array of the fieldnames stored in mtxExamples. This helps in
% indexing throught the mtxExamples
mtx_fields = fieldnames(mtxExamples);

for i = 1:size(mtx_fields, 1)
    % Loading the matrix stored at i_th position in mtx_fields
    mtx = mtxExamples.(mtx_fields{i});
    fprintf('mtx%d \n', i)
    % Running SVD on the mtx to extract U, S, and V
    [U, S, V] = svd(mtx);
    
    % The null space is determined by the columns of the S matrix that are
    % all 0s. The list of columns can be obtained by using find and all
    % functions
    null_cols = find(all(S<10^-3, 1));
    
    % The next step is to determine if the nullspace exists. It exists if
    % there are any elements in the null_cols array computed
    if size(null_cols, 2) > 0
        fprintf('There is a nullspace of size %d\n', size(null_cols, 2))
        % The input vector x that is spanned by the null-space rows of V'
        % or the null-space columns of V will all be mapped to 0 in the
        % output space. We can check for this as:
        x =  V(:, null_cols) * randn(size(null_cols, 2), 1)
        y = mtx * x
    else
        % If null_cols is empty, then the nullspace does not exist for the
        % given matrix
        fprintf('No nullspace exists for this matrix\n')
    end
    
    % The range of the matrix is determined by the rows of the matrix S
    % that have non-zero entries, which can be listed using find and any
    % functions
    range_rows = find(any(S>10^-3, 2));
    
    % The output vector y will belong to the range-space if it spanned by
    % the columns of U that belong to the range space
    y_range = U(:, range_rows) * randn(size(range_rows, 1), 1)
    % We can compute the corresponding x for the output vector y by using
    % the left pseudoinverse of the matrix. The resulting x can then be
    % mapped back to produce the y_range_recomputed
    x_range = pinv(mtx) * y_range;
    y_range_recomputed = mtx * x_range
end