clear; close all; clc;

%% a)
A = randn(2, 4);
plotVec2(A)

%%
v1 = [0 1]';
v2 = [1 0]';
[v1_length, v2_length, delta_theta] = vecLenAngle(v1, v2)

%%
M = rand(2, 2);
[U, S, V] = svd(M)
%%
function plotVec2(A)
    if size(A, 1) == 2
        plotv(A, '--o');
        axis([-1, 1, -1, 1]);
        %ylim([-1, 1]);
        %axis('equal')
    else
        disp('Matrix height is greater than expected')
    end       
    
end

function [v1_length, v2_length, delta_theta] = vecLenAngle(v1, v2)
    v1_length = sqrt(sum(v1.^2));
    v2_length = sqrt(sum(v2.^2));
    if v1_length == 0 || v2_length == 0
        delta_theta = 'Not defined';
    else
        delta_theta = acos((v1'*v2)/(v1_length*v2_length));
    end
    
end