clear; close all; clc;
%% Mrugank Dake

%% 
% The input vectors here are the intensities and the output vectors are the
% responses to the incident light produced by the retina. The weight vector
% here is the system that converts the input to the output.
w = [1, 4, 3.5, 2, 1];

%% a)
% The weight matrix or rather the weight vector, in this case, operates on
% the input vector to compute the weighted average of the input vector.
% This is a linear operation and hence the system is linear.
%%
% The response (r) can be expressed in terms of the weight matrix (m) and
% the input vector (v) as:
%%
% r = w' * v
%% b)
% Let the input vector that elicits the maximum response be u
%%
% The response for this input vector is given by:
%%
% r = w' * u
%% 
% However, this is basically a dot product of the vectors w and u and can
% be re-written as:
%%
% r = w' * u = ||||w|| * ||||u|| * cos(theta), where theta is the angle between
% the two vectors
%% 
% The response will be maximum for the given w and u, when cos(theta) is
% maximum i.e. when cos(theta) = 1 i.e. when theta = 0
%%
% Hence for the maximum response, w and u have to be collinear. Since u has
% to be a unit vector, we can compute u as the ratio of the vector w with
% its norm
%%
% Thus the unit-length stimulus vector that produces the largest response
% is:
u = w/sqrt(sum(w.^2))

%% c)
% Following the logic from (b), the smallest response will be produced by
% the stimulus vector with cos(theta) = 0 i.e. when theta = 90. However,
% since all the elements of the vector w are positive, any vector that is
% orthogonal to w will have at least one of the quantities < 0. However,
% for the stimulus to be physically realizable, all the elements of the
% input stimulus vector have to be greater than 0, since the intensities
% are supposed to be positive. Therefore, the goal here is to try to find
% out a vector that is as close to being orthogonal to the weight vector as
% possible but the components are still greater than 0.
%%
% The vector should be as close to being orthogonal and hence the best
% candidate would be one of the basis vectors. Amongst the basis vectors,
% the vector that will be closest to being orthonormal to the weight vector
% is the one that lies along the component of weight vector i.e. smallest.
%%
% Hence the stimulus vectors that will produce the smallest responses in
% these neurons are: [1 0 0 0 0] and [0 0 0 0 1]