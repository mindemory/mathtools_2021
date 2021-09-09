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
% (1)
met1_length = abs(sqrt(c(1)^2 + c(2)^2));
disp(['The length of the vector c using method 1 is: ', num2str(met1_length)]);

% (2)
met2_length = abs(sqrt(sum(c.^2)));
disp(['The length of the vector c using method 2 is: ', num2str(met2_length)]);

% (3)
met3_length = abs(sqrt(dot(c, c)));
disp(['The length of the vector c using method 3 is: ', num2str(met3_length)]);

% (4)
met4_length = norm(c, 2);
disp(['The length of the vector c using method 4 is: ', num2str(met4_length)]);

% (4)
met5_length = abs(sqrt(c * c'));
disp(['The length of the vector c using method 5 is: ', num2str(met5_length)]);


%% 2c)
% The angle between the two vectors can be computed as the arccos of the
% ratio of the dot product of two vectors and the product of norms of the
% two vectors
theta = acos(dot(c, a) / (norm(c, 2) * norm(a, 2)));
disp(['The angle between the vectors c and a is: ', num2str(theta)])


%% 2d)
figure;
plot(a);
hold on;
plot(b);
plot(c);