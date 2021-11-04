clear; close all; clc;

%%
p = rand(10, 1);
p = p/sum(p);
F = [sum(p(1:k)) for k = 1:10];
sum(p)


function samples = randp(p, num)
    samples = []
    for i = 1:num
        samples = [samples, 
    
end