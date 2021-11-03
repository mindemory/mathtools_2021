%% MATH TOOLS LAB 6: Summary statisitcs, probability distributions and sampling
clear all; close all; clc; %rng(1);
%% PART 1: L1, L2 and LP
% This part of the tutorial aims to walk you through L1, L2, and LP norm,
% and their cost functions (or you can call them objective functions). We
% will also prove that the value that minimizes L1 norm is the median, the
% value that minimizes L2 norm is the mean, and the value that minimizes LP 
% when P approaches infinity is the midpoint of the range.

% Let's first load the data
C            = load('Lab6_fake1d_data.mat');
% Each datapoint is 1d
D_x_1d       = C.D{1};
% But to make it easier to see, we will add some arbitrary jitter in the
% y-axis.
D_y_jitter   = randn(1, length(D_x_1d)).*0.03;

% Then let's compute the mean and the median
mean_D_x     = mean(D_x_1d); 
median_D_x   = median(D_x_1d); 
midpoint_D_x = (max(D_x_1d) - min(D_x_1d))/2 + min(D_x_1d);

% Use a scatterplot and a histogram to visualize the raw data.
figure
subplot(2,1,1)
scatter(D_x_1d, D_y_jitter, 20, 'ko','filled','MarkerFaceAlpha', 0.2,...
    'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 0.3); 
xlim([-3, 5]); ylim([-0.3, 0.3]); yticks(-0.1:0.1:0.1); box off
xlabel('Data'); ylabel('Jitter'); set(gca,'FontSize',15);

subplot(2,1,2); hold on
binsEdge = linspace(min(D_x_1d)-0.5, max(D_x_1d)+0.5, 80);
histogram(D_x_1d, binsEdge, 'FaceColor','r','FaceAlpha', 0.2); 
h1 = plot([median_D_x, median_D_x], [0, 35], 'b--', 'lineWidth', 3); 
h2 = plot([mean_D_x, mean_D_x], [0, 35], 'g--', 'lineWidth', 3); 
h3 = plot([midpoint_D_x, midpoint_D_x], [0,35], 'k--', 'lineWidth',3);
box off; legend([h1,h2,h3],{['Median = ', num2str(median_D_x)],...
    ['Mean = ', num2str(round(mean_D_x,2))], ['Midpoint = ', num2str(midpoint_D_x)]}); 
xlim([-3, 5]); ylim([0, 35]);
xlabel('Data'); ylabel('Counts'); set(gca,'FontSize',15);

%% Exercise 1
%Let's first formulate the cost functions for L1, L2, and LP:

%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%L1: absolute value of the difference between c and x (datapoint)
objFunc_L1 = ;

%L2 norm: squared difference between c and x
objFunc_L2 = ;

%LP norm: the absolute difference between c and x, to the power of a
objFunc_LP = ; 
%--------------------------------------------------------------------------

%Let's then define a range of candidate c, compute L1, L2 and LP norm
%for each candidate c, and find which c is the optimal value that minimizes
%the cost functions.
num_c_hyp  = 1e3;
c_hyp      = linspace(-0.5, 1.5, num_c_hyp);
P          = 100; %for LP, assume the power is 100.

%Use the anonymous functions you formulated above to compute the loss.
%Bonus: you can compute the loss by using for loops, but you can also do it
%in one line. Hint: use arrayfun.m

%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%Compute loss
loss_L1 = ;
loss_L2 = ;
loss_LP = ;

%Find the minimum value and the corresponding index with respect to 
%loss_L1, loss_L2, loss_LP
[minVal_L1, minInd_L1] = min(loss_L1);
[minVal_L2, minInd_L2] = min(loss_L2);
[minVal_LP, minInd_LP] = min(loss_LP);

%Given the indices minInd, we can find out what c_opt is for each cost
%function.
median_D_x_L1   = ;
mean_D_x_L2     = ;
midpoint_D_x_LP = ;
%--------------------------------------------------------------------------

%Now we compute the mean, median, and the midpoint of range via two ways:
%(1) directly calculating them given the datasets
%(2) testing a bunch of candidate c and find the one that minimizes the loss

%Now let's test if the mean, median and the midpoint calculated in one way 
%are similar to those calculated in the other way
if abs(median_D_x_L1 - median_D_x) < 1e-2; disp('Same medians: Checked!'); end
if abs(mean_D_x_L2 - mean_D_x) < 1e-2; disp('Same means: Checked!'); end
if abs(midpoint_D_x_LP - midpoint_D_x) < 5e-2; disp('Same midpoints: Checked!'); end

%% Exercise 1: continued 
%Visualizing the cost function, the value c's that minimize L1, L2 and LP
%norm. The code of this section do not require changes.

figure
subplot(1,2,1); hold on;
h1 = plot(c_hyp, loss_L1,'b','lineWidth',3); 
scatter(c_hyp(minInd_L1),minVal_L1,150,'b*'); 
h2 = plot(c_hyp, loss_L2,'g','lineWidth',3); 
scatter(c_hyp(minInd_L2),minVal_L2,150,'g*'); 
h3 = plot(c_hyp, loss_LP,'k','lineWidth',3); 
scatter(c_hyp(minInd_LP),minVal_LP,150,'k*'); hold off
legend([h1,h2,h3],{'L1 norm','L2 norm', 'LP norm'});
xlabel('Candidate c'); ylabel('Loss'); xlim([c_hyp(1),c_hyp(end)]);
set(gca,'FontSize',15);

subplot(1,2,2); hold on;
histogram(D_x_1d, binsEdge, 'FaceColor','r', 'FaceAlpha', 0.2);
f1 = plot([c_hyp(minInd_L1), c_hyp(minInd_L1)], [0, 35], 'b-', 'lineWidth', 3); 
f2 = plot([c_hyp(minInd_L2), c_hyp(minInd_L2)], [0, 35], 'g-', 'lineWidth', 3); 
f3 = plot([c_hyp(minInd_LP), c_hyp(minInd_LP)], [0, 35], 'k-', 'lineWidth', 3); 
box off; legend([f1,f2,f3],{['c that minimizes L1 = ', ...
    num2str(round(median_D_x_L1,2))],...
    ['c that minimizes L2 = ', num2str(round(mean_D_x_L2,2))], ...
    ['c that minimizes LP = ', num2str(round(c_hyp(minInd_LP),2))]}); 
xlim([min(D_x_1d)-0.5, max(D_x_1d)+0.5]); ylim([0, 35]);
xlabel('Data'); ylabel('Counts'); set(gca,'FontSize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.7, 0.6]);

%% Part 2: Expected value, moments and variance
%This part of the tutorial aims to walk you through how to compute the 
%expected value and the 2nd moments as well as variance given different 
%probability distributions. 

load('Lab6_fake1d_data_dists.mat', 'fakeD');
%from the .mat file, we can find fake data generated using a uniform
%distribution assuming minimum noise
x_uniform        = fakeD.uniform;
%how many unique x's 
x_unique         = unique(x_uniform); 
binSize          = diff(x_unique(1:2)); %0.01
%when we plot histograms, we want to specific bin edges
x_edges          = [x_unique, x_unique(end)+binSize]-binSize/2;
%the total number of data
numX_uni         = length(x_uniform);

%in the same file, we can also find fake data generated using a gaussian
%distribution
x_Gauss          = fakeD.Gauss;
%the total number of data
numX_Gauss       = length(x_Gauss);

figure
subplot(1,2,1)
histogram(x_uniform, x_edges, 'FaceAlpha', 0.5, 'EdgeColor','none'); hold on;
ylim([0, 12]); yticks([]); xlabel('x'); title('Uniform distribution');
set(gca,'Fontsize',15);

subplot(1,2,2)
histogram(x_Gauss, x_edges, 'FaceAlpha', 0.5, 'EdgeColor','none'); hold on
ylim([0, 42]); yticks([]); xlabel('x'); title('Gaussian distribution');
set(gca,'Fontsize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.5, 0.4]);

%% Exercise 2
%Expectation of value given a distribution is the integral of the pointwise 
%product of x and the probability of x, denoted as p(x). Integration works 
%fine on paper, but MATLAB is discrete and not continuous, so instead of 
%taking an integral, we have to sum up the pointwise product of x and p(x).

%The 2nd moment is simply the integral of the pointwise product of x^2 and
%p(x). Calculating the 2nd moment is useful because we can it to compute 
%the variance. 

%1. Uniform distribution
%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%compute the mean of x using the formula sum(x_1+x_2+... x_n)/n;
mean_uni         = ;
%compute the variance of x using the formula sum((x_i - x_mean).^2)/n;
var_uni          = ;
%given the data x_uniform and x_edges, use histcounts.m to compute p(x)
prob_uni         = ;
%calculate the expected value
expV_uni         = ;
%calculate the 2nd moment 
moment_2nd_uni   = ;
%calculate the variance of x using the expected value and the 2nd moment
var_uni_m2_m1    = ;
%--------------------------------------------------------------------------
%The expected value of the uniform distribution should be the same as the
%mean(x).
if abs(expV_uni - mean_uni) < 1e-4 && abs(var_uni_m2_m1 - var_uni) < 1e-4
    disp('Expected value = mean; Variance = E(x^2) - E(x)^2.');
end

%2. Gaussian distribution
%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%compute the mean of x using the formula sum(x_1+x_2+... x_n)/n;
mean_Gauss       = ;
%compute the variance of x using the formula sum((x_i - x_mean).^2)/n;
var_Gauss        = ;
%given the data x_uniform and x_edges, use histcounts.m to compute p(x)
prob_Gauss       = ;
%calculate the expected value
expV_Gauss       = ;
%calculate the 2nd moment 
moment_2nd_Gauss = ;
%calculate the variance of x using the expected value and the 2nd moment
var_Gauss_m2_m1  = ;
%--------------------------------------------------------------------------
%The expected value of the uniform distribution should be the same as the
%mean(x).
if abs(expV_Gauss - mean_Gauss) < 1e-4 && abs(var_Gauss_m2_m1 - var_Gauss) < 1e-4
    disp('Expected value = mean; Variance = E(x^2) - E(x)^2.');
end

% Visualize the means (expected values)
figure
subplot(1,2,1)
histogram(x_uniform, [x_unique, x_unique(end)+binSize]-binSize/2, ...
    'FaceAlpha', 0.5, 'EdgeColor','none'); hold on;
plot([expV_uni, expV_uni], [0, 12], 'r--', 'lineWidth', 2); hold off; 
ylim([0, 12]); yticks([]); xlabel('x'); title('Uniform distribution');
set(gca,'Fontsize',15);

subplot(1,2,2)
histogram(x_Gauss, [x_unique, x_unique(end)+binSize]-binSize/2, ...
    'FaceAlpha', 0.5, 'EdgeColor','none'); hold on
plot([expV_Gauss, expV_Gauss], [0, 42], 'r--', 'lineWidth', 2); hold off; 
ylim([0, 42]); yticks([]); xlabel('x'); title('Gaussian distribution');
set(gca,'Fontsize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.5, 0.4]);

%% Part 3: Covariance matrix and affine transformation
%This part of the tutorial aims to walk you through how to calculate the 
%covariance matrix and pearson correlation coefficient by hand. 

%first load the 2d data 
load('Lab6_fake2d_data.mat');
%let's visualize the fake data sampled from a multivariate normal distribution.
mvGauss_x = X_mvGauss(:,1);
mvGauss_y = X_mvGauss(:,2);
num_mvG   = length(mvGauss_x);

figure
scatterhist(mvGauss_x, mvGauss_y, 'Location','SouthEast','Direction','out');
xlim([-2,6]); ylim([-4,10]); axis square
xlabel('x'); ylabel('y'); set(gca,'FontSize',15)
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.7]);

%% Exercise 3
%1. Covariance matrix
%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%compute the mean for x and y 
%Bonus: use matrix multiplication instead of mean.m 
mvGauss_x_mean = ;
mvGauss_y_mean = ;
%put them together in a column vector
mu_mvGauss = [mvGauss_x_mean; mvGauss_y_mean];
%The mean's used to generate the data is [2, 3], compare this with 
%mu_mvGauss, they should be very similar.
disp(mu_mvGauss);

%compute demeaned responses by subtracting the mean from the data (do it
%separately for x and y). Again, we encourage you to use matrix
%multiplication.
mvGauss_x_demeaned = ;
mvGauss_y_demeaned = ;
%put the demeaned responses in a matrix
mvGauss_demeaned = [mvGauss_x_demeaned, mvGauss_y_demeaned]';

%Compute the sample covariance using matrix multiplication
C_d = ;
%The covariance matrix used to generate the data is [1 1.5; 1.5 3], compare
%this with C_d, they should be very similar.
disp(C_d);
%--------------------------------------------------------------------------

%Visualize it
figure
scatter(mvGauss_x, mvGauss_y, 100, 'filled', 'MarkerFaceColor','b',...
    'MarkerFaceAlpha',0.2); hold on;
scatter(mu_mvGauss(1), mu_mvGauss(2), 200, '+', 'lineWidth', 5,...
    'MarkerFaceColor','k', 'MarkerEdgeColor','k'); hold on;
xlim([-2,6]); ylim([-4,10]); axis square
xlabel('x'); ylabel('y'); set(gca,'FontSize',15)
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.3, 0.6]);

%2. Affine transformations
%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%To do Affine transformations, we first need to compute demeaned responses,
%which we've already done above. The next step is to get matrix M.
%first compute s_x and s_y (avoid using function std.m)
sd_mvGauss_x  = ;
sd_mvGauss_y  = ;
M             = ;
mvGauss_trans = ;
C_b           = ;
%You will notice that the diagonals are all 1's after we re-center and
%normalize the data. The element in the 1st row 2nd column is pearson
%correlation coefficient.
disp(C_b); 

%pearson correlation coefficient
r = C_d(1,2)/(sd_mvGauss_x*sd_mvGauss_y);
disp(r);
%--------------------------------------------------------------------------

%% PART 4: Distributions and sampling
% This part of the tutorial aims to familiarize you with different
% distributions (uniform, the sum of rolls of two dice, gaussian,
% cumulative gaussian, poisson), and also walk you through how to 
% simulate data given these distributions.

%% Exercise 4a 
%There are a total of 6 different rolls for a die
numV       = 6;
%We can sample a few times (e.g., 10) or sample a bunch of times (e.g., 1e6)
%Assume that the more samples we draw, the simulated distribution will
%get closer and closer to a uniform distribution.
numSamples = round(10.^(1:1:6));
%initialize matrix numRoll that stores the number of times we get each roll
numRoll    = NaN(length(numSamples),numV);

%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
for i = 1:length(numSamples)
    %There are two ways of simulating the data:
    %1st: we can just resample [1,2,3,4,5,6] a bunch of times, and compute 
    %the number of times you get each roll.
    %Hint: use randsample.m and histcounts.m
    
%     samps = ;
%     numRoll(i,:) = ;

    %2nd: we can just sample from a standard uniform distribution (from 0
    %to 1), and then compute the how many times the samples fall between
    %[0,1/6], [1/6, 2/6], [2/6, 3/6], [3/6, 4/6], [4/6, 5/6], and [5/6, 1].
    %Hint: use rand.m and histcounts.m 
    samps = ;
    numRoll(i,:) = ;
end
%--------------------------------------------------------------------------

%Let's visualize the uniform distribution and the probability of getting
%each roll from simulation. The code below does not require changes.
figure
subplot(1,2,1)
x = 1:numV; y = ones(1,numV)./numV; bar(x,y,'LineWidth',2);
title('Distribution for a fair die roll');
xlabel('x'); ylabel('P(X=x)'); ylim([0,0.4]); yticks(1/6); yticklabels('1/6');
box off; set(gca,'FontSize',15);

subplot(1,2,2)
for i = 1:length(numSamples)
    bar(x, numRoll(i,:)./numSamples(i),'y','LineWidth',2);
    xlabel('x'); ylabel('P(X=x)'); ylim([0,0.4]); 
    title(['Simulation: number of samples = ', num2str(numSamples(i))]); box off
    yticks(1/6); yticklabels('1/6'); set(gca,'FontSize',15);
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.5, 0.5]);
    pause(1)
end

%% Exercise 4b 
%There are 11 different possible sums of rolls of two dics
sum2dice = 2:12;
%the probability of getting each summed roll
numComb  = numV*numV;
probSum2 = [1:5,6,5:-1:1]./numComb;
%initialize matrix numSums that stores the sum of rolls of two fair dice
numSums  = NaN(length(numSamples),length(sum2dice));

%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
for i = 1:length(numSamples)
    %There are two ways of simulating the data:
    %1st: we roll two dice, compute the sum, and then compute how many
    %times we get each summed roll.
    %Hint: use randsample.m and histcounts.m
    
%     samps_die1   = ;
%     samps_die2   = ;
%     samps_sums   = ;
%     numSums(i,:) = ;

    %2nd: we can just sample from a standard uniform distribution (from 0
    %to 1), and then compute the how many times the samples fall between
    %[0,probSum2(1)], [probSum2(1), probSum2(1)+probSum2(2)], ....
    %Hint: use rand.m, histcounts.m and cumsum.m
    samps = ;
    numSums(i,:) = ;
end
%--------------------------------------------------------------------------

%Let's visualize the distribution and the probability of getting each
%possible sum of rolls of two fair dice from simulation. The code below 
%does not require changes.
figure
subplot(1,2,1)
x = sum2dice; y = probSum2; bar(x,y,'LineWidth',2);
title('Distribution for sum of rolls of two fair dice')
xlabel('x'); ylabel('P(X=x)'); ylim([0,0.25]); box off
set(gca,'FontSize',15);

subplot(1,2,2)
for i = 1:length(numSamples)
    bar(x, numSums(i,:)./numSamples(i),'y','LineWidth',2);
    xlabel('x'); ylabel('P(X=x)'); ylim([0,0.25]); 
    title(['Simulation: number of samples = ', num2str(numSamples(i))]); box off
    set(gca,'FontSize',15);
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.65, 0.5]);
    pause(1)
end

%% Exercise 4c 
%Define the mean and the standard deviation for the gaussian
mu    = 0;
sig   = 1;
x     = -4:0.01:4;
Gauss = normpdf(x, mu, sig);

%THINKING EXERCISE: 
%When you make sig smaller (e.g., 0.1), and print out max(Gauss), you'd
%notice that the value is greater than 1. How can probability exceed 1?
%Why does it happen and how do we fix it?

%Given the Gaussian distribution, let's now make the cumulative gaussian.
%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
cumGauss  = ; 
%--------------------------------------------------------------------------

%Now let's visualize the gaussian and the cumulative gaussian, and how they
%relate to each other. The code below does not require changes.
figure
x_slc = -4:0.2:4;
for i = 1:length(x_slc)
    subplot(1,2,1)
    plot(x, Gauss, 'Color','k','lineWidth',2); hold on;
    idx_start = find(abs(x - x_slc(i)) < 1e-3,1); xx = x(1:idx_start);
    patch([xx, fliplr(xx)], [Gauss(1:idx_start), ...
        zeros(1,idx_start)], [13, 183, 200]./255, 'FaceAlpha',0.2); hold on;
    plot([x_slc(i), x_slc(i)], [0,0.5],'Color', 'r','lineWidth',2); hold off
    xlabel('X=x');  ylabel('P(X = x)'); title('Gaussian distribution');
    set(gca,'Fontsize',15);
    
    subplot(1,2,2)
    plot(x(1:idx_start), cumGauss(1:idx_start), 'Color', [13, 183, 200]./255, ...
        'lineWidth', 3); hold off; ylim([0,1]); xlim([x(1), x(end)]); hold on
    plot([x_slc(i), x_slc(i)], [0, cumGauss(idx_start)],'r-','lineWidth',2); hold on
    plot([x_slc(1), x_slc(i)], [cumGauss(idx_start), cumGauss(idx_start)],'r--','lineWidth',2); hold off
    xlabel('X=x');  ylabel('P(X < x)'); title('Cumulative gaussian distribution');
    set(gca,'Fontsize',15);
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.55, 0.35]);
    pause(0.2)
end

%% Exercise 5 (optional)
%In each trial, participants are presented with an auditory and a visual
%stimulus with a temporal discrepancy between them. The discrepancy can
%have various levels, ranging from -400 to 400 ms with an increment of 50
%ms. Positive values represent the visual stimulus coming before the
%auditory stimulus; negative values represent the auditory stimulus coming
%first. After stimulus presentation, participants are asked to report
%whether they judge the temporal order, i.e., report which stimulus comes
%first (V or A). Each temporal discrepancy (a.k.a. stimulus onset
%asynchrony; SOA) is tested multiple times.

%let's first define some experimental info
%the levels of SOA
t_diff        = -400:50:400;
%the total number of levels
len_deltaT    = length(t_diff);
%the number of trial for each audiovisual pair 
numTrials     = 20; 

%sounds are normally perceived faster as visual stimuli by ~60ms. 
%In other words, participants perceive an auditory and a visual stimulus 
%as simultaneous when the auditory stimulus is delayed by 60ms.
mu_delta_t    = 60; 
%Sigma controls participants' threshold. A high value represents
%participants are really bad at the task; a low value means participants
%are able to tell even if the temporal offset is very small.
sigma_deltaT  = 80;

%Define the cumulative Gaussian
P_reportV_1st = normcdf(t_diff, mu_delta_t, sigma_deltaT);

%Simulation
%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------





%sim_prob_V1st is a column vector that stores the probabilities of
%reporting V-first given simulated data. There are a lot of ways of doing
%it. To not limit your creativity, I deleted all the steps before this.
sim_prob_V1st = ;
%--------------------------------------------------------------------------

%Now let's visualize the psychometric curve and simulated data. If the 
%code you write above is correct, then it should be very close to the curve.
%Hard to tell? Try increase numTrials from 20 to 1000.
%The code below does not require changes.

figure
plot(t_diff, P_reportV_1st, 'Color',[13, 183, 200]./255, 'lineWidth', 3,...
    'lineStyle','--'); hold on;
scatter(t_diff, sim_prob_V1st', 300,'filled', 'MarkerFaceColor',...
    0.5.*[13, 183, 200]./255, 'MarkerEdgeAlpha', 0, 'MarkerFaceAlpha',0.5); hold on
xlim([-400, 400]); ylim([0,1]); box off; 
xlabel(['$t_A - t_V$','(ms)'],'Interpreter','latex'); 
ylabel('Probability of reporting ''V-first'''); xticks(t_diff(1:2:end)); 
yticks([0,0.5,1]); legend({'Psychometric function',...
    'Simulated data'},'Location','northwest'); legend boxoff
set(gca,'Fontsize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.5]);
