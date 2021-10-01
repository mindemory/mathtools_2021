%% Lab 3: Linear regression and PCA
% 1 Oct 2021
clc; clear; close all;
%% 1a. Linear Regression: Visual neuron 
% In this section, we will model the response of a simulated neuron
% responding to the luminance of a visual stimulus, and characterize its
% "spiking" with linear regression

% Create test data simulating a neuron firing in response to a light stimulus:

luminance = (1:10)'; % Arbitrary units to represent input intensities

% what is the spike response to different luminance? Model it as a linear
% amplification plus baseline and noise. Include a term to introduce
% experimental error to random trials

baseline = 10; noise = 2; amp = 2; experimentalError = (rand(10,1)>.5).*randn(10,1)*5;
spikes = (amp * luminance + baseline + randn(length(luminance),1).*noise) + experimentalError ; 

% Plot Luminance vs Spikes
figure(1); % we give these figures a designated handle so that we can repeatedly plot in them 
plot(luminance,spikes,'ko', 'MarkerSize',18); 
xlabel('Luminance')
ylabel('Spikes')
set(gca,'Fontsize',18)
set(gcf,'color','w')
xlim([0 11])
ylim([0 max(spikes)+10])

% Find a value of beta (i.e., a slope of the regression line) such that
% when it's multiplied to each of the luminance values, we minimize the
% distance to the actual measurements.
% aka we look for beta_op which minimizes ||y - betaOpt * x||^2

% Find betaOpt (optimal beta) via SVD!

[U, S, V] = svd(luminance);
betaOpt = V*pinv(S)*U'*spikes; % **Replace this code! 

% Now that we have our optimal weight to scale each luminance value, how
% well does our beta do at predicting the spikes? 

prediction = luminance*betaOpt; %our prediction is multiplying each luminance value by beta
predError = spikes - prediction; % get distance between measured spikes and our model prediction
predErrorOpt = predError'*predError; % get the squared distance

% Plot our prediction
hold on
plot(luminance, prediction, 'k', 'LineWidth', 3, ...
    'DisplayName', ['lin reg, sq error = ', num2str(predErrorOpt)])
legend('location', 'southoutside')

%  Was this the best beta? 
% Test a range of other beta values to confirm that your beta truly
% minimizes ||y - betaOpt * x||^2
numBeta = 200; testBetas = linspace(0, betaOpt*2, numBeta); % Try 200 betas over a range of 0:2*betaOpt
%sqrError = nan*ones(1,numBeta); % initialize 

prediction_mat = luminance * testBetas;
spikes_mat = spikes * ones(1, numBeta);
predError_mat = spikes_mat - prediction_mat;
sqrError_mat = diag(predError_mat' * predError_mat);
% for ii = 1:numBeta % Loop through all the betas
%     prediction = testBetas(ii) .* luminance;
%     % We need to now explicitly represent the distance metric
%     sqrError(ii) = sum((prediction-spikes).^2); %Sum of squares
% end

% now check if predErrorOpt is indeed optimal:
if min(sqrError_mat) >= predErrorOpt 
    % if the closest error you got by force was no smaller than your calculated beta opt....
   disp('seems like we found a great solution')
end

% Now plot the error for different betas
figure(2)
plot(testBetas,sqrError_mat,'b','linewidth',3) % plot the error of your set of test betas
line([betaOpt betaOpt], [min(ylim) max(ylim)], 'color','r','linewidth',2); % plot a line at your calculated optimal beta value
xlabel('Beta')
ylabel('Error metric')
set(gca,'fontsize',18)
set(gcf,'color','w')

%% 1b. Multiple Linear Regression: Visual neuron model with y-intercept

% What if we would like to add a y-intercept? We can do this with
% multiple linear regression if we model our independent variable (x)
% as a matrix whose columns contain the x values of the first order
% polynomial yPredicted = Beta0*x^0 + Beta1*x^1 (think y = b + mx). Solving
% this multiple linear regression will produce two beta values, Beta0 (scaling
% the y intercept term) and Beta1 (scaling the slope of the regression line)


% Create your regressor matrix and calculate the optimal betas with
% linear algebra

% create your "multivariate" data matrix X (containing x^0 and x^1 as
% columns)
X = [luminance.^0, luminance.^1]; % **Replace this code! 

% Calculate betaOpt 

[U_int, S_int, V_int] = svd(X);
betaOptYInt = V_int * pinv(S_int) * U_int' * spikes; % **Replace this code! 

% Now that we have our optimal weight to scale each luminance value, and our y-interecpt, how
% well does our beta (now a vector) do at predicting the spikes? 

predictionYInt =  X * betaOptYInt; % **Replace this code! 
predErrorYInt =  predictionYInt - spikes; % **Replace this code! 
predErrorOptYInt =  predErrorYInt' * predErrorYInt; % **Replace this code! 

% Plot our prediction with error value
figure(1)
hold on
plot(luminance, predictionYInt, '-.', 'LineWidth', 3, ...
    'DisplayName', ['regression with y-int, sq error = ', num2str(predErrorOptYInt)])
legend('location', 'southoutside')


% How did our beta do? first, create two beta vectors we will use to
% calculate our error surface:

nBetas = 200;
beta0 = linspace(-betaOptYInt(1),betaOptYInt(1)*2,nBetas);
beta1 = linspace(1,betaOptYInt(2)*2,nBetas);

% now compute the errors to make a surface representing your error
allerr = nan*ones(nBetas);
for ii = 1:nBetas
    for jj = 1:nBetas
        bb = [beta0(ii);beta1(jj)];
        allerr(ii,jj) = sum((X*bb-spikes).^2);
    end
end

% plot the contour and compare the old prediction error (without the
% y-int) with the new prediction error - did you do any better?
% what does it mean for our neuron that the model with the y-intercept has
% lower error?

figure(3);

[betaX,betaY] = meshgrid(beta0,beta1);

s = surf(betaX,betaY,allerr', 'DisplayName', 'sq err');
hold on
scatter3(betaOptYInt(1), betaOptYInt(2), predErrorOptYInt, 100,'r', 'filled', ...
        'DisplayName', 'Optimal beta')
hold on
scatter3(0, betaOpt, predErrorOpt, 100,'m', 'filled', ...
        'DisplayName', 'Old optimal beta')
xlabel('beta0')
ylabel('beta1')
zlabel('squared error')
rotate3d on
alpha 0.5
s.EdgeColor = 'none';
legend('location', 'northoutside')


%% Clear your workspace and close all figures
clear all;
close all;

%% 2a. Multiple Linear Regression: Auditory Neuron
% Now let's try an example of multiple linear regression. The structure
% mult_linreg contains three fields: freq1, freq2 (the dB intensities of two
% frequencies of sound stimuli) and response, the response of an auditory
% neuron. 

% First, load and plot the data ( in 3D!)
load('mult_linreg.mat');

figure(3)
plot3(data.freq1, data.freq2, data.response, '.', 'MarkerSize',20, 'DisplayName', 'Auditory neuron response') % **Replace this code!

xlabel('Freq 1')
ylabel('Freq 2')
zlabel('Firing rate')
set(gca,'fontsize',18)
rotate3d on
grid on

% Perform linear regression

X = [data.freq1, data.freq2];
y = data.response;
[U, S, V] = svd(X);
betaOpt = V * pinv(S) * U' * y;% **Replace this code!

% Calculate error between the prediction from our model, and the measured
% values

prediction =  X * betaOpt; % **Replace this code! 
predError =  prediction - y; % **Replace this code! 
predErrorOpt =  predError' * predError; % **Replace this code! 


% plot our linear model (it's a plane!)
x = linspace(0, max(data.freq1*1.2),100)';
y = linspace(0, max(data.freq2*1.2),100)';

[xx, yy] = meshgrid(x,y);
zz = betaOpt(1)*xx + betaOpt(2)*yy;
figure(3)
hold on
s = surf(xx,yy,zz, 'DisplayName', ['lin reg, sq error = ', num2str(predErrorOpt)]);
alpha 0.5
s.EdgeColor = 'none';
s.FaceColor = [1,.8,.9];
legend('location','northoutside')


% Was this the best beta?
% first, create vectors of your possible betas in both dimensions
% (hint: what did we do in section 1b?)
nBetas = 200;

beta0 = linspace(-betaOpt(1),betaOpt(1)*2,nBetas);
beta1 = linspace(1,betaOpt(2)*2,nBetas);

% now compute the errors
allerr = nan*ones(nBetas);
for ii = 1:nBetas
    for jj = 1:nBetas
        bb = [beta0(ii);beta1(jj)];
        allerr(ii,jj) = sum(([data.freq1, data.freq2]*bb-data.response).^2);
    end
end

% plot the contour!
figure(4);
% in 3D, our possible errors are a surface instead of a line. We will use
% the meshgrid function to plot this surface.

[betaX,betaY] = meshgrid(beta0,beta1);
s = surf(betaX,betaY,allerr');
xlabel('beta0')
ylabel('beta1')
zlabel('squared error')
rotate3d on
alpha 0.5;
s.EdgeColor = 'none';

%% 2b. Multiple Linear Regression: Auditory neuron with baseline firing
% What if we suspect the auditory neuron has some baseline firing rate? We
% might want to include an intercept term in our regression. We can do this
% the same way we added an intercept term to the standard linear
% regression: by adding an x^0 term to our linear regression.

% Linear Regression 
newX = [data.freq1 .^0, data.freq1, data.freq2]; % **Replace this code!
[newU, newS, newV] = svd(newX);
y = data.response;
betaOptnew = newV * pinv(newS) * newU' * y; % **Replace this code!

% Calculate error between the prediction frorm our model, and the measured
% values

prediction_new =  newX * betaOptnew; % **Replace this code! 
predError_new =  prediction_new - y; % **Replace this code! 
predErrorOpt_new =  predError_new' * predError_new; % **Replace this code!  

% plot our linear model (it's a plane!)

xx = linspace(0,max(data.freq1)*1.2,nBetas)';
yy = linspace(0,max(data.freq2)*1.2,nBetas)';


[newBetaX,newBetaY] = meshgrid(xx,yy);
newS = betaOptnew(1)*ones(size(xxxx)) + betaOptnew(2)*xxxx + betaOptnew(3)*yyyy; % **Replace this code! 
figure(3)
hold on
s = surf(newBetaX,newBetaY,newS, 'DisplayName', ['lin reg + zint, sq error = ', num2str(predErrorOpt_new)]);
alpha 0.5
s.EdgeColor = 'none';
s.FaceColor = [.9,.5,.1];
legend('location','northoutside')

% How else might you improve the fit of our surface? 
%% Tidy your workspace
clear all;
close all;

%% 3. Intro to PCA
% now let's look at what PCA does using a toy data set. First, load the
% sample PCA data. The columns of this matrix are the values of two
% correlated measurements (e.g. height and body weight). Load this data
% set...

load ('pca_example.mat')

% Plot the data
figure(5)
subplot(1,3,1)
plot(X(:, 1),X(:, 2),'k.', 'DisplayName', 'data') % **Replace this code! 

% What do you notice? As a reminder, in order to calculate the
% eigenvectors, our data must be centered around the mean values of each
% variable

% your code
meanX = mean(X, 1);
meanX_mat = ones(size(X, 1), 1) * meanX;
centeredX = X - meanX_mat;
figure(5)
subplot(1,3,1)
plot(centeredX(:, 1),centeredX(:, 2),'k.', 'DisplayName', 'data')% **Replace this code! 

% Calculate the eigenvectors and eigenvalues of your data matrix

% First, find the covariance matrix
n = size(centeredX, 1);
C = (centeredX' * centeredX)/(n - 1); % **Replace this code!

% Then the eigenvectors and eigenvalues
[V,D]=eig(C); 

% unlike the SVD, eig() does not return sorted eigenvalues
[d,ind] = sort(diag(D), 'descend'); 
D = D(ind,ind);
V = V(:,ind);

% The columns of V are the eigenvectors (principal components axes) of our
% data matrix
PC1 = V(:, 1); % **Replace this code!
PC2 = V(:, 2); % **Replace this code!

% Now, plot the axes of greatest variance (the eigenvectors * eigenvalues)
v1= 0; % **Replace this code! 
v2= 0; % **Replace this code!

subplot(1,3,1)
hold on
plot([0 v1(1)],[0 v1(2)],'b-', 'LineWidth', 2, 'DisplayName', 'PC 1');
hold on
plot([0 v2(1)],[0 v2(2)],'r-', 'LineWidth', 2, 'DisplayName', 'PC 2');
l = legend;
l.Position = [0.0838 0.5611 0.1782 0.2062];
axis equal
xlim([-10, 10])
ylim([-10, 10])
grid on

set(gca,'fontsize',18)

% We can use the axes we just extracted to reduce the dimensionality of our
% data (i.e., represent our data in one of the two dimensions that
% represent the greatest variance in the data set). We do this by
% projecting our data on to these principal component axes
 
projPC1 =  0; % **Replace this code!
subplot(1,3,2)
plot(X(:,1),X(:,2),'k.')
hold on
scatter(projPC1(1,:), projPC1(2,:),10,'b','filled')
hold on

for ii = 1:length(X)
    plot([X(ii,1) projPC1(1,ii)], [X(ii,2) projPC1(2,ii)], 'color',[0.5, 0.5, 0.5])
end

axis equal
xlim([-10, 10])
ylim([-10, 10])
line(xlim, [0 0],'color',[.5 .5 .5]);
line([0 0], ylim,'color',[.5 .5 .5]);
grid on
title('Data projected onto 1st PC')
set(gca,'fontsize',18)


% Now project data onto second principal component

projPC2 =  0; % **Replace this code!
subplot(1,3,3)
plot(X(:,1),X(:,2),'k.')
hold on
scatter(projPC2(1,:), projPC2(2,:),10,'r','filled')

hold on
for ii = 1:length(X)
    plot([X(ii,1) projPC2(1,ii)], [X(ii,2) projPC2(2,ii)], 'color',[0.5, 0.5, 0.5])
end
axis equal
xlim([-10, 10])
ylim([-10, 10])
line(xlim, [0 0],'color',[.5 .5 .5]);
line([0 0], ylim,'color',[.5 .5 .5]);
grid on
title('Data projected onto 2nd PC')
set(gca,'fontsize',18)

f = figure(5);
f.Position = [1 41 1536 748.8000];

% While we don't need to reduce the dimensionality of this data set in
% order to more easily visualize it, we might use these axes to separate
% out groups within our data. Try plotting histograms of the data along the
% X, Y, and PC1 axes. Do you see a pattern within the data?

figure
nBins = 20;
subplot(3,1,1)
histogram(X(:,1), nBins)
ylabel('counts')
title('Var 1')

subplot(3,1,2)
histogram(X(:,2), nBins)
ylabel('counts')
title('Var 2')

% now, plot a histogram of your data projected on the first principal
% component
subplot(3,1,3)
histogram(projPC1, nBins)
ylabel('counts')
title('Projection onto PC1')
%% Tidy your workspace
clear all
close all

%% 4. Dimensionality reduction via PCA: Gene expression
% We can use PCA to understand large data sets with many variables. In this
% data set, the expression levels of 208 genes has been measured under 79
% conditions. Each of these genes has also been classified with a category.
% GeneData.mat contains a struct with two fields. Let's load it and take a
% look.

% data from Brown et al 2000
load('GeneData.mat')

% The first field is the category of
% each gene (TCA: tricarboxylic acid cycle, Resp: respiration, Ribo:
% Cytoplasmatic ribosomal proteins, Prot: Proteasome, Hist: Histones)

% The second field contains the change in expression levels of each of the
% 208 genes in each of the 79 experimental condition relative to baseline.
% Increases and decreases of expression level are reported by positive and
% negative values, respectively.

categoryLabels = GeneData.geneCategory';
expression = GeneData.geneExpression';
categories = unique(categoryLabels);
numCategories = length(unique(categoryLabels));

% Make sure that your data is oriented correctly so that you reduce the
% dimension you want to reduce! Since we are trying to reduce the
% "experiment conditions" dimension, we want the experimental conditions in
% the columns of our data matrix (in other words, a 208 x 79 matrix)

% What is the interpretation of dimensional reduction along the other
% dimension (the genes)?


% Consider first two experimental conditions, in a 2D scatter plot, plot
% the expression levels of condition 2 as a function of condition 1 and
% color the points according to the gene categories
colors = lines(numCategories);
figure

% Color data points by gene category 

for c = 1:numCategories
    hold on
    cat = find(strcmp(categoryLabels,categories{c}));
    scatter(expression(cat,1), expression(cat,2), 30, colors(c,:), 'filled', 'DisplayName', categories{c})
end
legend
set(gca,'fontsize',18)
xlabel('expression under condition 1')
ylabel('expression under condition 2')

% Do you notice any structure in the data? Can you cluster the data based
% on these two conditions? 

% Lets see if PCA can help us differentiate the groups of gene classes
% based on their change in expression over the 79 trials. To do this, we
% would like to reduce the dimensionality of the trials in order to
% visualize each gene's expression pattern in a lower dimension. 

% First, center the data (remember to correctly orient your data matrix!)

exp_centered =  0; % **Replace this code!

% Take SVD/compute eigenvectors of X'X 

% Your code here 

% Plot the singular values (or eigenvalues) in each principal direction
% (this captures how spread out your data is when projected down to the
% corresponding principal component)  
figure
plot(d, '.', 'MarkerSize', 20);
xlabel('principal component')
ylabel('spread')
set(gca,'fontsize',18)

% What does the data look like projected onto the first principal axes?

% Because we sorted the principal axes (eigenvectors) based on their eigenvalues, these
% axes should capture the greatest variability within the data.
projectPC12 =  0; % **Replace this code!
figure
for c = 1:numCategories
    hold on
    cat = find(strcmp(categoryLabels,categories{c}));
    scatter(projectPC12(cat,1), projectPC12(cat,2), 30, colors(c,:), 'filled', 'DisplayName', categories{c})
end
legend
set(gca,'fontsize',18)
xlabel('PC1')
ylabel('PC2')

% What does the data look like in based on the last two PCs? 

%Conversely, these axes should capture the least variability within the data
projectPCLast =  0; % **Replace this code!
figure
for c = 1:numCategories
    hold on
    cat = find(strcmp(categoryLabels,categories{c}))';
    scatter(projectPCLast(cat,1), projectPCLast(cat,2), 30, colors(c,:), 'filled', 'DisplayName', categories{c})
end
legend
set(gca,'fontsize',18)
xlabel('second to last PC')
ylabel('last PC')

% Project data onto first 3 principal components and show in 3d
proj_3pcs =  0; % **Replace this code!
figure
for c = 1:numCategories
    hold on
    cat = find(strcmp(categoryLabels,categories{c}))';
    scatter3(proj_3pcs(cat,1), proj_3pcs(cat,2), proj_3pcs(cat,3))
end
rotate3d on
grid on
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
%% Tidy your workspace
clear all
close all

%% 5. Dimensionality reduction via PCA: Reaching task
% Now, lets use PCA to analyze neural activity in motor cortex of monkeys
% during a reaching task. First, load the reaching data set:

load ('ReachData.mat')

% This data set contains neural signals recorded in macaque motor cortex
% during a center-out reaching task. The variable "R" contains the firing
% rate of 143 neurons recording during 158 trials, while the variable
% "direction" encodes the direction of the reach on each trial (direction 1
% is rightward, direction 2 is 45 deg, direrction 3 to straight forward (90
% deg) etc). 

directions = unique(direction);
numDirections = length(unique(directions));

% We would like to capture the total response (the activity of all 143
% neurons) in a format that is easier for us to visualize and interpret
% (i.e., in 2D). Ultimately we would like to see if we can predict the
% reach condition based on this measure of neural activity. 

% Before we start decomposing our data matrix, make sure you have the data
% in the correct orientation! We want the dimension that we are going to
% reduce (in this case, the 143 neurons) as the columns of R.

R = R';

% What is the interpretation of dimensional reduction along the other
% dimension (the trials)?


% Can you tell the reach direction from the firing rate of just two neurons?
% In a 2D scatter plot, plot the activity of the first neuron as a function
% of the second neuron and color the points according to the different
% direction for that trial. Does it look like the responses of these two
% neurons during each reaching condition is distinct from the responses
% during other conditions?

colors = jet(numDirections); % make a color map to color code the plot based on the reach condition
figure
for d = 1:numDirections
    hold on
    dirs = find(direction == directions(d));
    scatter(R(dirs,1), R(dirs,2), 30, colors(d,:), 'filled', 'DisplayName',['direction ' num2str(directions(d))])
end
legend
xlabel('Neuron 1 Firing Rate');
ylabel('Neuron 2 Firing rate');

% Use PCA in order to visualize the data from all the neurons in a lower
% dimension, and hopefully we will be able to separate out the different
% reach conditions!

% your code here 

% Plot the singular values (or eigenvalues) in each principal direction
% (this captures how spread out your data is when projected down to the
% corresponding principal component)  
figure
plot(0, '.', 'MarkerSize', 20); % **Replace this code!
xlabel('principal component')
ylabel('spread')
set(gca,'fontsize',18)

% What does the data look like projected onto the first two principal axes?
% This will give us a 2D plot of the data we can easily visualize. Color
% each point based on the reach condition during that trial. 
projectPC12 = 0; % **Replace this code!
figure
for d = 1:numDirections
    hold on
    dirs = find(direction == directions(d));
    scatter(projectPC12(dirs,1), projectPC12(dirs,2), 30, colors(d,:), 'filled', 'DisplayName',['direction ' num2str(directions(d))])
end
legend
set(gca,'fontsize',18)
xlabel('PC1')
ylabel('PC2')

