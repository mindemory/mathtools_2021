%MathTools Lab9: Bootstrapping, Cross-validation, Model comparison
clear all; close all; clc; rng(1);

%% Part 1 - Bootstrapping
%In a medical study, doctors aimed to examine whether aspirin can reduce 
%the incidents of heart attacks. To this aim, they recruited a large amount
%of subjects and divided them to aspirin group and placebo group. After 
%receiving the aspirin or the placebo pill for a period of time, doctors
%observed that people in the aspirin group ended up only having 104 heart
%attacks incidents, and people in the placebo group ended up having 189
%incidents.

%Given the data, we can compute the percentages of the heart attack
%incident for both groups, and compute a ratio. The fact that the ratio is
%smaller than 1 suggests that aspirin therapy is effective in preventing
%heart attacks. But how sure are we of this estimate? Does the 95%
%confidence interval include 1?

%------------------------YOUR CODE STARTS HERE-------------------------
%aspirin group
aspirin_heart   = 104;
aspirin_total   = 11037;
aspirin_data    = ; 
%placebo group
placebo_heart   = 189;
placebo_total   = 11034;
placebo_data    = ;
% Calculate statistic for original sample
ratio_empirical = ; 
disp(['The ratio computed from empirical data is ', num2str(ratio_empirical)])

%Let's say we bootstrap (sample with replacement) 10,000 times
n_boot          = 1e4; 
%ratio_boot is used to store the ratio for each bootstrapped dataset
ratio_boot      = zeros(n_boot, 1);
for i = 1:n_boot
    % resample aispiring data
    boot_asp         = ;
    % resample placebo data 
    boot_placebo     = ; 
    n_boot_asp       = sum(boot_asp);
    n_boot_placebo   = sum(boot_placebo);    
    % Calculate statistic for resampled data      
    ratio_boot(i)    = ; 
end

% Find 95% confidence interval
ratio_boot_sorted = ; % Sort ratio_boots from lowest to highest
lower_bound_ind   = ; % Find index of 2.5% value
upper_bound_ind   = ; % Find index of 97.5% value

% Use lower/higher index to find value corresponding to 2.5%/97.5% position
lower_bound       = ;
upper_bound       = ; 
%----------------------------------------------------------------------

%% Part 1 - Bootstrapping (continued)
%This section does not require changes.

cMAP1  = [50,205,50; 255,165,0]./255;
figure
h(1) = histogram(ratio_boot, 'FaceColor',cMAP1(1,:), 'FaceAlpha', 0.3, ...
    'EdgeColor', cMAP1(1,:)); hold on
h(2) = plot([1, 1]*ratio_empirical, [0, 1]*700,'-', 'lineWidth',3);
h(3) = plot([1, 1]*lower_bound, [0, 1]*700,'--', 'Color', 'k', 'lineWidth',3);
h(4) = plot([1, 1]*upper_bound, [0, 1]*700,'-.', 'Color', 'k', 'lineWidth',3);
hold off; xlabel('Ratio'); ylabel('Counts'); box off
yticks(0:300:600); xticks(round([lower_bound, ratio_empirical, upper_bound],2));
legend(h, {'Bootstrapped ratios', 'Empirical ratio', '95% CI (lower bound)', ...
    '95 % CI (upper bound)'});
set(gca,'FontSize', 15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.4]);
set(gcf,'PaperUnits','centimeters','PaperSize',[25 15]);
saveas(gcf, 'Fig1', 'pdf'); 

%The fact that the 95% confidence interval does not include 1 implies that 
%we can be very confident that the aspirin therapy is effective in
%preventing heart attacks.

%% Part 2 - Cross validation (Leave-one-out)
%Cross-validation is a resampling method that uses different portions of 
%the data to test and train a model on different iterations. The goal of 
%cross-validation is to test the model's ability to predict new data that 
%was not used in estimating it, in order to flag problems like overfitting 
%and to give an insight on how the model will generalize to a new (or 
%unknown) dataset. 

%let's first load the file
load regress1.mat %x: x-values; y: y-values
%visualize the data
figure
scatter(x, y, 100, 'filled', 'MarkerFaceColor', cMAP1(1,:),...
    'MarkerFaceAlpha',0.5);
xlabel('x'); ylabel('y'); xlim([-2,3]); ylim([-3,4]); grid on
set(gca,'FontSize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.3, 0.45]);
set(gcf,'PaperUnits','centimeters','PaperSize',[20 15]);
%saveas(gcf, 'Fig2', 'pdf'); 

%Now, we will fit the data with polynomial linear models without using 
%cross validation. This will serve as a comparison to see how cross 
%validation alters our interpretation of the 'best' model.

% Number of data points
n_pts    = length(y); 
% All regressors that we will use for each fit
XX       = [ones(n_pts,1), x, x.^2,x.^3,x.^4,x.^5,x.^6,x.^7,x.^8,x.^9,x.^10,x.^11]; 
% Number of regressors (should be 12)
n_models = size(XX,2); 
order    = 0:(n_models-1);
%initialize matrices mse_vec (which stores mean squared error) and fit_mat 
%(which stores predicted y values)
mse_vec  = zeros(n_models, 1); 
fit_mat  = zeros(n_models, n_pts); 
betas    = cell(1,n_models);

%--------------------------YOUR CODE STARTS HERE---------------------------
% Loop through each polynomial order
for i = 1:n_models 
    % Get betas for polynomial order
    betas{i}      = ;
    % Get fit by multiplying betas with regressors
    fit_mat(i, :) = ; 
    % Find and save mean square error
    mse_vec(i)    = ; 
end
%--------------------------------------------------------------------------

% Visualize fits - which one looks best?
figure
for i = 1:n_models
    subplot(3,4,i)
    scatter(x, y, 100, 'filled', 'MarkerFaceColor', cMAP1(1,:),...
        'MarkerFaceAlpha',0.5); hold on;
    plot(x, fit_mat(i, :), 'LineWidth', 2, 'Color',cMAP1(2,:)); hold on
    xlabel('x'); ylabel('y'); xlim([-2,3]); ylim([-3,4]); grid on
    title_str = append('Order: ', int2str(i-1)); title(title_str)
    set(gca,'FontSize',15);
end
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.7, 0.8]);
set(gcf,'PaperUnits','centimeters','PaperSize',[35 25]);
%saveas(gcf, 'Fig2', 'pdf'); 

%% Part 2 - Cross validation (Leave-one-out) continued
% Next let's re-do the model fitting with leave-one-out cross validation 
% and then re-evaluate our models.

%------------------------YOUR CODE STARTS HERE-----------------------------
% Let's fit this data with polynomials:
[mse_train, mse_test] = deal(NaN(n_pts, n_models)); % Matrix to hold MSE values for each model for each cross validation
for i = 1:n_models 
    for j = 1:n_pts
        %Make j the test set
        x_train = ;
        x_test  = ;
        y_train = ;
        y_test  = ;
        
        % Compute the fit (Get betas using training data)
        betas         = ; 
        fit_train     = ;
        mse_train(j,i)= ;
        
        % Test the fit (Get model's prediction for test point)
        fit_test      = ; 
        mse_test(j,i) = ; % MSE of test fit
    end
end

% Find LOWEST MSE
mean_mse_test      = mean(mse_test);
mean_mes_train     = mean(mse_train);
[min_mse, min_ind] = min(mean_mse_test);
%--------------------------------------------------------------------------

% Plot the mean MSE 
figure
plot(order, mean_mse_test, 'linewidth',2, 'Marker', 's', 'Color',...
    cMAP1(2,:), 'MarkerSize',15); hold on
scatter(min_ind-1, min_mse,150,'rs','filled'); hold on
plot(order, mean_mes_train , '-*','linewidth',2, 'Color',...
    cMAP1(1,:), 'MarkerSize',6); hold off
xlim([0,n_models-1] + [-0.5, 0.5]); xlabel('Order'); grid on; 
yticks(0:1:5); ylabel('Mean squared error (MSE)'); box off;
legend({'MSE (test set)','MSE of the ''best'' model', 'MSE (training set)'});
set(gca,'FontSize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.3, 0.7]);
set(gcf,'PaperUnits','centimeters','PaperSize',[25 30]);
%saveas(gcf, 'Fig3', 'pdf'); 

% Which model is best?
disp(['The best model is when order = ', num2str(order(min_ind))]);

%% Part 3 - Model Comparison (AIC, BIC)
%In this section, we will fit a psychometric function to simulated data,
%and compute AIC/BIC using the negative log likelihood

%first load the fake data
D         = load('TOJ_fakeData.mat', 'fakeData');

%The first cell stored in D.fakeData is the SOA (the timing of the auditory 
%stimulus - the timing of the visual stimulus in ms). Positive values 
%represent the visual stimulus coming before the auditory stimulus; 
%negative values represent the auditory stimulus coming first. Like in a
%real experiment, all the SOA's are randomized.
t_diff    = D.fakeData{1}; 

%The second cell stored in D.fakeData is the corresponding binary responses
%1: V first; 0: A first
bool_V1st = D.fakeData{2};

%we can also code the responses in the opposite way
%1: A first; 0: V first
%(this becomes handy when you compute negative log likelihood later on) 
bool_V2nd = 1 - bool_V1st;

%before fitting, let's visualize the fake data
s_unique  = unique(t_diff); %unique SOA's
lenS      = length(s_unique); %the number of unique SOA's 
nTTrials  = length(t_diff); %number of total trials
numTrials = nTTrials/lenS; %number of trials per level

%As in a real experiment, all the SOA's are randomized. We want to
%visualize the proportion of V-first responses as a function of SOA, so we
%have to first organize the responses. We do so by storing responses for
%the same SOA in one row.
r_org     = NaN(lenS, numTrials);
for i = 1:lenS; r_org(i,:) = bool_V1st(t_diff == s_unique(i)); end
%compute the number of V-first responses for each SOA
nT_V1st      = sum(r_org,2)';
%compute the proportion of V-first responses for each SOA
P_V1st       = nT_V1st/numTrials;

%Plotting starts here
cMAP = [200, 40, 40; 255, 128, 0; 13, 183, 200]./255;
figure
scatter(s_unique, P_V1st, 300,'filled', 'MarkerFaceColor',...
    0.5.*cMAP(3,:), 'MarkerEdgeAlpha', 0, 'MarkerFaceAlpha',0.5); hold on
xlim([-400, 400]); ylim([0,1]); box off; 
xlabel(['$t_A - t_V$','(ms)'],'Interpreter','latex'); yticks([0,0.5,1]);
ylabel('Probability of reporting ''V-first'''); xticks(s_unique(1:2:end));  
set(gca,'Fontsize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.5]);
set(gcf,'PaperUnits','centimeters','PaperSize',[35 25]);
saveas(gcf, 'Fig4', 'pdf'); 

%% Part 3 - Model Comparison (AIC, BIC) continued
%Let's test three hypothesized models:

%M1: we assume that this participant does not have any bias (i.e., 
%PSS corresponds to a difference of 0 in time). Additionally, we assume 
%that this participant is always attentive to the stimuli (i.e., lapse rate
%= 0). The only unknown free parameter we need to measure is sigma (the
%slope of the psychometric curve).

%M2: we assume that this participant has a bias (i.e., the physical
%temporal difference has to be nonzero for him/her to perceive
%simultaneity. As in M1, this participant is assumed to pay attention to
%the task all the time. Therefore, this model has two free parameters, the
%center and the slope of the psychometric curve. 

%M3: we assume that this participant occassionally made mistakes during the
%experiment (i.e., nonzero lapse rate). This model has three free
%parameters, the center and the slope of the psychometric curve as well as
%the lapse rate.

%Note that these three models are NESTED!!!

%------------------------YOUR CODE STARTS HERE-----------------------------
%Let's test three models:
numM     = 3; %number of models
numP     = [1,2,3]; %number of free parameters for M1, M2, M3 respectively
fmat     = {@(x, p) normcdf(x, 0, p);... %M1
            @(x, p) normcdf(x, p(1), p(2));...%M2
            @(x, p) };%M3

%define upper and lower bounds
lb       = {80, [-50, 80], [ -50, 80, 1e-1]};
ub       = {200, [150, 200], [150, 200, 0.2]};
init_fun = @(a,b) rand(1,length(a)).*(b-a) + a;
options  = optimoptions(@fmincon,'MaxIterations',1e5,'Display','off');
[min_NLL, L_test] = deal(NaN(1,numM));
estP              = cell(1,numM);
[AIC, BIC]        = deal(NaN(1, numM));

%loop through the three models
for m = 1:numM
    %the initial point for matlab to start searching
    init = init_fun(lb{m}, ub{m});
    %negative log likelihood
    nLogL = @(p) ;
    %use fmincon.m to fit
    [estP{m}, min_NLL(m)] = fmincon(nLogL, init,[],[],[],[],...
        lb{m}, ub{m},[],options);  
    %compute the AIC/BIC
    AIC(m) = ;
    BIC(m) = ;
end
%--------------------------------------------------------------------------

%plot the data along with the model fits
figure
lstyle = {'--','-.',':'};
scatter(s_unique, P_V1st, 300,'filled', 'MarkerFaceColor',...
    0.5.*cMAP(3,:), 'MarkerEdgeAlpha', 0, 'MarkerFaceAlpha',0.5); hold on
for m = 1:numM
    predP = fmat{m}(s_unique, estP{m});
    plot(s_unique, predP, 'Color', cMAP(m,:),'lineWidth',2,'lineStyle', lstyle{m}); hold on
end
xlim([-400, 400]); ylim([0,1]); box off; 
xlabel(['$t_A - t_V$','(ms)'],'Interpreter','latex'); yticks([0,0.5,1]);
ylabel('Probability of reporting ''V-first'''); xticks(s_unique(1:2:end)); 
legend({'Fake data','Model fits (M1)', 'Model fits (M2)', 'Model fits (M3)'},...
    'Location','northwest'); legend boxoff
set(gca,'Fontsize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.5]);

%plot AIC BIC
figure
subplot(2,1,1)
heatmap(AIC - min(AIC), 'XData',{'M1','M2','M3'}, 'Colormap', flipud(bone),...
    'ColorLimits', [0, 14], 'ColorbarVisible', 'off', 'GridVisible', 'off',...
    'FontSize', 15, 'YLabel', 'AIC - min(AIC)'); colorbar

subplot(2,1,2)
heatmap(BIC - min(BIC), 'XData',{'M1','M2','M3'}, 'Colormap', flipud(bone),...
    'ColorLimits', [0, 14], 'ColorbarVisible', 'off', 'GridVisible', 'off',...
    'FontSize', 15, 'XLabel','Model', 'YLabel', 'BIC - min(BIC)');colorbar
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.5]);

%% Part 4 - AIC/BIC vs. Cross validation
%Leave-one-out cross validation starts here
[min_NLL_LOOCV, L_test_LOOCV] = deal(NaN(numM, nTTrials));

%------------------------YOUR CODE STARTS HERE-----------------------------
%loop through all the trials
for i = 1:nTTrials
    %display the counter see where we are at (ends at 170)
    %disp(i)
    
    %select the trial indices
    idx_slc       = ;
    %selected SOA's
    t_diff_slc    = ; 

    %selected response
    bool_V1st_slc = ;
    bool_V2nd_slc = ;
    
    %loop through all the three models
    for m = 1:numM
        %the initial point for matlab to start searching
        init   = init_fun(lb{m}, ub{m});
        %negative log likelihood
        nLogL  = @(p) ;
        %use fmincon.m to fit
        [estP, min_NLL_LOOCV(m,i)] = fmincon(nLogL, init,[],[],[],[],...
            lb{m}, ub{m},[],options);  
        %compute the likelihood of 
        L_test_LOOCV(m,i) = ;
    end
end
%--------------------------------------------------------------------------

%% Part 4 - AIC/BIC vs. Cross validation continued.
%This part of the code shows you the results of cross-validation. No
%changes are required.

%define a function that find 95% confidence intervals
get95CI      = @(v,n) v([ceil(0.025*n), floor(0.975*n)]);
%compare the likelihood of M3 with that of M2
Lratio_M3_M2 = L_test_LOOCV(3,:)./L_test_LOOCV(2,:);
%compare the likelihood of M3 with that of M1
Lratio_M3_M1 = L_test_LOOCV(3,:)./L_test_LOOCV(1,:);
%compare the likelihood of M2 with that of M1
Lratio_M2_M1 = L_test_LOOCV(2,:)./L_test_LOOCV(1,:);
%put them together
Lratio       = {Lratio_M3_M2,Lratio_M3_M1,Lratio_M2_M1};

%print out the proportion of each likelihood ratio greater than 1
disp(['p(Likelihood ratio btw M3 & M2 > 1) = ', num2str(sum(Lratio_M3_M2>1)/nTTrials)]);
disp(['p(Likelihood ratio btw M3 & M1 > 1) = ', num2str(sum(Lratio_M3_M1>1)/nTTrials)]);
disp(['p(Likelihood ratio btw M2 & M1 > 1) = ', num2str(sum(Lratio_M2_M1>1)/nTTrials)]);

%call get95CI.m for computing confidence intervals
Lratio_CI   = arrayfun(@(idx) get95CI(sort(Lratio{idx}), nTTrials), 1:numM, 'UniformOutput', false);
%compute the mean likelihood ratio
Lratio_mean = arrayfun(@(idx) mean(Lratio{idx}), 1:numM);
edges       = 0.85:0.02:1.5;

%plot histograms for likelihood ratios
figure
for i = 1:numM
    subplot(numM, 1,i)
    histogram(Lratio{i}, edges,'FaceAlpha', 0.2); hold on; 
    errorbar(Lratio_mean(i), 40,0,0, Lratio_mean(i)-Lratio_CI{i}(1),...
        Lratio_CI{i}(2)-Lratio_mean(i), 'r-*'); hold on;
    plot([1,1],[0, 50],'k--');box off
    xlim([0.85, 1.5]); ylim([0, 50]);
    if i == 3; xlabel('Likelihood ratio');end; ylabel('Counts');
    if i == 1; title('M3 vs. M2');
    elseif i == 2; title('M3 vs. M1');
    else; title('M2 vs. M1');end
    set(gca,'FontSize',12);
end
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.7]);

