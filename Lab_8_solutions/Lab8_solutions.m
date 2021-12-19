%MathTools Lab 8
clear all; close all; clc; rng(1);

%% Exercise 1: Estimate the probability of heads, by observing samples
%Assume we have a unfair coin, the chance of getting a head is greater than
%the chance of getting a tail (i.e., P('H') = 0.5).
p_head_true             = 0.7;

%let's throw this coin 150 times, and each time, we compute the likelihood
%of a range of hypothesized P('H').
numFlips                = 150;
p_head_hyp              = 0.01:0.01:0.99;

%initialize the following matrices:
%flips                : stores binary values (1: head; 0: tail) for each 
%                           coin flip
%numHeads             : stores the number of heads you get up until each 
%                           time point
%L_p_hyp, LL_p_hyp    : the likelihood / the log likelihood of P('H') given a
%                           range of hypothesized p at each time point
%p_hat_byL, p_hat_byLL: the p value that corresponds to the highest
%                           likelihood value / the highest log likelihood 
%                           value at each time point
[flips,numHeads]        = deal(NaN(1,numFlips)); 
[L_p_hyp, LL_p_hyp]     = deal(NaN(numFlips, length(p_head_hyp)));
[p_hat_byL, p_hat_byLL] = deal(NaN(1,numFlips));

%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%let's loop through all the coin flips
for i = 1:numFlips
    %flip the coin, and see if it's a head to a tail (hint: rand.m)
    flips(i)          = rand < p_head_true;
    
    %compute the number of heads you got so far
    numHeads(i)       = nansum(flips);
    
    %compute the likelihood of each hypothesized p value
    L_p_hyp(i,:)      = binopdf(numHeads(i), i, p_head_hyp);
    
    %compute the log likelihood of each hypothesized p value
    LL_p_hyp(i,:)     = numHeads(i).*log(p_head_hyp) + ...
                        (i - numHeads(i)).*log(1 - p_head_hyp);
                    
    %find the p value that corresponds to the max likelihood and store it
    %in p_hat_byL(i)     
    [~, max_idx_byL]  = max(L_p_hyp(i,:));
    p_hat_byL(i)      = p_head_hyp(max_idx_byL);
    
    %find the p value that corresponds to the max likelihood and store it
    %in p_hat_byLL(i)      
    [~, max_idx_byLL] = max(LL_p_hyp(i,:));
    p_hat_byLL(i)     = p_head_hyp(max_idx_byLL);    
end
%--------------------------------------------------------------------------

%% Exercise 1. continued
%No changes are needed for this section

figure
for i = 1:20:numFlips %for plotting, change it to 1:20:numFlips
    %plot how the likelihood function changes as number of coin flips
    subplot(1,2,1)
    plot(p_head_hyp, L_p_hyp(i,:),'lineWidth', 3); hold on;
    scatter(p_hat_byL(i),max(L_p_hyp(i,:)),180,'r*'); hold on;
    text(p_hat_byL(i),max(L_p_hyp(i,:))-0.1, ['p_{max} = ', ...
        num2str(p_hat_byL(i))],'fontSize',15); hold off; box off;
    xlim([0, 1]); xlabel(['Hypothesized ', '$P(''H'')$'],'Interpreter','latex');
    ylabel('$L(P(''H'')|data)$','Interpreter','latex'); box off; 
    set(gca,'FontSize',15);
    
    %plot how the log likelihood function changes as number of coin flips
    subplot(1,2,2)
    plot(p_head_hyp, LL_p_hyp(i,:),'lineWidth', 3); hold on
    scatter(p_hat_byLL(i),max(LL_p_hyp(i,:))-0.1,180,'r*'); hold on
    text(p_hat_byLL(i),max(LL_p_hyp(i,:)), ['p_{max} = ', ...
        num2str(p_hat_byLL(i))],'fontSize',15); hold off; box off;
    xlim([0, 1]); xlabel(['Hypothesized ', '$P(''H'')$'],'Interpreter','latex');
    ylabel('$\log L(P(''H'')|data)$','Interpreter','latex'); box off; 
    set(gca,'FontSize',15);
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.7, 0.5]);
    sgtitle(['#heads = ', num2str(sum(flips(1:i))), ' out of #flips = ', num2str(i)]);
    pause(0.01)
end

%plot the proportion of heads as a function of coin flips 
%Notice that it converges to p_head_true = 0.7.
figure
plot(1:numFlips, numHeads./(1:numFlips), 'k.-', 'lineWidth', 2); hold on
plot([1, numFlips],[p_head_true, p_head_true], 'r--', 'lineWidth', 2); hold off
ylim([0,1]); box off; xlabel('Flip number'); ylabel('Proportion of Heads');
set(gca,'FontSize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.6, 0.45]);

%% Exercise 2. Estimate the posterior probability of the coin getting heads
%(incorporating prior knowledge)

%first we assume three different possible priors
%1. given past experience, you think it's most likely that coins are fair
prior_p_head_fair   = normpdf(p_head_hyp, 0.5,0.1);
prior_p_head_fair   = prior_p_head_fair./sum(prior_p_head_fair);

%2. you are in a underground casino, and you suspect that the coin in unfair
prior_p_head_biased = betapdf(p_head_hyp, 2, 5);
prior_p_head_biased = prior_p_head_biased./sum(prior_p_head_biased);

%3. you are a new born baby, and you have never seen a coin before
prior_p_head_uni    = ones(1,length(p_head_hyp))./length(p_head_hyp);
prior_all = [prior_p_head_fair; prior_p_head_biased; prior_p_head_uni];
ttl       = {'Suspect fair', 'Suspect biased', 'No idea'};

%visualize the priors
figure
for i = 1:size(prior_all,1)
    subplot(1,size(prior_all,1),i)
    plot(p_head_hyp, prior_all(i,:), 'lineWidth', 3); box off; grid on
    title(ttl{i}); xlabel('Hypothesized P(''H'')'); ylabel('Probability');
    set(gca,'FontSize',15);
end
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.8, 0.5]);

%compute the posterior each time you get a new coin flip
posterior_p_hyp = NaN(numFlips, size(prior_all,1), length(p_head_hyp));
L_p_hyp_1flip   = NaN(numFlips, length(p_head_hyp));

%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
for i = 1:numFlips
    %if this is your first coin flip, use the prior we define above
    if i == 1; prior = prior_all;
    %if you've already had some observations of the coin flips, the updated
    %prior of the current trial is the posterior from the previous trial.
    else; prior = squeeze(posterior_p_hyp(i-1,:,:)); 
    end
    
    %compute the likelihood of P('H') given the current coin flip (just 1
    %observation)
    L_p_hyp_1flip(i,:) = binopdf(flips(i), 1, p_head_hyp);
    
    %compute the posterior probability of P('H') by taking the prior into
    %account (make sure the posterior probability sums up to 1)
    for j = 1:size(prior_all,1)
        posterior_p_hyp_temp = L_p_hyp_1flip(i,:).*prior(j,:);
        posterior_p_hyp(i,j,:) = posterior_p_hyp_temp./sum(posterior_p_hyp_temp);
    end
end
%--------------------------------------------------------------------------

%% Exercise 2. continued
%No changes are needed for this section
figure
for t = 1:numFlips %for plotting, change it to 1:20:numFlips
    %visualize the priors (which are updating on each trial)
    for i = 1:size(prior_all,1)
        subplot(3,size(prior_all,1),i)
        if t == 1; plot(p_head_hyp, prior_all(i,:), 'lineWidth', 3); 
        else; plot(p_head_hyp, squeeze(posterior_p_hyp(t-1,i,:)), 'lineWidth', 3);  end
        box off; grid on
        if t == 1; title(ttl{i}); else; title(['Updated prior ', num2str(i)]);end
        xlabel('Hypothesized P(''H'')'); ylabel('Probability');
        set(gca,'FontSize',15);
    end 

    %visualize the likelihood given only one observation
    subplot(3,size(prior_all,1),5)
    plot(p_head_hyp, L_p_hyp_1flip(t,:), 'lineWidth', 3); box off; grid on
    if flips(t); r = 'H'; else; r = 'T'; end
    title(['Likelihood of P(''H''), Given coin flip = ', r]); 
    set(gca,'FontSize',15);

    %visualize the posterior probability of P('H')
    for i = 1:size(prior_all,1) 
        subplot(3,size(prior_all,1),6+i)
        plot(p_head_hyp, squeeze(posterior_p_hyp(t,i,:)), 'lineWidth', 3); 
        box off; grid on
        title('Posterior of P(''H'')'); xlabel('Hypothesized P(''H'')'); 
        ylabel('Probability'); set(gca,'FontSize',15);    
    end
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.8,1]);
    pause(0.05)
end

%You're probably wondering why we updated the likelihood of P('H') when
%more observations come in (lines 4-93), but here we update the prior and
%only compute the likelihood of P('H') of the current coin flip 
%(lines 100-180). Mathematically they are the same! 
%updated likelihood given all past coin flips x original prior 
%= likelihood of the current coin flip x updated prior given all past coin flips 

%% Exercise 3: Simulate 2IFC
%In each trial, participants are presented with an auditory and a visual
%stimulus with a temporal discrepancy between them. The discrepancy can
%have various levels, ranging from -400 to 400 ms with an increment of 50
%ms. Positive values represent the visual stimulus coming before the
%auditory stimulus; negative values represent the auditory stimulus coming
%first. After stimulus presentation, participants are asked to report
%whether they judge the temporal order, i.e., report which stimulus comes
%first (V or A). Each temporal discrepancy (a.k.a. stimulus onset
%asynchrony; SOA) is tested multiple times.

%let's first define levels of SOA (in ms)
t_diff        = -400:50:400;
%the number of levels
len_deltaT    = length(t_diff);
%the number of tested trials for each level
numTrials     = 20; 

%sounds are normally perceived faster as visual stimuli by ~60ms. 
%In other words, participants perceive an auditory and a visual stimulus 
%as simultaneous when the auditory stimulus is delayed by 60ms.
mu_delta_t    = 60; 
%Sigma controls participants' threshold. A high value represents
%participants are really bad at the task; a low value means participants
%are able to tell even if the temporal offset is very small.
sigma_deltaT  = 80;

%On a small proportion of trials, observers will respond independently of 
%the stimulus level. For example, observers might have missed the
%presentation of the stimulus, perhaps due to a sneeze or a momentary lapse
%of attention. On such trials, observers may produce an incorrect response
%even if the stimulus level was so high that they would normally have
%produced a correct response. As a result of these lapses, the psychometric
%function will asymptote to a value that's slightly less than 1 when 
%t_A - t_V is large, and asymptote to a value that's slightly greater than 
%0 when t_A - t_V is small.
lapse         = 0.05;

%Define the cumulative Gaussian
P_tilde       = @(mu, sig,lambda, x) lambda/2 + (1-lambda).*normcdf(x, mu, sig);

%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%pass the variables we've defined into the function P_tilde to compute the
%probability of reporting 'V-first'
P_reportV_1st = P_tilde(mu_delta_t, sigma_deltaT, lapse, t_diff);

%For simulating data, here we'll try two methods, which will give the same
%results.
%------------------------ Method 1: using for loops------------------------
randNum       = NaN(len_deltaT, numTrials);
numT_V1st     = NaN(1, len_deltaT); 
sim_prob_V1st = NaN(1, len_deltaT); 
for i = 1:len_deltaT %for each SOA 
    %we first generate random numbers (size = 1 x numTrials) from the 
    %standard uniform distribution
    randNum(i,:)     = rand(1, numTrials);
    
    %get logicals (booleans) for whether the random numbers are smaller
    %than the predicted probability at stimulus level i
    %1: 'V-first' response
    %0: 'V-second' or 'A-first' response
    bool_V1st        = randNum(i,:) < P_reportV_1st(i);
    
    %compute the number of simulated 'V-first' responses
    numT_V1st(i)     = sum(bool_V1st);
    
    %compute the probability of 'V-first' responses
    sim_prob_V1st(i) = numT_V1st(i)/numTrials;
end

%------------------------ Method 2: using matrices ------------------------
%we first replicate the vector P_reportV_1st to match the size of the
%matrix randNum
P_reportV_1st_rep = repmat(P_reportV_1st',[1,numTrials]);

%then we get logicals (booleans) for whether the random numbers are smaller
%than the predicted probability at all stimulus levels at the same time
bool_V1st_        = randNum < P_reportV_1st_rep;

%compute the number of simulated 'V-first' responses (be careful with the
%dimension you choose for summation)
numT_V1st_        = sum(bool_V1st_,2);

%compute the probability of 'V-first' responses at all stimulus levels at 
%the same time
sim_prob_V1st_    = numT_V1st_'/numTrials;


%if your calculation is correct, then sim_prob_V1sit_ should be the same as
%sim_prob_V1st. Check:
if isequal(round(sim_prob_V1st,4), round(sim_prob_V1st_,4)); disp('Check!'); end
%--------------------------------------------------------------------------

%% Exercise 3. continued
%Now let's visualize the psychometric curve and simulated data. If the 
%code you write above is correct, then it should be very close to the curve.

%Hard to tell? Try increase numTrials from 20 to 1000.
%(The code below does not require changes)

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

%% Exercise 4: Fitting a psychometric curve to the fake data
%Now given the simulated fake data, let's try fitting a psychometric
%function to the data by maximizing the log likelihood. 

%For simplicity, let's say the only unknown parameter is mu, and we know
%sigma and the lapse rate.
LogL_mu_hyp = @(p) sum(numT_V1st.*log(P_tilde(p, sigma_deltaT, lapse, t_diff))) + ...
    sum((numTrials-numT_V1st).*log(1 - P_tilde(p, sigma_deltaT, lapse, t_diff)));

%For fitting a psychometric curve, here we'll try two methods, which will 
%give similar results.
%----------------------- Method 1: using grid search ---------------------- 
%specify all the mu you want to test and compute its likelihood
mu_hyp             = -60:1:150;        
%use arrayfun to compute the log likelihood of every single hypothesized mu
LogL_mu            = arrayfun(@(idx) LogL_mu_hyp(mu_hyp(idx)), 1:length(mu_hyp));
%find the mu that corresponds to the greatest log likelihood
[max_val, max_idx] = max(LogL_mu);
mu_hat             = mu_hyp(max_idx);

%let's plot the log likelihood along with the best-fitting mu and the true
%mu we used to simulated the data
figure
h1 = plot(mu_hyp, LogL_mu, 'lineWidth', 3); hold on; 
scatter(mu_hyp(max_idx), max_val, 180, 'r*');
h2 = plot([mu_hat, mu_hat], [min(LogL_mu), max_val],'r-');
h3 = plot([mu_delta_t, mu_delta_t], [min(LogL_mu), max_val], 'k--'); hold off
xlabel(['Hypothesized ', '$\mu$'],'Interpreter','latex');
ylabel('$\log L(\mu|data)$','Interpreter','latex'); box off; grid on
yticks(round(linspace(min(LogL_mu), max_val, 3),1)); 
legend([h1,h2,h3],{'$\log L(\mu|data)$', '$\hat{\mu}$', ['True ', '$\mu$']},...
    'Interpreter','latex');
set(gca,'FontSize',15);

%----------------------- Method 2: using fmincon -------------------------- 
%We want to maximize the log likelihood function. Equivalently, we want to
%minimize the negative log likelihood function (MATLAB likes to minimize
%instead of maximize).
nLogL_mu_hyp = @(p) -sum(numT_V1st.*log(P_tilde(p, sigma_deltaT, lapse, t_diff))) - ...
                   sum((numTrials-numT_V1st).*log(1 - P_tilde(p, sigma_deltaT,...
                   lapse, t_diff)));

%To find the best-fitting mu that minimizes the negative log likelihood, 
%we will use fmincon.m in MATLAB. To use this function, we need to define 
%lower and upper bounds for each parameter (i.e., search space) as well as 
%an initial point for MATLAB to start searching. 
lb      = -60; 
ub      = 150;
init    = rand*(ub-lb) + lb;
%You can also define how many times you want MATLAB to search
options = optimoptions(@fmincon,'MaxIterations',1e5,'Display','off');

%fmincon returns best-fitting parameters that minimize the cost function as
%well as the corresponding value for the cost function (in this case, the
%negative log likelihood). If you are curious what you can put in those
%empty brackets, type help fmincon to find out.
[mu_hat_, min_val] = fmincon(nLogL_mu_hyp, init,[],[],[],[],lb,ub,[],options);   

%display the best-fitting parameter and check if the answer is very similar
%to the one you got from the grid search
disp(mu_hat_);
if abs(mu_hat - mu_hat_) < 1; disp('Checked!'); end
disp(min_val);

%plot the fitted curve
figure
plot(t_diff, P_reportV_1st, 'Color',[13, 183, 200]./255, 'lineWidth', 3,...
    'lineStyle','--'); hold on;
plot(t_diff, P_tilde(mu_hat_, sigma_deltaT, lapse, t_diff), 'k-', 'lineWidth', 1); hold on;
scatter(t_diff, sim_prob_V1st', 300,'filled', 'MarkerFaceColor',...
    0.5.*[13, 183, 200]./255, 'MarkerEdgeAlpha', 0, 'MarkerFaceAlpha',0.5); hold on
xlim([-400, 400]); ylim([0,1]); box off; 
xlabel(['$t_A - t_V$','(ms)'],'Interpreter','latex'); 
ylabel('Probability of reporting ''V-first'''); xticks(t_diff(1:2:end)); 
yticks([0,0.5,1]); legend({'True psychometric function',...
    'Best-fitting psychometric function', 'Simulated data'},...
    'Location','northwest'); legend boxoff
set(gca,'Fontsize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.5]);

%Note that the tolerance is set to be pretty big because the the bin we use
%for mu_hyp is big. If you want to try finner grids 
%(e.g., mu_hyp = -60:0.01:150), go ahead! Just keep in mind that the finer
%the grids are, the longer it's gonna take for you to compute all the log
%likelihoods.

%% Exercise 4. continued
%Now let's assume sigma_deltaT is also unknown, so we have two free
%parameters. Again, we'll try two methods, which will give similar results.
sigma_hyp = 10:1:160;

%YOUR CODE STARTS HERE
%--------------------------------------------------------------------------
%----------------------- Method 1: using grid search ---------------------- 
%we have to first define a new function that takes two free parameters and
%computes the negative log likelihood of the parameters
nLogL_mu_sigma_hyp = @(p) -sum(numT_V1st.*log(P_tilde(p(1), p(2), lapse, t_diff))) - ...
    sum((numTrials-numT_V1st).*log(1 - P_tilde(p(1), p(2), lapse, t_diff)));

%for each combination of mu_hyp and sigma_hyp, call the function 
%nLogL_mu_sigma_hyp for computing the negative log likelihood
nLogL_hyp_mu_sig = NaN(length(mu_hyp), length(sigma_hyp));
for i = 1:length(mu_hyp)
    for j = 1:length(sigma_hyp)
        nLogL_hyp_mu_sig(i,j) = nLogL_mu_sigma_hyp([mu_hyp(i), sigma_hyp(j), lapse]);
    end
end

%find the value that minimizes the negative log likelihood
%Hint: you'll find function ind2sub.m useful
[min_val, min_idx] = min(nLogL_hyp_mu_sig(:));
[row, col]         = ind2sub(size(nLogL_hyp_mu_sig), min_idx);
mu_hat             = mu_hyp(row);
sigma_hat          = sigma_hyp(col);

%----------------------- Method 2: using fmincon -------------------------- 
%Again, let's first define the upper, lower bounds and an initialization
%for MATLAB to start searching
lb      = [-100, 10]; 
ub      = [150, 200];
init    = rand(1,length(lb)).*(ub-lb) + lb;
[estP, min_NLL] = fmincon(nLogL_mu_sigma_hyp, init,[],[],[],[],lb,ub,[],options);   
%display the best-fitting parameters
disp(estP);

%plot the log likelihood as a function of mu_hyp and sigma_hyp using
%surf.m along with the best-fitting parameters found using grid search
%or fmincon.m. 
[MU_hyp, SIGMA_hyp] = meshgrid(mu_hyp, sigma_hyp);
figure
surf(MU_hyp, SIGMA_hyp, nLogL_hyp_mu_sig','FaceAlpha', 0.2, 'EdgeColor','none'); hold on
plot3(mu_hat, sigma_hat,min_val, 'r*'); 
plot3([mu_delta_t,mu_delta_t], [sigma_deltaT, sigma_deltaT],...
    [min(nLogL_hyp_mu_sig(:))-10, max(nLogL_hyp_mu_sig(:))+10],'k-'); 
plot3(mu_delta_t, sigma_deltaT, min(nLogL_hyp_mu_sig(:))-10, 'k+'); hold off
xlabel(['Hypothesized ', '$\mu$'],'Interpreter','latex');
ylabel(['Hypothesized ', '$\sigma$'],'Interpreter','latex');
zlabel('$-\log L(\mu,\sigma|data)$','Interpreter','latex'); box off; 
zlim([min(nLogL_hyp_mu_sig(:))-10, max(nLogL_hyp_mu_sig(:))+10]);
set(gca,'FontSize',15);
%--------------------------------------------------------------------------

%plot the fitted curve
figure
plot(t_diff, P_reportV_1st, 'Color',[13, 183, 200]./255, 'lineWidth', 3,...
    'lineStyle','--'); hold on;
plot(t_diff, P_tilde(estP(1), estP(2), lapse, t_diff), 'k-', 'lineWidth', 1); hold on;
scatter(t_diff, sim_prob_V1st', 300,'filled', 'MarkerFaceColor',...
    0.5.*[13, 183, 200]./255, 'MarkerEdgeAlpha', 0, 'MarkerFaceAlpha',0.5); hold on
xlim([-400, 400]); ylim([0,1]); box off; 
xlabel(['$t_A - t_V$','(ms)'],'Interpreter','latex'); 
ylabel('Probability of reporting ''V-first'''); xticks(t_diff(1:2:end)); 
yticks([0,0.5,1]); legend({'True psychometric function',...
    'Best-fitting psychometric function', 'Simulated data'},...
    'Location','northwest'); legend boxoff
set(gca,'Fontsize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.4, 0.5]);

%% Exercise 5: Bootstrapping
%Bootstrap uses random sampling with replacement (i.e., say that you have
%[1,2,3,4,5] in a magic bag. You randomly draw one number from the bag
%each time, and then put it back). This means as you repeatedly sample from
%the bag, each number can be selected more/less than once. Bootstrap is
%often used when you want to get a confidence interval on estimated 
%parameters. 

% YOUR CODE START HERE
%--------------------------------------------------------------------------
%let's bootstrap 1000 times
numBtst   = 1e3;
estP_btst = NaN(numBtst, 2);
minNLL    = NaN(numBtst, 1);
for i = 1:numBtst
    %disp(i)
    %fill in the function bootstrap
    [~, nT_V1st_btst] = bootstrap_solutions(t_diff, bool_V1st_, numTrials);
    
    %define nLL function given each bootstrapped dataset
    nLL = @(p) -sum(nT_V1st_btst*log(P_tilde(p(1), p(2), lapse, t_diff))') -...
            sum((numTrials - nT_V1st_btst)*log(1-P_tilde(p(1), p(2), lapse, t_diff))');
        
    %fit a psychometric curve to each bootstrapped dataset
    [estP_btst(i,:), minNLL(i)] = fmincon(nLL,init,[],[],[],[],lb,ub,[],options);  
end

%find 95% confidence interval
%first sort the vector in an ascending order
mu_sorted    = sort(estP_btst(:,1)); 
CI_ub_mu     = mu_sorted(ceil(numBtst*0.975)); %upper bound
CI_lb_mu     = mu_sorted(floor(numBtst*0.025));%lower bound

%do the same for sigma
sigma_sorted = sort(estP_btst(:,2));
CI_ub_sigma  = sigma_sorted(ceil(numBtst*0.975));
CI_lb_sigma  = sigma_sorted(floor(numBtst*0.025));

%--------------------------------------------------------------------------

%% Exercise 5. continued
%Plot the best-fitting parameters given each bootstrapped dataset
%No changes are needed for this section

figure
subplot(1,2,1)
%estimated mu by 1000 bootstrapped datasets
h1= histogram(estP_btst(:,1),'FaceColor', 'r', 'FaceAlpha', 0.3,'EdgeColor','r'); hold on
%estimated mu by the orignal dataset
h2=plot([estP(1), estP(1)], [0, numBtst*0.3],'r-', 'lineWidth',3); hold on;
%the lower bound for the 95% bootstrapped confidence interval
h3=plot([CI_lb_mu, CI_lb_mu], [0, numBtst*0.3],'r--', 'lineWidth',3); hold on;
%the upper bound for the 95% bootstrapped confidence interval
plot([CI_ub_mu, CI_ub_mu], [0, numBtst*0.3],'r--', 'lineWidth',3); hold on;
%the value of mu we used to generate the fake data
h4=plot([mu_delta_t, mu_delta_t], [0, numBtst*0.3],'k--', 'lineWidth',3); hold off
xlim([min(estP_btst(:,1)-5), max(estP_btst(:,1)+5)]); ylim([0, numBtst*0.3]); 
xlabel('$\mu$', 'Interpreter', 'latex'); ylabel('Counts');box off;
legend([h1,h2,h3,h4],{'estimates by bootstrapped dataset', ...
    'estimates by the orignal dataset', '95% bootstrap confidence interval',...
    'true value used for generating the data'},'Location', 'northwest');
set(gca,'FontSize',15);

%this subplot is for sigma
subplot(1,2,2)
histogram(estP_btst(:,2),'FaceColor', 'b', 'FaceAlpha', 0.3,'EdgeColor','b'); hold on
plot([estP(2), estP(2)], [0, numBtst*0.3],'b-', 'lineWidth',3); hold on
plot([CI_lb_sigma, CI_lb_sigma], [0, numBtst*0.3],'b--', 'lineWidth',3); hold on;
plot([CI_ub_sigma, CI_ub_sigma], [0, numBtst*0.3],'b--', 'lineWidth',3); hold on;
plot([sigma_deltaT, sigma_deltaT], [0, numBtst*0.3],'k--', 'lineWidth',3); hold off
xlim([min(estP_btst(:,2)-5), max(estP_btst(:,2)+5)]); ylim([0, numBtst*0.3]); 
xlabel('$\sigma$', 'Interpreter', 'latex'); ylabel('Counts');box off;
set(gca,'FontSize',15);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.8, 0.55]);

%% Fun demo: SDT
%the mu for noise condition
mu_N_vec  = [3,     3,   3,   3];
%the mu for noise + signal condition
mu_NS_vec = [3.4, 3.8, 4.2, 4.6]; 
%the width of the distribution (assumed to be the same for both conditions)
sigma_vec = 0.55; %or 0.7
%x-axis
x         = 0:0.1:8;
ylim_ub   = 0.1;
d_prime   = GenerateROC(mu_N_vec, mu_NS_vec, sigma_vec, x, ylim_ub);




