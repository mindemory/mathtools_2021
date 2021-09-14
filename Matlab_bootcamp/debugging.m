%% Debugging and matrix operations practice

%% 1. Creating the network

% first choose a number of presynaptic cells for each of the fifty neurons
num_presyn = randn(1,50);
num_presyn = num_presyn.*3+15;
num_presyn = round(num_presyn);

% mark the connections in an adjacency matrix
adj_mat = zeros(50);
for i=1:1:50
    presyn = randsample([1:i-1,i+1:50], num_presyn(i)); 
    % immediately upon running the code an error in the above line pops up.
    % before proceeding, we need to fix this.
    
    adj_mat(i,presyn) = 1;
end

% Stop here: add a breakpoint at the next line of code and check if
% the adjacency matrix has roughly the correct amount of 1s.  About how many do we expect and why?

% In the command window, enter sum(adj_mat, 'all')
% Is this similar to the expected number of 1s? If not, try to find the
% error.

%% 2. Creating the first vector of spikes

spiked = zeros(1,50);
spiked(1,randsample(1:50,15)) = 1;

%% 3. Simulate the population
spike_counts = spiked;  % variable to store the number of total spikes per neuron
for t=1:1:10000
    num_pre_spiked = spiked*adj_mat'; % what does the calculation on this line do?
                                     % Is it consistent with what the
                                     % student intended to do?
    spiked = num_pre_spiked >= 7;

    % determine neurons that spiked randomly
    rand_spikes = rand(1,50);
    rand_spikes = (rand_spikes < .1);
    
    spiked = max(spiked, rand_spikes);
    
    % write some code here to check that 'spiked' only has 0s and 1s.  If it
    % has larger numbers, can you find out why and fix it?
    
    spike_counts= spike_counts + spiked;
end

%% 4. plot the histogram of firing rates

figure;
histogram(spike_counts);
xlabel('number of spikes')
ylabel('number of neurons')
title('spike count distribution');