function [r_slc, nT_V1st_slc] = bootstrap_solutions(s_unique, r_org, nT) 
%initialize resampled responses
r_slc = NaN(length(s_unique), nT);

% YOUR CODE START HERE
%--------------------------------------------------------------------------
%for each stimulus location, we resample the responses
for j = 1:length(s_unique)
    %randomly select trial indices (indices are allowed to occur more
    %than once, since we resample with replacement).
    %Hint: you'll find function randi.m useful
    idx        = randi([1 nT],[1 nT]);
    %store the resampled responses
    r_slc(j,:) = r_org(j,idx);
end
%--------------------------------------------------------------------------
%compute the total number of V-first responses given each SOA
nT_V1st_slc = sum(r_slc,2)';