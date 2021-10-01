clear; clc; close all;

%%
load('colMatch.mat')

light = rand(31, 1);
primaries = P;
[knobs] = humanColorMatcher(light, primaries);

sum_of_primaries = mean(primaries, 2);
figure(1);
plot(wl, sum_of_primaries, 'k', 'DisplayName', 'primaries', ...
    'LineWidth', 2)
hold on;
plot(wl, light, 'r', 'DisplayName', 'test light', ...
    'LineWidth', 2)
xlabel('Wavelength (nm)')
ylabel('Intensity (AU)')
title('Comparing test light and primaries spectra')
legend('location','eastoutside')

%%
% The spectra for the primaries is obtained by summing the primaries which
% are linearly independent from each other. The test light, on the other
% hand, can be any light which doesn't necessarily have to be a linear
% combination of the primaries. The human color matching experiment
% involves a linear transformation of the primaries from a higher
% dimensional space, in this case, 31 dimensions, onto a lower dimensional
% space, i.e. 3 dimensions owing to the three different type of cone
% responses. As a result, the perceived light is not a linear combination
% of the primaries but instead related to the cone responses for the three
% primaries and the test light. As long as, two lights can elicit similar
% cone resposes, they will be appear identical to humans despite being
% different spectrally.