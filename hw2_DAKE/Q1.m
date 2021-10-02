clear; clc; close all;

%% a)
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
% space, i.e. 3 dimensions owing to the three different type of cones
% in humans. As a result, the perceived light is not a linear combination
% of the primaries but instead related to the cone responses for the three
% primaries and the test light. As long as, two lights can elicit similar
% cone resposes, they will be appear identical to humans despite being
% different spectrally. One way to think about it is that the human visual
% system projects the wavelengths onto a 3-dimensional space and hence the
% matching of the colors happens in this lower-dimensional space. Hence all
% the sets of wavelengths that can produce the same responses in this
% 3-dimensional space are bound to appear identical.

%% b)
% Let C = Cones, P_{old} = P, P_{new} = eP, k_{old} = l= knobs,
% k_{new} = eknobs
%%
% The old color matching equation is:
%%
% Ct = CP_{old}k_{old}
%%
% And the new color matching equation is:
%%
% Ct = CP_{new}k_{new}
%%
% Since the perception of both the old primaries with the old settings and
% the new primaries with the new stting is the same, we can equate the
% right-hand side of the two equations
%%
% Hence, CP_{old}k_{old} = CP_{new}k_{new}
%%
% Given the invertibility of CP_{new}, we can create a new matrix
% old_to_new as
%%
% old_to_new = inv(CP_{new}) * (CP_{old})
%%
% Therefore, k_{new} = old_to_new * k_{old}
%%
% The old_to_new matrix can be computed by substituting the corresponding
% matrices in the formula:
%%
old_to_new = inv(Cones * eP) * (Cones * P);