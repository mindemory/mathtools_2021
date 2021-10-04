clear; clc; close all;

%% a)
load('colMatch.mat')

light = rand(31, 1);
primaries = P;
[knobs] = humanColorMatcher(light, primaries);

sum_of_primaries = mean(primaries, 2);
figure(1);
plot(wl, sum_of_primaries, 'k', 'DisplayName', 'primaries', ...
    'LineWidth', 0.5)
hold on;
plot(wl, light, 'r', 'DisplayName', 'test light', ...
    'LineWidth', 0.5)
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
eknobs_pred = old_to_new * knobs
[eknobs_actual] = humanColorMatcher(light, eP)
%%
% As can be seen the predicted knob settings from the old_to_new matrix do
% not match the actual knob settings. This is the case because the new
% primaries are not linearly independent from each other. One way to check
% for this is by creating a plot of the new primaries:
figure(2);
plot(wl, eP)
legend('eP1', 'eP2', 'eP3')
xlabel('Wavelength (nm)')
ylabel('Intensity (AU)')
title('Spectra of new primaries')
%%
% From the plot we can see that the eP3 curve exhibits two peaks that align
% very well with the peaks of eP1 and eP2. Therefore, it appears that eP3
% is a linear combination of eP1 and eP2 vectors (or close to being a
% linear combination). However, the linear dependence is not prominent just
% by visualization.
%% 
% Another way to check for the linear dependence of the three primaries is
% to look at the diagonal elements of the S matrix obtained upon SVD.
% Performing SVD on eP, we get:
[eU, eS, eV] = svd(eP);
diag(eS)
%%
% The final element along the diagonal of S is almost 0. Hence the third
% vector is a linear combination of the other two vectors. Thus, resulting
% in linearly dependent primaries which violate the assumptions of the
% color-matching experiment.

%% c)
% Visualizng the Cones spectral sensitivies
figure(3)
plot(wl, Cones', 'LineWidth', 2)
legend('L', 'M', 'S')
xlabel('Wavelength (nm)')
ylabel('Responsitivity of the cone (AU)')
title('Responsivity curves for the three cones')

%%
% For the old primaries, the resulting light obtained by the combination of
% the primaries with the obtained knob settings from the color matching
% experiment are:
prim_light = P * knobs;
%%
% We have to determine if the responses for the cones from the randomly
% generated test light and the combination of the primaries are the same.
response_test_light = Cones * light
response_prim_light = Cones * prim_light

%%
% As can be seen the responses from the cones for the primary lights with
% the knob settings obtained from the color matching experiment and the
% responses from the cones for the test light are exactly identical. This
% is an informal way of justifying that the cones provide a physical
% explanation of the color matching experiment.

%%
% The color matching experiment can be defined in terms of the C = Cones, t
% = test_light, P = primaries, and k = knob settings. The response of the
% cones to the test lights is the same as the response of the cones to the
% combination of the primaries. The response of the cones to the test light
% is Ct. The combination of primaries with the given knob settings is Pk
% and hence the response of the cones to the combination of the given
% primaries is CPk. Therefore, the color matching experiment can be
% summarized as:
%%
% Ct = CPk
%% 
% Thus the knob settings for the color matching experiment can be obtained
% for any given test light as:
%%
% k = (CP)^-1*C*t
%%
% As shown earlier, the color matching matrix M = (CP)^-1*C and hence the
% result can be summarized as:
%%
% k = Mt
%%
% We can then write the code to compute the knob settings given the
% primaries and test light using the above procedure.
knobs_manual = inv(Cones * P) * Cones * light
%%
% These knob settings match the knob settins obtained from the
% humanColorMatcher function