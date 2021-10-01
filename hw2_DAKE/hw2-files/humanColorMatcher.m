function knobSettings  = humanColorMatcher(testLight, primaries)

% knobSettings = humanColorMatcher(testLight, primaries)
%
% Simulate a particular human observer in a color-matching
% experiment.
%
% TESTLIGHT should be a 31-dimensional column vector, containing the
% spectral distribution of a test light, with wavelengths sampled from
% 400 to 700 nm in 10nm increments.  Alternatively, it can be a 31-row
% matrix, with each column containing a test light.
%
% PRIMARIES should be a 31-row matrix containing the primary lights
% that the human must mix in order to match the appearance of the test
% light(s).  Normally, this would contain 3 primaries (3 columns), but
% the function will produce appropriate responses for any number of
% primaries.
%
% KNOBSETTINGS is the human response: a vector containing the
% intensities of the primary lights that the human chooses to best
% match the test light.

error('Code has been removed - this is a stub file, intended to provide documentation.  Execute the p-file instead.');

