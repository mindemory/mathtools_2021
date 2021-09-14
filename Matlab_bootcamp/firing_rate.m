function [fr_trials, fr] = firing_rate(raster)
    [n, m] = size(raster);
    fr_trials = sum(raster, 2)*1000/m;
fr = sum(fr_trials, 1)/n;
end


