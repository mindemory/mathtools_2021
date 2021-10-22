function kernel = get_gaussian_kernel(sigma, t_range)
    if nargin < 2; t_range = 20;end
    if nargin < 1; sigma = 0.8; end
        
    t = -t_range:1:t_range;
    kernel = exp(-t.^2 /(2*sigma^2));
    kernel = kernel./sum(kernel);
end