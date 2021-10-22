function Fourier_basis= get_Fourier_basis(N)
% In the slides, each basis is organized by columns. In this function, we
% do it slightly differently. We store all the basis by rows, so the output
% is actually F transposed.

Fourier_basis = nan(N);
n = 0: N-1;
for k = n
    if k == 0
        f = ones(1,N) ;  
    elseif rem(k,2) == 1
        %your code here
    elseif rem(k, 2) == 0
        % your code here
    end
    Fourier_basis(k+1,:) =f;
end
end
