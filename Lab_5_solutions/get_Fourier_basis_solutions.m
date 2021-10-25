function Fourier_basis= get_Fourier_basis_solutions(N)
% In the slides, each basis is organized by columns. In this function, we
% do it slightly differently. We store all the basis by rows, so the output
% is actually F transposed.

% your code here

Fourier_basis = nan(N);
n = 0: N-1;
for k = n
    if k == 0
        f = ones(1,N);  
    elseif rem(k,2) == 1
        freq = (k + 1) / 2;
        f = cos(2 * pi * freq * n / N);
    elseif rem(k, 2) == 0
        freq = k / 2;
        f = sin(2 * pi * freq * n / N);
    end
    Fourier_basis(k+1,:) =f;
end
end
