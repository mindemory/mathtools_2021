function F_x_power = get_signal_power(F_x)
    
    %Remove the first and last entries 
    F_x_power = F_x(2:end-1);
    
    %Pair up the sines and cosines of each "middle" frequency and take
    %their squared norms as 2-vectors
    F_x_power = sum(reshape(F_x_power,2,[]).^2);
    
    %Add the first and last entries' absolute values back in
    F_x_power = [F_x(1),  F_x_power ,F_x(end)]; 
    
end