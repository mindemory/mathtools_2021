function double_frequencies = get_frequencies_solutions(N)
    
    %your code here
    double_frequencies = [0, sort([1:round(N/2)-1, 1:round(N/2)-1]), round(N/2)];
end 
