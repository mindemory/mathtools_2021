function output = manualPadConv(signal, kernel, pad_mode)
    if strcmp(pad_mode, 'valid')
        output = conv(signal, kernel, 'valid');
    elseif strcmp(pad_mode, 'full')
        % pad the input signal with the correct amount of zeros for full
        % ** YOUR CODE STARTS HERE **
        num_pad = ;
        
        padded_signal = ; 
        output = conv(padded_signal,kernel, 'valid');
    elseif strcmp(pad_mode, 'same')
        % NOTE: You only need to replicate the built-in function for 'same'
        % in the case that the length of the kernel is odd. (In the case 
        % of even kernel size, we need to pad asymetrically in order to 
        % replicate, for those curious).
        % ** YOUR CODE STARTS HERE ** 
        if rem(size(kernel,1),2) == 1 %odd number
            num_pad = ;
            padded_signal = ; 
        else %even number
            num_pad_left = ;
            num_pad_right = ;
            padded_signal = ; 
        end
        output = conv(padded_signal,kernel, 'valid');    
    else
        error('unsupported pad mode');
    end
end
