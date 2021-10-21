clear; clc; close all;

%%
impulse_pos = [1, 2, 4, 8, 64];
impulse = cell(length(impulse_pos), 1);
output = cell(length(impulse_pos), 3);
N = 64;
for i = 1:length(impulse_pos)
    figure();
    impulse{i} = zeros(N, 1);
    impulse{i}(impulse_pos(i)) = 1;
    for j = 1:3
        fname = str2func(sprintf('unknownSystem%d', j));   
        output{i, j} = fname(impulse{i});
        plot(output{i, j}, 'DisplayName', ['Output: System ', num2str(j)], ...
            'LineWidth', 2)
        hold on;
    end
    plot(impulse{i}, 'DisplayName', 'Impulse', 'LineWidth', 2);
    legend()
    title(['Impulse at position ', num2str(impulse_pos(i))])
    xlabel('Position')
    ylabel('Spike count')
end

comb_impulse = impulse{2} + impulse{3};
comb_output = cell(1, 3);
for j = 1:3
    fname = str2func(sprintf('unknownSystem%d', j));   
    comb_output{1, j} = fname(comb_impulse);
    %figure()
    %plot(output{2, j} + output{3, j}, 'DisplayName', 'Combined Output', ...
    %    'LineWidth', 1)
    %hold on;
    %plot(comb_output{1, j}, 'DisplayName', 'Combined Input', 'LineWidth', 2)
    lin_check = abs(sum(comb_output{1, j} - (output{2, j} + output{3, j}), 1)) < 0.01;
    if lin_check
        sprintf('System %d is linear', j)
    else
        sprintf('System %d is nonlinear', j)
    end
end

%% b)
input_freqs = [2, 4, 8, 16];
output_sinusoid = cell(1, 3);
for i = 1:length(input_freqs)
    angs = 0:input_freqs(i)*pi/N:input_freqs(i)*pi;
    angs_with_phase = angs + rand;
    input_sinusoid = 1 + sin(angs_with_phase(:, 1:end-1));
    size(input_sinusoid')
    for j = 1:3
        fname = str2func(sprintf('unknownSystem%d', j));
        output_sinusoid{1, j} = fname(input_sinusoid');
        size(output_sinusoid{i, j})
        figure()
        plot(angs, input_sinusoid', output_sinusoid{1, j}')
    end
    
    
end
