clear; clc; close all;

%%
impulse_pos = [1, 2, 4, 8, 64];
impulse = cell(length(impulse_pos), 1);
output = cell(length(impulse_pos), 3);
N = 64;

for i = 1:length(impulse_pos)
    impulse{i} = zeros(N, 1);
    impulse{i}(impulse_pos(i)) = 1;
    figure();
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

%%

%%
sc2 = randi(10);
sc3 = randi(100);
comb_impulse = sc2 * impulse{2} + sc3 * impulse{3};
comb_output = cell(1, 3);
impulse_shift_by = randi(10);
shift_impulse = circshift(impulse{2}, impulse_shift_by);
shift_output = cell(1, 3);

for j = 1:3
    fname = str2func(sprintf('unknownSystem%d', j));   
    comb_output{1, j} = fname(comb_impulse);
    shift_output{1, j} = fname(shift_impulse);
    %figure()
    %plot(output{2, j} + output{3, j}, 'DisplayName', 'Combined Output', ...
    %    'LineWidth', 1)
    %hold on;
    %plot(comb_output{1, j}, 'DisplayName', 'Combined Input', 'LineWidth', 2)
    lin_check = abs(sum(comb_output{1, j} - (sc2 * output{2, j} + sc3 * output{3, j}), 1)) < 0.0001;
    shift_check = abs(sum(shift_output{1, j} - circshift(output{2, j}, impulse_shift_by))) < 0.001;
    if lin_check
        if shift_check
            sprintf('System %d is linear and shift-invariant', j)
        else
            sprintf('System %d is linear but not shift-invariant', j)
        end
    else
        if shift_check
            sprintf('System %d is nonlinear but shift-invariant', j)
        else
            sprintf('System %d is nonlinear and non shift-invariant', j)
        end
    end
    
end

%% b)
input_freqs = [2, 4, 8, 16];
n = (0 : N-1)';

%output_sinusoid = cell(length(input_freqs), 3);
%fname = str2func(sprintf('unknownSystem%d', j));

for i = 1:length(input_freqs)
    phi = rand;
    
    input_sinusoid = 1 + sin(pi * input_freqs(i) * n / N + phi);
    %plot(angs, input_sinusoid);
    %size(angs')
    %size(input_sinusoid')
    output_sinusoid = unknownSystem2(input_sinusoid);
    fft_input = fft(input_sinusoid);
    fft_output = fft(output_sinusoid);
    diff_amp = abs(fft_input) - abs(fft_output)
    %size(output_sinusoid{1, j})
    figure()
    plot(n/N, input_sinusoid, 'DisplayName', 'input'); hold on;
    plot(n/N, output_sinusoid, 'DisplayName', 'output');
    figure()
    %plot([abs(fft_input), abs(fft_output)])
    %plot(diff_amp)
    title(['System 2, input freq ', num2str(input_freqs(i))])
    legend();
end
