clear; clc; close all;

%% a)
% Plotting the signal as a function of time and then overlaying the original
% signal on the subsampled version of the signal with every 4th element of
% the signal being sampled.

%%
load('myMeasurements.mat')
figure();
plot(time, sig, 'ko-')
hold on;
time_subsampled = time(4:4:end); % Subsampling time with every 4th timepoint
sig_subsampled = sig(4:4:end); % Subsampling signal with every 4th datapoint
plot(time_subsampled, sig_subsampled, 'r*-')
xlabel('Time (s)')
ylabel('Voltage (V)')
title('EEG Signal')

%%
% The subsampled signal does not provide a good summary of the original
% signal as it ends up choosing some elements in the signal while
% elementing remaining elements. It appears that subsampling operation is
% better at capturing low-frequency signals in the original signal.
%%
% The sampling operation is linear because it can be expressed as a matrix
% multiplication aX = b, where the system X is an identity matrix with some
% of the diagonal elements replaced with 0s, a is the original signal and 
% b is the subsampled signal.
%%
% If the input a is shifted, the 1s in the diagonal of X will now be
% multiplied by different input values which will result in a different
% output. Therefore, the subsampling operation is not shift-invariant.

%% b)
% The EEG signal can be transformed into the frequency domain by performing
% fft over the signal. We can also compute the shifted fft by using
% fftshift to center the DC component of the fft. The frequency range goes
% from 0:N-1 and is scaled by the sampling frequency and the length of the
% signal vector N. The shifted frequency scale goes from -N/2 to N/2 - 1
% with same scaling. Plotting the FFT and the shifted FFT, we get:

%%
fs = 120; % sampling frequency
N = length(sig); % length of the signal vector
f = (0:N-1) * fs/N; % creating the frequency vector
fshift = (-N/2: (N/2)-1) * fs/N; % creating a shifted frequency vector
F_sig = fft(sig); % computing the Fourier Transform of signal
F_shift_sig = fftshift(F_sig); % Shifting the Fourier Transform so that DC component is in the middle

figure();
plot(f, abs(F_sig))
xlabel('Frequency (Hz)')
ylabel('Amplitiude')
title('FFT of signal')
figure();
plot(fshift, abs(F_shift_sig))
xlabel('Frequency (Hz)')
ylabel('Amplitiude')
title('FFTshift of signal')

%%
% The strongest signal is obtained at 12 Hz which belongs to the Alpha band
% at 36 Hz which belongs to the Gamma band. I am unsure of the exact source
% of this signal. Here is the thought process that I have went through. At
% first instance, I expected there to be line noise in the signal since its
% an EEG signal. But 36 Hz is too low a frequency for line noise. It could
% just be some neural activity related to the task which should give signal
% in the gamma band of the frequency range. However, this is unlikely to
% give a peak at a single frequency. But then this is synthetic data, so
% maybe that's what it is signifying? Another way to look at it is that 36
% Hz is a second harmonic of 12 Hz. However, harmonics in fft generally
% exhibit lower amplitude than the fundamental. Here the fundamental of
% second harmonic is higher than fundamental (absurd?!), and the first
% harmonic is missing. So I am unsure of the source of the 36 Hz signal
% here.

%% c)
% The bandWiseReconstruct computes the signalPart in a given time-range, for
% a given fftsig within a frequency band. It does so by first computing a
% vector of zeros of length of F_sig and then adding ones at frequencies of
% interest within a given bname. It then takes an element-wise product of
% F_sig with this vector to create an FsigPart which has amplitudes at
% frequencies of interest and is 0 otherwise:
bandnames = ["Delta", "Theta", "Alpha", "Beta", "Merged"]; % array of frequency bands, merged is the entire frequency range
for bname = bandnames
    signalPart = bandWiseReconstruct(time, F_sig, bname); % computing signalPart from F_sig for given frequency bandname
end
%%
% We can see that since the FFT of the signal only has peaks in the alpha
% band and the gamma band, the reconstructed signal from these bands has
% some structure to it whereas for other frequencies, we only see noise
% with amplitude of the signal very close to 0.

%% d)
% The upsampled sigals after downsampling can be computed by creating an
% array of zeros with 1s at positions of downsampling vector and then
% taking an element-wise product of this vetor with the signal. The fft and
% fftshift can then be computed for this upsampled signal.
%%
ds_facts = [2, 3, 4]; % downsampling factors
for fct = ds_facts
    sig_downsamp_pos = zeros(1, length(sig));
    sig_downsamp_pos(fct:fct:end) = 1; % Creating an array of ones at positons where sub-sampled frequencies will be
    sig_upsamp = sig .* sig_downsamp_pos; % Upsampling by computing element-wise product of signal with downsampled positions
    F_sig_upsamp = fft(sig_upsamp); % Computing FFT of upsampled signal
    F_sig_upsamp_shift = fftshift(F_sig_upsamp); % Computing FFTshift of upsampled signal
    
    figure();
    plot(fshift, abs(F_sig_upsamp_shift)); % Plotting amplitudes of shifted FFT
    xlabel('Frequency (Hz)')
    ylabel('Amplitiude')
    title(['FFT of signal sampled at k = ', num2str(fct)])
end

%%
% The downsampling and upsampling happening over here is naive. This
% results in artifacts in the reconstructed signal. The original sampling
% frequency is 120 Hz. Upon downsampling and upsampling, the sampling
% frequency becomes 120/k, where k is the downsampling factor. Hence, the
% Nyquist frequency of the downsampled signal will be 120/2k. For k = 2,
% Nyquist frequency is 120/4 = 40 Hz. The maximum frequency in the original
% signal was 36 Hz which is below Nyquist and hence there is no aliasing.
% However, since the signal is downsampled and upsampled, it creates two
% additional peaks at 60 - 12 = 48 Hz and 60 - 36 = 24 Hz. For k = 3, the
% Nyquist frequency is 120/6 = 20 Hz which is below the maximum frequency
% in the signal resulting in aliasing. Thus we see that the amplitude of
% the first peak in fft is higher than the second. The same happens in the
% case of k = 4. In order to avoid aliasing, signal should be low-pass
% filtered to ensure that there is no component in the downsampled signal
% that is above the Nyquist frequency of the downsampled signal.

%%

%% Functions
function signalPart = bandWiseReconstruct(time, Fsig, bandname)
    % Defining frequency bands and their min and max frequency ranges
    bnames = ["Delta", "Theta", "Alpha", "Beta", "Merged"];
    freqs = [[0, 4, 8, 16, 0]; ...
        [4, 7, 15, 31, 60]];
    band_freqs = freqs(:, bnames == bandname); % Computing the min and max frequency bounds for the given bandname    
    bvec = zeros(1, length(Fsig));
    bvec(band_freqs(1) + 1: band_freqs(2) + 1) = 1; % Creating a vector of zeros with ones at frequencies of interests
    FsigPart = Fsig .* bvec; % Taking an element-wise product of Fsig with bvec to create Fsig with amp values at frequencies of intrest and 0 otherwise
    signalPart = ifft(FsigPart); % Computing inverse FFT of FsigPart
    
    figure()
    plot(time, real(signalPart), 'ro-') % The recomputed signal has imaginary parts which are not of our interest since the signal has only real values
    xlabel('Time (s)')
    ylabel('Voltage (V)')
    title(['Reconstructed EEG Signal at ', bandname])
    
end