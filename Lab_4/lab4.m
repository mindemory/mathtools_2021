%% Lab 4: Convolution in 1 and 2-D
load('lab4_files.mat'); 

%% Exercise 1: The mechanics of convolution

% Question 1: When you use the conv or conv2 function in matlab, you are 
% implicitly using zero padding by default. Fill in the function 
% manualPadding that replicates the native conv function, but by manually 
% applying the appropriate zero padding and using the 'valid' mode only.
%
% Bonus: Try replicating conv by building the convolution matrix from
% scratch for each setting, instead of using 'valid'.


%% Question 2: Fill in the skeleton code below, then interpret the resulting
% plots. Specifically, think about what it means for the kernel size to
% increase (in this specific case, and in general). 

x = 0:0.01:2*pi;
y = sin(x) + 0.1*randn(size(x));

figure;
set(gcf, 'Position', [0, 0, 1500, 400]);

subplot(1, 4, 1); 
plot(x, y);
title('Raw Signal');
counter = 2;
for kernel_sz = [5, 10, 100, 600]
    % define a kernel for the convlution that takes a rolling average of
    % the signal

    % ** YOUR CODE STARTS HERE **
    k = ;
    
    % convolve the noisy sine wave y with the kernel using 'same'
    % padding

    % ** YOUR CODE STARTS HERE **
    smoothed_y = ; 
    
    % visualize the results
    subplot(1, 5, counter);
    plot(x, smoothed_y); ylim([-1,1]);
    title(['Averager Kernel Size = ', num2str(kernel_sz)]);
    counter = counter + 1;
end
    

%% Question 3:Interpret the meaning of larger vs. smaller kernel size for 
%convolutions with time series data. (In the general case, the above is one 
%possible example). Is there an "optimal" size for some settings? How would 
%we know? (Philosophical question.) 
%
% ** ANSWER **








%% convolution as matrix multiplication
% Although convolution has a nice formula c_k = r_k * sum_{k'} x_k - k'
%that is NOT at face value a matrix multiplication, it can still be 
%augmented to allow intepretation as a matrix operation. This allows us to 
%make connections with linear algebra we've already learned.

%Let's define our kernel r as a Gaussian window average
r = exp(-(-10:1:10).^2/9);
avg = r./ sqrt(dot(r,r));
figure
plot(avg)

%We want to define a matrix R that implements this convolution on signals 
%of a given dimensionality
N_signal = 1e3;
x = linspace(0, 2*pi, N_signal);
y = sin(x) + randn(1, length(x)).* 0.1; 
figure
plot(x, y)

%%
%Remember, for a normal matrix operation, we can have our system act on 
%each basis function. Once we have those answers recorded, we know how it 
%should act on *anything*.

%Let's do this inefficiently first: apply our kernel r to every single 
%basis vector in N_signal-dimensional space. Store the results as columns 
%in a matrix.

R = [];
e_i = eye(N_signal); %Take the ith row of the identity
for i = 1:N_signal
    %define a column of R and append to our list of R columns
 
    % ** YOUR CODE STARTS HERE **
    r_column = ;
    
    R = ;
end
R_trans = R';

figure
subplot(2,1,1)
imagesc(R_trans); axis square;
%Okay now just zoom in on a chunk of R
%Recognize this?
subplot(2,1,2)
imagesc(R_trans(1:30, 1:30));axis square;

%% question 4
%Why is this method "inefficient"? What property of convolution as a linear 
%operator are we failing to exploit in checking the result on each temporal 
%basis vector?

%At the same time, why might this characterization as a matrix be 
%interesting? What theoretical tools might be be able to lean on? Is there 
%a coordinate system where this matrix might get nice? (These are "teaser" 
%questions more than questions we expect you to be able to answer!)

%% Exercise 2: image convolution
% Convolving a 2-D signal with a kernel compares the similarity of a pieces
% of the input with the kernel (in the dot product sense of similarity).
% The image that the convolution results in can be thought of as a heat-map
% of how similar a part of the input is with the kernel. In computer vision
% this operation is called "feature extraction," with the kernel being the
% "feature."
%
% Fill in the code below to implement the procedure described above for
% features called 2-D "gabor functions," which are often used to model V1
% receptive fields.

clear all; 
load('lab5_files.mat');
% Gabor parameters 
wavelength = 20; % scales size of kernel
theta = pi/4; % changes the orientation of the kernel
x = gabor_fn(theta, wavelength);

% visualize some gabors by messing with the value of theta above
% (note: you can try highlighting and running this section of the code by 
% itself to compare figures)
figure;
set(gcf, 'Position', [100, 100, 500, 500]);
imagesc(x);

% display input image (loaded at top of script)
figure; 
imagesc(feature_input_image); 
title('Feature Extraction Input Image');

% compute model convolutions with kernels at different angles/sizes
% (note the difference between 3*pi/4 and pi/4, at least)
theta = 3*pi/4; % controls orientation
wavelength = 30; % scales the size of the kernel
kernel = gabor_fn(theta, wavelength);
responses = 0; % REPLACE THIS LINE
responses = conv2(feature_input_image, kernel); % ** ANSWER ** 
          
% visualize the results
figure; 
imagesc(abs(responses)); 
title(['Convolution with Theta= ', num2str(theta),...
    ', Wavelength= ', num2str(wavelength)]);
  
% Run the above code fora  variety of choices of theta. 
% What effect does changing the size and orientation of the 
% kernel have on the output. Can you link this to the analogy 
% between V1 cells receptive fields and convolution that was hinted
% at/discussed?


% ** ANSWER ** 












