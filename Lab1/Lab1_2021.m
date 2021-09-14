clear; clc; close all;

%% Math Tools Lab 1: Matlab 1
% Percent symbols are used to comment. Double %s create a section, and you
% can run sections separately instead of the entire script

%% Section 1: Review of matlab basics

% This section reviews some core concepts that were covered in the bootcamp

% Basic operations in matlab include:

% Addition: +, subtraction: -, multiplication: *, division: /, ?: sqrt(), 
% raising a number to a power: ^, exponential base e: exp(), logarithm base: e log(),
% absolute value: abs()
% ex. 5+2^2 (ans = 9)
% You can try these out in the command window! 


%% Variables

% Variables are used to store information (in the form of a number, or multiple
% numbers, such as vectors or matrices,etc.) to be referenced and manipulated in a
% computer program. 

% Variables label data with a descriptive name, so our
% programs can be understood more clearly by the reader and ourselves.  

% To create a new variable, enter the variable name in the command window, 
% followed by an equal sign (=) and the value you want to assign to the
% variable. We can then apply the operations we learned to those variables.

%  Try creating a variable in the command window! You can see the variable
% you created in the workspace. You can also use the commands "who" and "whos" in the
% command window to see the variables in your current workspace

% ex. 
a = 2
b = 5
a + b

electrical_bill = 60
gas_bill = 30
total_utilities = electrical_bill + gas_bill

% According to good coding practice, your variable names should be long and
% descriptive, so when you read through it you know exactly what each
% variable represents. (See example above with "electrical_bill".) However,
% a potential downside of this is the need to meticously type out their
% names without typos---what do we do about this?

% The solution: use the "tab" key to autocomplete variable names. This
% saves time while guaranteeing you avoid typos in variable names. Any
% variable that already exists and is compatible with what you've typed so
% far will appear as an option.

% try it! start typing "electrical_bill" and press tab to autocomplete
electrical_bill
% Matlab lets you assign many types of data to variables! Numbers, strings,
% and arrays can all be given variable names

% try it: create a cell array containing your shopping list

grocery_list = {'celery', 'raisins', 'peanut butter'}


%% Scripts
% A script is a file that contains multiple sequential lines of MATLAB
% commands and function calls. You can run a script by typing  run(script name) at the command line. 

% Right now we are in the Scripts section. To create sections, start a line
% with %% followed by a space. Then you can run sections of the script without running the entire script 
% keyboard shortcut for running a section: command(?) + enter on Mac or fn+F9 on Windows

% Write a script to generate two variables.
% What do you notice in the command window after you have created your
% variables, and run the script or script section?

% Use a semicolon (;)  in order to suppress the output. This will become
% very useful when you have lots of calculations in a script or a large variable 

% note: 
% to clear the command window: clc
% to clear everything in the workspace: clear
% everything except some variable x: clear -except x
% to close open windows from plotting: close all

% It's good housekeeping to begin scripts by closing all open figure, and
% clearing the workspace and command line 

%% Vectors 

% A vector is an ordered list of numbers
% There are column vectors and row vectors, which matters for operations!
% A row vector is written in matlab like so: v = [1 2]
% A column vector like this: u = [1; 2] 
% A row vector can be converted into a column vector (or vice versa) with
% an apostrophe like this : u = v'

% For scalar multiplication, we can use the same operator for
% multiplication (*)
% eg. a=2; a*v

% To multiply two vectors, we have to specifically use element wise
% multiplucation
% e.g. u.*v

% Sometimes you will need to generate useful vectors with a set length,
% number of elements, or both. Two convienient ways of creating evenly
% spaced vectors are shown below:

a = linspace(1,100)

b = 1:100

% Both functions create a linearly spaced vector between the endpoints
% given to them. However, they also allow you to specify the spacing of the
% vector:

a = linspace(1,100,10)

b = 1:10:100

% What is the difference between these two outputs? How are they generated?
% You might also want to generate a vector of a single value, as below:

ones(1,10)

% notice what this output is saved as in the workspace. As with other pre-assigned variables
% such as i and j, it will save you future headaches if you avoid creating
% variables with this name

% NaN(x,y) ("not a number"), and zeros (x,y) are similar functions that you can experiment with.
% The functions rand, randn, and randi allow you to generate vectors and
% matricies of random numbers

% You can retrieve elements of the vector via indexing. Matlab is 1-indexed 
% so the first element is a(1), the second is a(2). The last element is 
% a(end), and second to last is a(end-1). You can specify a subset of the
% vector with a vector of indices, e.g. a(1:5)

% To index into a matrix ( a "stack" of vectors, eg. C = randn(5) or C =
% [a;b]), specify both a row and a column (e.g. C(1,3)). If you'd like to
% pull out an entire row or column, specifcy the row you want and 
% use a colon to denote all columns (or vice versa -- e.g. C(2,:) returns
% the second row of C)
% Try generating a matrix of random values to practice indexing.
C = rand(4, 3);

% You can also use indexing to change the components in your matrix. Set
% the value of the second row, first column in your matrix to be zero using
% the syntax Matrix(m,n) = 0

% code here
C(4, 2) = 0
% You can clear values from a vector or matrix by setting them equal to []
% Try clearing the first row of your matrix this way.

% code here
C(1, :) = []
%you can also index into a matrix with a single value (e.g. C(5)). This
%index the matrix as a "folded" column vector -- index 1:m (# of rows)
%correspond to the first column, top to bottom, and m+1:2m correspond to
%the second, and so on. Try this method on the matrix you created in the
%last step.

% What happens if you delete a single value using this indexing system
% (i.e. C(5)=[])? 

% Why does this happen? Matab hates holes in matrices.

% Vector addition: write code to create two vector variables, d and e that are the 
% same dimension and orientation. Save their sum as f, and then run the section so 
% that only f displays in the command window. (reminder: use semicolons to
% suppress output)

% code here

d = rand(4, 1); e = rand(4, 1);
f = d + e

% Scalar multiplication
% Multiplying a vector by a scalar "stretches" the vector out. Define a
% vector and a scalar and multiply them. Does your result make sense 

% code here
alpha = 4; alpha * f

% Vector multiplication: 
% You can multiply individual elements of two vectors together, or perform 
% vector multiplication (which we will cover in more detail in class next week).
% To perform element wise multiplication of two vectors, use .* ( e.g. d.*e)

% Multiplying a column vector (u) by a row vector (v) results in a matrix of values
% where each row contains the scalar multiple of the row vector with
% an element of the column vector (e.g. u(1)*v, u(2)*v,...). Try
% multiplying a column vector by a row vector (e.g. [1; 2]*[3,4])

% code here
[1; 2]*[3,4]

% what happens if you multiply a row vector by a column vector? Do you know
% what this is called? 

[1, 2] * [3; 4]
% inner product of vectors or dot product if its two dimensions?

%% Exercize 1 : visualizing vectors with plotting

% Plotting is a great tool to build an intuitive understanding of the math
% in math tools.

% To create a new figure, you can call 'figure' to initialize a new figure
% window. You can also assign a handle to your new figure:

%fig = figure

% In the command line, you'll see the attributes of the figure window.

% There are many plotting functions built in to matlab. 
% The function plot() in Matlab takes in x-values, y-values, and display
% specifications.  
% To create a line plot, first set up the figure by typing "figure" to open
% a figure. On the next line we can then call the function: plot(x-values,y-values, LineSpec) 

% You can also specify properties of the plot, eg:
% * line color: preset matlab colors (red: 'r', green: 'g', blue: 'b') or
% 'Color' followed by the rgb values, eg red = [1 0 0]. Call 'uisetcolor'
% in the command window for some help picking your dream color scheme

% * line style: dashed line '--' , dotted line '.' eg. to make a dashed red
% line: '--r'
% *line width: plot(x,y, 'LineWidth', 2)

% note: you can also get help with matlab functions by typing "help
% function" into the command window. For example try "help plot"

% Try plotting the vectors v = [1; 2] in blue and  u = [3; 1] in red using plot()
% To see both vectors, on the same axes, specify 'hold on':

fig = figure;
v = [1 2]; u = [3 1];
plot([0 v(1)], [0 v(2)], 'b', 'DisplayName', 'v');
hold on;
plot([0 u(1)], [0 u(2)], 'r', 'DisplayName', 'u');


% Now, show vector addition by plotting v+u as a dashed line. To
% better illustrate the addition, try plotting u as a vector that
% originates at the endpoint of v

w = u + v;
plot([v(1) w(1)], [v(2), w(2)], 'r--', 'DisplayName', 'u_{translated}')
plot([0 w(1)], [0 w(2)], 'g--', 'LineWidth', 2, 'DisplayName', 'w')

% What is the vector length (L2 norm) of v+u? (hint: to square elements of a vector, use .^)

w_norm = L2_norm(w)
% What about the "city block" length (L1 norm)? Calculate the L1 and L2
% norms of v+u and print them to the command window using fprintf, eg:
% fprintf( 'your text here %.2f\n', your L1 norm), which will print the
% norm to two decimal places ( the %.2f)  

w_L1 = L2_norm(u) + L2_norm(v);
fprintf('The L1 norm of w is %.2f\n', w_L1);

% How can you create a unit vector (a vector with length 1) in the
% direction of v? On your existing axes, plot the unit vectors in the
% direction of v and u, with line width of 2. 

u_hat = u/L2_norm(u);
v_hat = v/L2_norm(v);
plot([0, u_hat(1)], [0, u_hat(2)], 'r', 'LineWidth', 2, 'DisplayName', 'u_{hat}');
plot([0, v_hat(1)], [0, v_hat(2)], 'b', 'LineWidth', 2, 'DisplayName', 'v_{hat}');

% Convince yourself that the unit vectors you calculated have a length of 1
% by plotting a circle over your vectors.

r = 1; % radius of the circle 
x = linspace(-r,r,1000);
y = sqrt(r^2 - x.^2);
plot([x, flip(x)],[y,-y], 'c', 'DisplayName', 'unit circle')
axis equal % 


% You can change the axis limits with xlim([min, max]), ylim([min max])
xlim([0, 4.5]); ylim([0, 3.5]);
% Add axis labels with xlabel('your label'), ylabel('your label')
xlabel('x'); ylabel('y');

% Add a title using title('your title')!
title('vector addition', 'fontsize', 20);
% You can change the font size of the title by specifying after the title
% eg title('your title', 'fontsize', 20)

% Now add a legend:
% legend('first vector', 'second vector', ...etc) note: this will label data
% in the order you plotted them!
% OR You can also add 'DisplayName', 'name here' within the plot function:
% plot(x,y, LineSpec, 'DisplayName', 'name here') and then call "legend"
% When I plot a legend, it always shows up in the worst possible place.
% To avoid this, you can specify a location when you generate the legend:
% legend(__,'Location',lcn), e.g. legend(__,'Location','BestOutside'). You
% can also give the legend a handle (l = legend(...)) and use this handle
% to change the Location attribute, l.Location (caps are important here) to
% something that works for your figure
legend('Location', 'BestOutside');

% Save the plot: you can either just get the current plot and save (saveas(gca,
% filename.filetype) or save using the handle we assigned the figure (
% saveas(fig, 'filename.filetype')). These save the figure in the current
% folder.
saveas(fig, 'vector_addition.fig')
% This will save the figure in the current folder.
% note: to close a figure type "close" into the command window
% type "close all" to close all figures

%% 
% Now try plotting higher dimension vectors in a figure with multiple
% subplots!

% The function subplot(m, n, p) creates a m-by-n grid of axes that you can
% plot on, and specifies the p-th axis for the current plot. You can also
% plot on multiple axes to suit your aesthetic sensibilities: e.g.
% subplot(2,3,1:3) creates a 2x3 grid of axes but specifies subplots 1-3
% (the top row) as your current axis. 

% Create a figure with two subpots. Create two vectors with three elements
% and plot them in the first subplot using plot3(x,y,z) (type help plot3 in
% the command window if you are confused about the syntax). Tip: add
% rotate3d on to your plotting code to allow 3D panning 
fig1 = subplot(1, 2, 1);
a = 3*rand(3, 1); b = 4*rand(3, 1);
plot3([0,a(1)], [0,a(2)], [0,a(3)], 'r', 'DisplayName', 'v_1')
hold on;
plot3([0,b(1)], [0,b(2)], [0,b(3)], 'b', 'DisplayName', 'v_2')
xlabel('x'); ylabel('y'); zlabel('z');
title('3d vectors', 'fontsize', 14);
legend();
% In the second subplot, create a vector with ten elements and plot it using stem(vector).
% Then calcuate the unit vector, and plot it using stem()

% Give both subplots axis labels and titles.
% hint: to plot multiple objects on the same axes, specify 'hold on' after you create the subplot
fig2 = subplot(1, 2, 2);

vector = rand(10, 1);
stem(vector, 'filled')
xlim([0, 11])
xlabel('vector counts'); ylabel('vector coefficients');
title('Stem plot', 'fontsize', 14);
% Depending on how you chose to orient your subplots, you may find that one
% of your graphs is squished in a way that makes it harder to read. Try
% playing with different layouts to get something that looks nice. 

saveas(gca, 'multi_dimensional_vectors.fig')

%% Exercize 2: other important plots
% Other types of plots include: bar(), hist(), scatter(), stem(), and many
% more! Matlab has extensive documentation about its plotting functions,
% which you can access at https://www.mathworks.com/help/matlab/creating_plots/types-of-matlab-plots.html

% In Math Tools, you will make a lot of histograms and scatter plots. Let's
% practice using these functions to visualize data.

% First, generate two vectors with 100 elements, one using the randn
% function and one using rand. We will treat these as the x and y values of
% a data set.
x = randn(100, 1); y = rand(100, 1);

% Create a figure with three subplots. In the first, plot the simulated
% data using scatter(x,y). Scatter is a nice function for scatter plots
% because you can easily change the size and color of data points with the
% syntax: scatter(X,Y,Size,Color). Type help scatter to read more about
% this function
fig1 = subplot(1, 3, 1);
scatter(x, y, 10, 'r');
xlabel('x'); ylabel('y');
title('Scatter plot')'

% in subplot 2 create a histogram of the x values using the histogram()
% function. You can either let the function automatically specify bins, or
% set the bins yourself (if you have a prefered bin size or number) by
% specifying a scalar number of bins (e.g. histogram(x,binNumb)) or by
% specifying a vector of bin edges (e.g. histogram(x,binEdges) where
% binEdges = -2:.1:2, for example)
fig2 = subplot(1, 3, 2);
histogram(x, 10)
title('x histogram')
ylabel('Frequency')
% Plot the x values with 10 bins

% Repeat the process in subplot 3 with the y values. What do you notice is
% different about these two histograms? You will see (and make!) many examples of
% similar histograms in future sections. 
fig3 = subplot(1, 3, 3);
histogram(y, 10)
title('y histogram')
ylabel('Frequency')
saveas('other_plots.fig')
