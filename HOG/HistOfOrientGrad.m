%% File Information 
%--------------------------------------------------------------------------
% File Name             : HistOfOrientGrad.m
%
% Description           : This file implements HOG (Histogram of Oriented 
%                         Gradients
%
% References            : https://www.learnopencv.com/histogram-of-oriented
%                        -gradients/
%
% Author(s)             : Aravind D. Chakravarti
%
% Version History       :
% Ver   Name        Change Description
% 1.0   Aravind D   Started with basics. Just a trial
% 1.1   Aravind D   Calculation of gradient magnitudes with respect to bins
%                   is successful.
% 1.2   Aravind D   Kind of visualization added. Testing with SVM or PCA is
%                   pending
%--------------------------------------------------------------------------

% List of To-Dos
% 1.    If a large image is given then program should crop the image with
%       aspect ratio
% 2.    16x16 histogram normalization
% 3.    Calculation of feature vector (Highest priority)
% 4.    Make use of aspect ratio. Espcially for visualization

function [features] = HistOfOrientGrad (I) %#ok<INUSD>
%% Initial Stuff; Delete this when function is done <TO-DO>
clc;
close all;
clear;

%% Inbuilt command HOG for trial
img = imread('cameraman.tif');
%img  = imread('Test_image.jpg');
[~, features] = extractHOGFeatures(img); 
figure;
imshow(img); hold on;
plot(features);
title ('HOG by Built-in Function');

%% Let us start Manual implementation

% For now, read some dummy image
I_raw = double (imread('cameraman.tif'));
%[im_row,im_col] = size(I_raw);
I = applyGaussianFilter(I_raw);

% This point forward, I am sticking here with desription provided by 
% https://www.learnopencv.com/histogram-of-oriented-gradients/

%% STEP - I : Preprocessing
% This step is skipped as of now. Since we are not tracking any object here
% <TO-DO>

%% STEP - II : Calculate the Gradient Images
% We are going to run SOBEL operator on the image to calculate x and y
% gradients of intensities, and angle of gradients. 
[I_Grad, I_Directn] = runSobelOnImage (I);
figure;
imshow (uint8(I_Grad));
title ('Sobel Operated Image');

%% STEP - III : Calculate Histogram of Gradients in 8×8 cells 
% In this step, the image is divided into 8×8 cells and a histogram of 
% gradients is calculated for each 8×8 cells.
% Note: The webpage talks about implementing HOG for RGB image. Hence he
% tells that image patch contains 8x8x3 = 192 pixels. Since we are
% implementing for Grayscale we will have 8x8 = 64 pixels

% Resolution of the image is
I_resolution = size(I,1)*size(I,2);
% We are analyzing patch of 64 pixels hence we need resolution/64 bins rows
% each of columns
bins = double (zeros(I_resolution/64, 9));
bins_index = 0;

for move_hor = 1:8:(size(I,2))
    for move_ver = 1:8:(size(I,1))
        bins_index = bins_index + 1;
        
        % Get the 8x8 patches and directions
        get_8by8_grads = I_Grad(move_hor:(move_hor+7), ...
                                move_ver:(move_ver+7));
        
        get_8by8_direc = I_Directn(move_hor:(move_hor+7), ...
                                move_ver:(move_ver+7));
        
        for temp_i = 1: size(get_8by8_direc(:),2)
            get_grad_dire = get_8by8_direc(temp_i);
            get_grad_mag  = get_8by8_grads(temp_i);
            
            % Function will calcuate the histogram
            [bins_nos, bins_mag] = ...
                            FindBinsAndMag(get_grad_dire, get_grad_mag);
            
            % Where to add those magnitudes?            
            bins(bins_index, bins_nos(1)) = ...
                    bins(bins_index, bins_nos(1)) +  bins_mag(1);
                
            bins(bins_index, bins_nos(2)) = ...
                    bins(bins_index, bins_nos(2)) +  bins_mag(2);    
            
        end
    end
end

%% STEP - IV : 16x16 Histogram normalization
% <To-Do> : Implement this. This is not important as of today!    

%% STEP - V : Calculate the HOG feature vector
% <To-Do> : Calculate feature vector. Ideally we should be returning this
% to calling function
visualizeHOG(bins, I);

end % End of HistOfOrientGrad Function

function [] = visualizeHOG(visualize_bins, Original_Image)
%% Function Info:
%  This function creates sort of visualization of data. Ofcourse useless 
%  for classification problems. Just don't call this function

type = ['o', '+', '*', 'x', 's', '^', 'd', 'p', 'h'];

for_sort = ones(size(Original_Image,1),1);

for tmp_index = 1:size(visualize_bins,1);
    [~, for_sort(tmp_index)] = max(visualize_bins(tmp_index, :));
end
    
mat_for_heat_plot   = reshape(for_sort, [32, 32]);

size_of_plot_mat    = size(mat_for_heat_plot);

figure; 
axis([1 32 1 32]);
hold on;

for temp_index_1 = 1: size_of_plot_mat(1) 
    for temp_index_2 = 1: size_of_plot_mat(2)
        temp = mat_for_heat_plot(temp_index_1, temp_index_2);
        if (temp == 7)
            plot (temp_index_1, temp_index_2, type(temp), 'color', 'r');
        elseif (temp == 6)
            plot (temp_index_1, temp_index_2, type(temp), 'color', 'b');
        else
            plot (temp_index_1, temp_index_2, type(temp), 'color', 'k');
        end
    end
end

hold off;
        
end

function [bin_nos, bin_magnitude] = FindBinsAndMag(angle, magnitude)
%% Function Info:
%  This function creates histogram of oriented gradients

angle_ratio_finder = [0, 20, 40, 60, 80, 100, 120, 140, 160];

to_bin_first = floor(angle/20)+1;

%                ----- % magnitude----- to_bin_first
%   digree------|
%                ----- % magnitude----- to_bin_second
if to_bin_first == 10
    to_bin_first = 1;
end

to_bin_second = to_bin_first + 1;

if to_bin_second == 10
    to_bin_second = 1;
end
% Histogram contains 9 bins.
% |0-20|20-40|40-60|60-80|80-100|100-120|120-140|140-160|160-180/0|
% |  1 |  2  |  3  |  4  |  5   |   6   |   7   |   8   |    9    |
% |  0 |  20 | 40  | 60  | 80   |  100  |  120  |  140  |   160   |

residual_angle = angle - angle_ratio_finder(to_bin_first);

if (residual_angle == 180)
    residual_angle = 0;
end

% What percentage to go where
percentage_of_mag_first_bin = magnitude * (1-(residual_angle*0.05));

percentage_of_mag_second_bin = magnitude - percentage_of_mag_first_bin;

bin_nos       = [to_bin_first, to_bin_second];

bin_magnitude = [percentage_of_mag_first_bin,percentage_of_mag_second_bin];

end

function [I] = applyGaussianFilter(image)
%% Function Info:
%  This function SMOOTHENS the image by applying gaussian filter.

G = [2,  4,   5,  4,  2;
    4,  9,  12,  9,  4;
    5, 12,  15, 12,  5;
    4,  9,  12,  9,  4;
    2,  4,   5,  4,  2];

G = (1/159).*G;

I = image;
[im_row,im_col] = size(I);

% Convolution of image with Guassian filter
for i = 3 : (im_row-2)
    for j = 3 : (im_col-2)
        test_location   = image(i-2:i+2,j-2:j+2);
        I(i,j)          = sum(sum(G.*test_location));
    end
end

end

function [I_Grad, I_Directn] = runSobelOnImage (I)
% Sobel X direction operator
G_x = [ -1,  0,  1;
        -2,  0,  2;
        -1,  0,  1];

% Sobel Y direction operator
G_y = [ 1,  2,  1;
        0,  0,  0;
       -1, -2, -1];
   
[im_row,im_col] = size(I);

% These two variables hold the magniture and direction of image intensities
I_Grad    = I;
I_Directn = uint8(zeros(size(I)));

% <TO-DO> Fix this ugly logc
for i = 2 : (im_row-1)
    for j = 2 : (im_col-1)
        % Calculating Sobel gradients' magnitude
        test_location       = I(i-1:i+1,j-1:j+1);
        x_grad              = sum(sum(G_x.*test_location));
        y_grad              = sum(sum(G_y.*test_location));
        
        % Magnitude
        I_Grad(i,j)         = sqrt(x_grad.^2 + y_grad.^2);
        % Angle (atan2(y_grad, x_grad))*(180/pi)
        I_Directn(i,j) 		= uint8(atan2(y_grad, x_grad)*(180/pi));
        if (I_Directn(i,j)  < 0)
            I_Directn(i,j)  = I_Directn(i,j) + 180;
        end       
    end
end

end