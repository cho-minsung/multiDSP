function hist_eq()

I = mat2gray('D:\Documents\Universities\UNLV\Spring_2021\multiDSP\hw1\sample_images\jetplane.tif');
% I = imread('D:\Documents\Universities\UNLV\Spring_2021\multiDSP\hw1\sample_images\jetplane.tif');
size(I)

figure
subplot(1,2,1)
imshow(I)
subplot(1,2,2)
imhist(I,64)

