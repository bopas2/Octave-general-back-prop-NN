pkg load image
img = imread('four.png');
img = rgb2gray(img);
img = double(img)./255;
img = reshape(img,1,784);

load('theta1.txt');
load('theta2.txt');
load('theta3.txt');

disp(predict(bestTheta1,bestTheta2,bestTheta3,img));