#Xy = csvread('train.csv');
#X0 = Xy(:,2:end); %load examples 
#Xdata = X0(2:101,:);
#save "smallX.txt" Xdata;
data = load('smallX.txt');
image = reshape(Xdata(17,:),28,28);
disp(image);
imshow(image);
