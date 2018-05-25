Xy = csvread('train.csv');
X = Xy(:,2:end)./255; %load examples 
y = Xy(:,1); %load answers to the examples
m = size(X, 1);  

#set zeros to 10th index
for j = 1:m,
  if y(j,:) == 0,
    y(j,:) = 10;
  endif;
endfor;

batch_size = 100; %how many examples we train on at a time
numOfExamples = int64(m*.8); %we don't use the entire set for training

#use some of the training set for testing accuracy (20% is used for testing)
cutoff_for_test_set = numOfExamples;
X = X(1:cutoff_for_test_set,:);
y = y(1:cutoff_for_test_set,:);
Xtest = X((cutoff_for_test_set + 1):m,:);
ytest = y((cutoff_for_test_set + 1):m,:);

disp(m);
disp(size(X));
disp(size(y));
disp(size(Xtest));
disp(size(ytest));