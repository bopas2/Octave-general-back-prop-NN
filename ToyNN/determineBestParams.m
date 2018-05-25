%% Setup the parameters 
input_layer_size  = 28*28;  % number of inputs
hidden_layer_size = 40;   % number of hidden neurons, FIRST LAYER
hidden_layer_two_size = 25; # number of hidden neurons, SECOND LAYER
num_labels = 10;          % number of possible outputs

Xy = csvread('train.csv');
X0 = Xy(:,2:end)./255; %load examples 
y0 = Xy(:,1); %load answers to the examples
m = size(X0, 1);  

#set zeros to 10th index
for j = 1:m,
  if y0(j,:) == 0,
    y0(j,:) = 10;
  endif;
endfor;

batch_size = 200; %how many examples we train on at a time
numOfExamples = int64(m*.8); %we don't use the entire set for training

#use some of the training set for testing accuracy (20% is used for testing)
cutoff_for_test_set = numOfExamples;
X = X0(1:cutoff_for_test_set,:);
y = y0(1:cutoff_for_test_set,:);
Xtest = X0((cutoff_for_test_set + 1):end,:);
ytest = y0((cutoff_for_test_set + 1):end,:);
mTest = size(ytest,1); 

lambdas = [3,1,.3,.1,.03,.01];
alphas = [3,1,.3,.1,.03,.01];
num_of_iterations = 10; #how many times we train on a set of data
dataBase = ones(37,3);
rowcount = 1;

for r = lambdas,
  for e = alphas,   
    Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size); % make random set of weights
    Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_two_size);
    Theta3 = randInitializeWeights(hidden_layer_two_size, num_labels);  
    for i = 1:num_of_iterations,
      for j = 1:int64((numOfExamples - 2*batch_size) / batch_size),
      Xselect = X((j*batch_size):(j*batch_size+batch_size),:);
      yselect = y((j*batch_size):(j*batch_size+batch_size),:);
      [T1, T2, T3, J] = nnCostFunction(Theta1,Theta2,Theta3,hidden_layer_size,hidden_layer_two_size,num_labels,Xselect,yselect,r); # gets gradients
      Theta1 = Theta1 - e*T1; #trains
      Theta2 = Theta2 - e*T2;
      Theta3 = Theta3 - e*T3;
      endfor;
    endfor;
    d = predict(Theta1,Theta2,Theta3,Xtest) == ytest;
    correct = 0;
    for k = 1:mTest,
      if(d(k,:)==1),
        correct = correct + 1;
      endif;
    endfor;
    dataBase(rowcount,:) = [r,e,correct/(size(ytest,1))*100];
    rowcount = rowcount + 1;
  endfor;
endfor;  
    
disp(dataBase);
