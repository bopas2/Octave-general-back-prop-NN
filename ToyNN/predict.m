function p = predict(Theta1, Theta2, Theta3, X)
m = size(X,1);
num_labels = size(Theta3,1);

p = ones(m,1);

adjustedX = [ones(m,1),X];
a2 = tanh(adjustedX * Theta1'); 
a2 = [ones(size(a2,1),1),a2];
a3 = tanh(a2*Theta2');
a3 = [ones(size(a3,1),1),a3];
a4 = sigmoid(a3*Theta3');

[dummy, p] = max(a4, [], 2);


% =========================================================================


end
