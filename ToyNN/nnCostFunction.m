#TWO hidden layers
function [Theta1_grad Theta2_grad Theta3_grad J] = nnCostFunction(Theta1, ...
                                   Theta2, ...
                                   Theta3, ...
                                   num_Hidden, ...
                                   num_Hidden_layer2, ...
                                   num_labels, ...
                                   X, y, lambda)

% Number of X inputs
m = size(X, 1);
         
% What we need to return (cost and cost gradients)
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

#Forward propagation
adjustedX = [ones(m,1),X];
a2 = tanh(adjustedX * Theta1'); 
a2 = [ones(size(a2,1),1),a2];
a3 = tanh(a2*Theta2');
a3 = [ones(size(a3,1),1),a3];
a4 = sigmoid(a3*Theta3');
%MAKE Y VECTOR [0,0,0...1..0,0] from [1;2;1;1]. etc
y2 = zeros(m,num_labels);
for d = 1:m,
  y2(d,y(d))=1; 
endfor;
# Regularization and Cost function
R = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2))+sum(sum(Theta3(:,2:end).^2)));
J = 1/m * sum(sum((-1.*y2).*log(a4) - (1.-y2).*log(1-a4))) + R; 

% Back Propagation (compute gradients) (include regularization)
for i = 1:m,
  a1 = [1,X(i,:)];
  z2 = a1*Theta1';
  a2 = [1,tanh(z2)];
  z3 = a2*Theta2';
  a3 = [1,tanh(z3)];
  z4 = a3*Theta3';
  a4 = sigmoid(z4);
  yI = zeros(1,num_labels);
  yI(y(i)) = 1;
  
  cost4 = a4-yI;
  cost3 = cost4*Theta3.*[tanhGradient(a3)];
  cost3 = cost3(:,2:end);
  cost2 = cost3*Theta2.*[tanhGradient(a2)];
  cost2 = cost2(:,2:end);
  
  Theta1_grad = Theta1_grad + cost2'*a1;
  Theta2_grad = Theta2_grad + cost3'*a2;
  Theta3_grad = Theta3_grad + cost4'*a3;
  
endfor;

#regularization
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
Theta3_grad = Theta3_grad / m;
Theta1_grad = Theta1_grad + lambda/m * Theta1;
Theta2_grad = Theta2_grad + lambda/m * Theta2;
Theta3_grad = Theta3_grad + lambda/m * Theta3;

end