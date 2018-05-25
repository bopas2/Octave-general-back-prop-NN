function g = sigmoidGradient(z)
%computes the sigmoid gradient, used for backprop
 
g = sigmoid(z).*(1.-sigmoid(z));
end
