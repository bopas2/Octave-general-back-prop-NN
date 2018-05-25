function t = tanhGradient(z)
%computes the tanh gradient, used for backprop
t = 1.-tanh(z).^2;
end
