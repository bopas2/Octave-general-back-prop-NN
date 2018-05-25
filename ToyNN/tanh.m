function t = tanh(z)
%tanh function 
t = 2.0*sigmoid(2.*z).-1.0;
end
