function [J, grad] = costFunction(theta, X, y)


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


z = X * theta;
h_x = sigmoid(z);

J = (1/m)*sum((-y.*log(h_x)) - ((1 - y).*log(1 - h_x)));
grad = (1/m)*(X'*(h_x - y));



end
