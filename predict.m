function p = predict(theta, X)


m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

h_x = sigmoid(X*theta);
p = [h_x >=0.5];


end
