function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    prediction = X*theta;                       % m x 1 
    error = prediction - y;                     % m x 1
    theta = theta - (alpha/m)*(X'*error);       % 2 x 1

    % ============================================================

    % Save the cost J in every iteration    
    % disp(size(J_history(iter)));
    % disp(size(computeCost(X, y, theta)));
    J_history(iter) = computeCost(X, y, theta);
    disp(J_history(iter));

end
% plot(J_history.*100, num_iters, 'bd')
end
