function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
             % each row for a hidden unit, each column for an input weight

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% X = [ones(m, 1) X];      % 5000 x 401
% Delta2 = zeros(size(Theta2));
% Delta1 = zeros(size(Theta1));
% for i = 1:m             % for all 5000 training examples
%     % Forward Prop
%     activation_2 = sigmoid(X(i,:)*Theta1');            % 1 x 25
%     activation_2 = [1 activation_2];                   % 1 x 26, a_0
%     activation_3 = sigmoid(activation_2*Theta2');      % 1 x 10
%     y_vec = zeros(1, num_labels);       % 1 x 10
%     y_vec(y(i)) = 1;
%     sigma_3 = activation_3 - y_vec;     % 1 x 10
%     sigma_2 = Theta2'*sigma_3'.*sigmoidGradient(activation_2'); % 26 x 1
%     sigma_2 = sigma_2(2:end, 1);     % 25 x 1
%     Delta2 = Delta2 + sigma_3'*activation_2;       % 10 x 26
%     % Note: activation_1 = X
%     Delta1 = Delta1 + sigma_2.*X(i,:);        % 25 x 401
%     J = J + (-1/m)*(y_vec*log(activation_3') + ...
%         (1-y_vec)*log(1-activation_3'));
% end
% J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + ...
%     sum(sum(Theta2(:,2:end).^2)));
% disp(size(Theta1_grad));
% disp(size(Theta2_grad));
% disp(size(Delta1./m));
% disp(size(Delta2./m));
% Theta1_grad = Delta1./m;
% Theta2_grad = Delta2./m;


% m = num_examples
% n = num_features
% k = num_labels
% sl = num_units for a given layer, l

% Set logical array
y_vec = zeros(m, num_labels);       % m x k
for i = 1:m
    y_vec(i, y(i)) = 1;
end

% Forward prop
a_1 = [ones(m, 1), X];              % m x (n+1)
z_2 = Theta1*a_1';                  % s1 x m = (s1 x n+1) * (n+1 x m)
a_2 = sigmoid(z_2);  
z_2 = [ones(1, m); z_2];
a_2 = [ones(1, m); a_2];             % s1+1 x m
a_3 = sigmoid(Theta2*a_2);          % k x m  = (k x s1+1) * (s1+1 x m)

% backprop
sigma_3 = a_3 - y_vec';              % 
Delta_2 = sigma_3*a_2';              % k x s1+1 = (k x m) * (s1+1 x m)
% s1+1 x m = (k x s1+1) * (k x m)
sigma_2 = Theta2'*sigma_3.*sigmoidGradient(z_2);
Delta_1 = sigma_2(2:end, :)*a_1;   % s1 x n+1 = (s1 x m) * (m x n+1)

reg_theta1_grad = [zeros(size(Theta1,1), 1), Theta1(:,2:end)];
reg_theta2_grad = [zeros(size(Theta2,1), 1), Theta2(:,2:end)];

Theta1_grad = Delta_1./m + (lambda/m)*reg_theta1_grad;
Theta2_grad = Delta_2./m + (lambda/m)*reg_theta2_grad;






% Cost function
% (k x m) * (k x m)
% We don't use normal vector multiplication b/c we only want a specific
% y value's cost prediction to multiply, so use element-wise..
% if you use normal vector mult., then you will get your desired 
% multiplcation AND a bunch of undesired multiplications
% See this link: http://mathworld.wolfram.com/MatrixMultiplication.html
J = (-1/m)*(sum(sum(y_vec'.*log(a_3))) + sum(sum((1-y_vec)'.*log(1-a_3)))); 
% disp((sum(sum(Theta1.^2)) + sum(sum(Theta2.^2))));
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + ...
    sum(sum(Theta2(:,2:end).^2)));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
