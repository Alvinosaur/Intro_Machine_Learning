function [j_val, gradient] = cost_function(theta)
j_val = (theta(1)-5)^2 + (theta(2)-5)^2;
gradient = zeros(2,1);  % 2x1 vector
gradient(1) = 2*(theta(1)-5);
gradient(2) = 2*(theta(2)-5);

