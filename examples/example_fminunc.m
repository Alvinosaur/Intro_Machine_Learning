% GradObj and on specify that we will provide our own gradient function
% MaxIter sets max # iterations to 100 in this case
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
% notice ... is like \ in python

% functionVal should --> 0 or whatever is converging
% exitFlag(1 or 0) tells us whether or not convergence is reached
% Note: theta must be >= 2 dimensions for fminunc to work
[optTheta, functionalVal, exitFlag] = ...
fminunc(@cost_function, initialTheta, options);

% cmd+/ to comment, cmd+T to uncomment
disp('optTheta'), disp(optTheta);
disp('functionalVal'), disp(functionalVal);
disp('exitFlag'), disp(exitFlag);