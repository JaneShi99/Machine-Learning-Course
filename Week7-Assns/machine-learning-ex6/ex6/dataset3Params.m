function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

err_matrix = zeros(length(C_arr),length(sigma_arr));
for c = 1:length(C_arr)
    for sig = 1:length(sigma_arr)
        model= svmTrain(X, y, C_arr(c), @(x1, x2) gaussianKernel(x1, x2, sigma_arr(sig)));
        predictions = svmPredict(model,Xval);
        err_matrix(c,sig) = mean(double(predictions ~= yval));
    endfor
endfor

disp(err_matrix);

minimum = min(min(err_matrix));
[C_index,sigma_index]=find(err_matrix==minimum);
C = C_arr(C_index);
sigma = sigma_arr(sigma_index);
disp(sigma);
disp(C);
% =========================================================================

end
