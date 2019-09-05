function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
% dim(X) -> 5000 x 400 (row, columns)
first_layer = X;
first_layer = [ones(size(first_layer, 1),1) first_layer]; % 5000 x 401

second_layer = sigmoid(first_layer * Theta1'); % 5000 x 401 * 401 x 25 = 5000 x 25
second_layer = [ones(size(second_layer, 1), 1) second_layer]; % 5000 x 26

third_layer = sigmoid(second_layer * Theta2'); % 5000 x 26 * 26 x 10

[vals, p] = max(third_layer'); % 5000 x 10 -> 10 x 5000

p = p';




% =========================================================================


end
