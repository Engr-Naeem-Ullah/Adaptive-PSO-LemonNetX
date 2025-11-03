% Load training data

% Load selected features after PSO
load('sFeat.mat');

% Use only selected features
X_train_selected = deeplemonTrainingfeatures(:, sFeat);


% Train Decision Tree using only selected features
tree_selected = fitctree(X_train_selected, Y_train);

% Visualize Decision Tree (Optional)
view(tree_selected, 'Mode', 'graph');
