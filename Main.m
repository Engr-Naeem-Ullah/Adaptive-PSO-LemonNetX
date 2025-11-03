%-------------------------------------------------------------------%
% Particle Swarm Optimization (PSO) source codes demo version      %
%-------------------------------------------------------------------%

%---Inputs-----------------------------------------------------------
% feat     : feature vector ( Instances x Features )
% label    : label vector ( Instances x 1 )
% N        : Number of particles
% max_Iter : Maximum number of iterations
% c1       : Cognitive factor
% c2       : Social factor
% w        : Inertia weight

%---Outputs-----------------------------------------------------------
% sFeat    : Selected features
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%---------------------------------------------------------------------

%% Particle Swarm Optimization
clc, clear, close; 

% Benchmark data set 
load deeporangenetTfeaturesandlabels.mat; 
%whos

% Display the total number of features
totalFeatures = size(features, 2);
disp(['Total Number of Features: ', num2str(totalFeatures)]);
% Display the features
disp('Features:');
disp(features);


% Display the labels
disp('Labels:');
disp(labels);
% Set 20% data as validation set
ho = 0.2; 

% Hold-out method
HO = cvpartition(labels,'HoldOut',ho);

% Parameter setting
N        = 10;
max_Iter = 100;
c1       = 2;     % cognitive factor
c2       = 2;     % social factor
w        = 1;     % inertia weight

% Particle Swarm Optimization
[sFeat, Sf, Nf, curve] = jPSO(features, labels, N, max_Iter, c1, c2, w, HO);

% Print the selected features, feature indices, and number of selected features
disp('Selected Features:');
disp(sFeat);
disp('Selected Feature Indices:');
disp(Sf);
disp(['Number of Selected Features: ', num2str(Nf)]);


% Save selected features to a MAT file
save('sFeat.mat', 'sFeat');

% Obtain predictions using selected features
selectedFeatures = features(:, Sf);
model = fitcknn(selectedFeatures(HO.training,:), labels(HO.training)); % Adjust the model as needed
predictions = predict(model, selectedFeatures(HO.test,:));

% Calculate confusion matrix
C = confusionmat(labels(HO.test), predictions);

% Display confusion matrix
disp('Confusion Matrix:');
disp(C);

% Plot convergence curve
plot(1:max_Iter, curve);
xlabel('Number of iterations');
ylabel('Fitness Value');
title('PSO Convergence Curve');
grid on;


