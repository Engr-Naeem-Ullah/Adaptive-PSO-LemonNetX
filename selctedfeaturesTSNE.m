% Load the selected features from the .mat file
loadedData = load('sFeat.mat');
whos

% You would replace 'path_to_your_sFeat_file.mat' with the actual path to your .mat file

% Extract the features from the loaded data
% Replace 'selectedFeatures' with the actual variable name of your selected features in the .mat file
selectedFeatures = loadedData.sFeat;

% If your labels are also in the .mat file, load them as well
% Replace 'labels' with the actual variable name of your labels in the .mat file
load('labelTraining.mat')
labels = labelTraining;

% Perform t-SNE on the selected features
tsneData = tsne(selectedFeatures, 'Algorithm','barneshut','NumPCAComponents',1);

% Create a scatter plot for the t-SNE data
figure;
gscatter(tsneData(:,1), tsneData(:,1), labels);
title('t-SNE Plot of Selected Features');
xlabel('t-SNE Component 1');
ylabel('t-SNE Component 2');
grid on;
