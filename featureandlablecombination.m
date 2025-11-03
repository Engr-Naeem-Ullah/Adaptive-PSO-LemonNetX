clear all;
clc;
close all;

% Load labels and features
load('label.mat'); % Load labels
load('deeplemonTfeatures.mat'); % Load features

% Combine labels and features
features = deeplemonTfeatures;
labels = label;

% Save combined dataset to MAT file
save('deeporangenetTfeaturesandlabels.mat', 'features', 'labels');
