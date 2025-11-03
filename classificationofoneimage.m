% Clear environment
clc;
clear;
close all;

% Load the trained model
%modelPath = 'E:\PhD course work\lemon quality dataset paper done\deeplemonnet.mat';
%loadedData = load(modelPath);
load deeplemonnet.mat;
%net = loadedData.net;

%used my trained model 
net = deeplemonnet;

% Read and preprocess the test image
imagePath = 'E:\PhD course work\lemon quality dataset paper done\Lemon quality testing dataset\bad_quality\bad_quality_1.jpg';
img = imread(imagePath);

% Resize to match input size of the model
inputSize = net.Layers(1).InputSize;
if size(img, 3) == 1
    img = cat(3, img, img, img); % Convert grayscale to RGB
end
imgResized = imresize(img, [inputSize(1), inputSize(2)]);

% Start timing
tic;

% Classify image
[label, score] = classify(net, imgResized);

% Stop timing
inferenceTime = toc;

% Display results
fprintf('Predicted label: %s\n', string(label));
fprintf('Inference time: %.4f seconds\n', inferenceTime);

% Optionally show image
figure;
imshow(img);
title(['Predicted: ', char(label), ', Time: ', num2str(inferenceTime, '%.4f'), ' sec']);
