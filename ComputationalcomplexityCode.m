clc;
clear;
close all;

%% Load the trained model
modelPath = 'deeplemonnet.mat'; % adjust if needed
loadedData = load(modelPath);
net = loadedData.deeplemonnet;

%% Read and preprocess one test image
imagePath = 'E:\PhD course work\lemon quality dataset paper done\Lemon quality testing dataset\bad_quality\bad_quality_1.jpg';
img = imread(imagePath);

% Resize to match network input size
inputSize = net.Layers(1).InputSize;
if size(img, 3) == 1
    img = cat(3, img, img, img); % grayscale to RGB
end
imgResized = imresize(img, [inputSize(1), inputSize(2)]);

%% Measure Inference Time on CPU
disp('Running inference on CPU...');
tic;
[label, score] = classify(net, imgResized);
inferenceTime = toc;

% Display prediction results
fprintf('Predicted Label: %s\n', string(label));
fprintf('Inference Time (CPU): %.4f seconds\n', inferenceTime);

%% Display Image with Prediction
figure;
imshow(img);
title(sprintf('Predicted: %s | Time: %.4f sec', char(label), inferenceTime));

%% Model Size on Disk
info = dir(modelPath);
modelSizeMB = info.bytes / (1024 * 1024);
fprintf('Model Size on Disk: %.2f MB\n', modelSizeMB);

%% Total Learnable Parameters
layers = net.Layers;
totalParams = 0;

for i = 1:numel(layers)
    if isprop(layers(i), 'Weights') && ~isempty(layers(i).Weights)
        totalParams = totalParams + numel(layers(i).Weights);
    end
    if isprop(layers(i), 'Bias') && ~isempty(layers(i).Bias)
        totalParams = totalParams + numel(layers(i).Bias);
    end
end

fprintf('Total Learnable Parameters: %.2f Million\n', totalParams / 1e6);

%% (Optional) Analyze Network
disp('Opening network analyzer (close to continue)...');
analyzeNetwork(net);

%% Summary
fprintf('\n=== Model Complexity Summary ===\n');
fprintf('Inference Time (CPU): %.4f seconds\n', inferenceTime);
fprintf('Model Size on Disk: %.2f MB\n', modelSizeMB);
fprintf('Total Learnable Parameters: %.2f Million\n', totalParams / 1e6);
