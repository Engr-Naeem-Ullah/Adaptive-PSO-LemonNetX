clear all;
clc;
close all;

%uploading testing dataset for feature extraction purpose
imds = imageDatastore('G:\PhD course work\lemon quality dataset paper done\Lemon quality Training dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

numTrainImages = numel(imds.Labels);
load deeplemonnet.mat;

%used my trained model 
net1 = deeplemonnet;
net1.Layers
analyzeNetwork(net1)

inputSize = net1.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(inputSize(1:2),imds);

%Feature extraction from last fully connected layer
layer = 'fc7';
deeplemonTfeatures = activations(net1,augmentedTrainingSet,layer,'OutputAs','rows');

%training lables
label = imds.Labels;
save('label.mat','label');

%saving features
save('deeplemonTfeatures.mat','deeplemonTfeatures');

