deepDatasetPath1 = fullfile('G:\upcoming research\Orange diseases dataset\github_repo\New orange leaf disease dataset\test');
imds1 = imageDatastore(deepDatasetPath1, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Divide the data into training and validation data sets
%numTrainFiles = 25;
%[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
%[imdsTrain1,imdsValidation1] = splitEachLabel(imds,0.8);

augmenter = imageDataAugmenter('RandXReflection', true);
    % Resizing all training images to [224 224] for ResNet architecture
auimdtests = augmentedImageDatastore([227 227],imds1, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);


load net.mat;
load BeesNetL.mat;


% Common CNN Accuracy
YPred = classify(net,auimdtests);
YValidation = imds1.Labels;
CNNaccuracy = sum(YPred == YValidation)/numel(YValidation);
% Bees CNN Accuracy
YPredbee = classify(BeesNetL,auimdtests);
YValidationbee = imds1.Labels;
Beesaccuracy = sum(YPredbee == YValidationbee)/numel(YValidationbee);

%% Confusion Matrix
figure;
plotconfusion(YPred,YValidation);
title(['CNN Accuracy  =  ' num2str(CNNaccuracy)]);
figure;
plotconfusion(YPredbee,YValidationbee);
title(['Bees-CNN Accuracy  =  ' num2str(Beesaccuracy)]);

%% Statistics
fprintf('The CNN Accuracy Is =  %0.4f.\n',CNNaccuracy*100)
fprintf('The Bees CNN Accuracy Is =  %0.4f.\n',Beesaccuracy*100)
