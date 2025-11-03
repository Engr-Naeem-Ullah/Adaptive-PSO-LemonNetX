%loading dataset
digitDatasetPath = fullfile("E:\PhD course work\lemon quality dataset paper done\Lemon quality Training dataset");
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8);
digitDatasetPath1 = fullfile("E:\PhD course work\lemon quality dataset paper done\Lemon quality testing dataset");
imds1 = imageDatastore(digitDatasetPath1, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');




%%K-fold Validation
% Number of folds
%num_folds=5;

% Loop for each fold
%for fold_idx=1:num_folds
    
 %   fprintf('Processing %d among %d folds \n',fold_idx,num_folds);
    
  %  Test Indices for current fold
  %  test_idx=fold_idx:num_folds:num_images;

    % Test cases for current fold
   % imdsTest = subset(imds,test_idx);
    
    % Train indices for current fold
    %train_idx=setdiff(1:length(imds.Files),test_idx);
    
    % Train cases for current fold
    %imdsTrain = subset(imds,train_idx);
    
%end



augmenter = imageDataAugmenter('RandXReflection', true);
    % Resizing all training images to [224 224] for ResNet architecture
    augimdsTrain = augmentedImageDatastore([224 224],imds, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);

% Resizing all testing images to [224 224] for ResNet architecture   
    augimdsValidation = augmentedImageDatastore([224 224],imds1, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);
    

% Number of Images
num_images=length(imds.Labels);

%Load Pretrained Network
resnet18 = resnet18;

%mobilenetv2 done
%resnet18 done
%squeezenet done

%lgraph = layerGraph(net);
%analyzeNetwork(Wristfractureefficientnet)
resnet18.Layers(1)

inputSize = resnet18.Layers(1).InputSize;

lgraph = layerGraph(resnet18);

[learnableLayer,classLayer] = findLayersToReplace(lgraph);


numClasses = numel(categories(imds.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);


newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);





%train network



 
%miniBatchSize = 10;
%valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',8, ...
    'Verbose',false, ...
    'Plots','training-progress');

resnet18 = trainNetwork(augimdsTrain,lgraph,options);


save(model_name, "resnet18");

[YPred, scores] = classify(resnet18,augimdsValidation);
YActual = imds1.Labels;



%accuracy = sum(YPred == YActual)/numel(YActual);
%disp('Accuracy');
%disp(accuracy);
% g1 = [3 2 2 3 1 1]';	% Known groups
% g2 = [4 2 3 NaN 1 1]';	% Predicted groups
%Return the confusion matrix.
% C = confusionmat(g1,g2)
CM = confusionmat(YActual,YPred);

confusionchart(CM)

%ROC CURVE
%test_labels=double(nominal(imdsValidation.Labels));
%cscores = scores;
%ROC CURVE
%test_labels=double(nominal(imdsValidation.Labels));
%%cscores = scores;
% ROC Curve - Our target class is the first class in this scenario 
%[fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,cscores(:,1),1);
%figure;
%plot(fp_rate,tp_rate,'b-');
%grid on;
%xlabel('False Positive Rate');   
%ylabel('True Positive Rate');
%title('ROC Wrist Fracture Dection Using X-Rays') 

% Area under the ROC curve value
%AUC

%accuracy = sum(YPred == YActual)/numel(YActual);
%disp('Accuracy');
%disp(accuracy);
%Precision = TP / (TP + FP)
%Recall = TP / (TP + FN)
% % << for an existing Confision Matrix 'CM' >>
%precision = diag(CM)./sum(CM,2);
%overall_prec = mean(precision);
%disp('Recision');

%disp(overall_prec);

% % And another for recall
%recall = diag(CM)./sum(CM,1)';
%disp('Recall');
%overall_recall = mean(recall);
%disp(overall_recall);

%save("Wristfracturemobilnetv2net.mat","Wristfracturemobilenetv2net") 

%Testing phase
% Testing and their corresponding Labels and Posterior for each Case
   % [predicted_labels, scores] = classify(xceptionnet,auimdtests);
%%Performance Study
% Actual Labels
%actual_labels=imds1.Labels;
% Confusion Matrix
%figure;
%C = confusionmat(actual_labels,predicted_labels);
%cm = confusionchart(C)
%cm.Title = 'Testing Confusion Matrix';
%analyzeNetwork(net1)
 %save("xceptionnet.mat","xceptionnet")