%loading dataset
digitDatasetPath = fullfile("G:\PhD course work\lemon quality dataset paper done\final experiments for paper writing\Lemon quality Training dataset");
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8);
digitDatasetPath1 = fullfile("G:\PhD course work\lemon quality dataset paper done\final experiments for paper writing\Lemon quality testing dataset");
imdsValidation1 = imageDatastore(digitDatasetPath1, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

 
%googlenet code
augmenter = imageDataAugmenter('RandXReflection', true);
    % Resizing all training images to [224 224] for ResNet architecture
auimdtests = augmentedImageDatastore([227 227],imdsValidation1, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);

% Determine the split up
%total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);
% Visualize random images

% Number of Images      
%num_images=length(imdsTrain.Labels);



%%K-fold Validation
% Number of folds
num_folds=5;

% Loop for each fold
for fold_idx=1:num_folds
    
    fprintf('Processing %d among %d folds \n',fold_idx,num_folds);
    
  %  Test Indices for current fold
    test_idx=fold_idx:num_folds:num_images;

    % Test cases for current fold
    imdsTest = subset(imds,test_idx);
    
    % Train indices for current fold
    train_idx=setdiff(1:length(imds.Files),test_idx);
    
    % Train cases for current fold
    imdsTrain = subset(imds,train_idx);
    
%end


%googlenet code
augmenter = imageDataAugmenter('RandXReflection', true);
    % Resizing all training images to [224 224] for ResNet architecture
    auimds = augmentedImageDatastore([227 227],imdsTrain, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);

% Resizing all testing images to [224 224] for ResNet architecture   
    augValidationimds = augmentedImageDatastore([227 227],imdsTest, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);
    
    
% Number of Images
num_images=length(imds.Labels);

%%K-fold Validation
% Number of folds
%num_folds=10;
 
% Add Fuzzy Layer
%fuzzyLayer = fuzzyLayer('Name', 'fuzzy');

layers = [
    imageInputLayer([227 227 3],"Name","data")
    
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    leakyReluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","norm1","K",1)
    
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    
    %groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    %leakyReluLayer("Name","relu2")
    %crossChannelNormalizationLayer(5,"Name","norm2","K",1)
    
    %maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])

    %removed shufflenet unit
    groupedConvolution2dLayer([1 1],34,4,"Name","node_18")
    batchNormalizationLayer("Name","node_19")
    leakyReluLayer("Name","node_20")
    nnet.shufflenet.layer.ChannelShufflingLayer("shuffle_21to23",4)  
    groupedConvolution2dLayer([3 3],1,136,"Name","node_24","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_25")
    groupedConvolution2dLayer([1 1],34,4,"Name","node_26")
    batchNormalizationLayer("Name","node_27")
    
    
    %convolution2dLayer([3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    %batchNormalizationLayer("Name","bn2a_branch2a")
    %leakyReluLayer("Name","res2a_branch2a_relu")
    
    %convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    %batchNormalizationLayer("Name","bn2a_branch2b")
    
    %convolution2dLayer([3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    %batchNormalizationLayer("Name","bn2a_branch2a")
    %leakyReluLayer("Name","res2a_branch2a_relu")
    
    %convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    %batchNormalizationLayer("Name","bn2a_branch2b")
    
    %convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    %leakyReluLayer("Name","relu3")
    %batchNormalizationLayer("Name","node_14f")
    
    %groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    %leakyReluLayer("Name","relu4")
    %batchNormalizationLayer("Name","node_14kj")
    
   
    maxPooling2dLayer([3 3],"Name","pool51","Padding",[0 1 0 1],"Stride",[2 2])
    
    convolution2dLayer([1 1],48,"Name","fire5-squeeze1x1")
    batchNormalizationLayer("Name","node_14")
    leakyReluLayer("Name","fire6-relu_squeeze1x1")    
    convolution2dLayer([1 1],192,"Name","fire5-expand1x1")
    batchNormalizationLayer("Name","node_15")
    leakyReluLayer("Name","fire6-relu_expand1x1")
    convolution2dLayer([3 3],192,"Name","fire5-expand3x3","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_16")
    leakyReluLayer("Name","fire5-relu_expand3x3") 
    
    %groupedConvolution2dLayer([1 1],68,4,"Name","node_103")
    %batchNormalizationLayer("Name","node_104")
    %leakyReluLayer("Name","node_105")
    
    %groupedConvolution2dLayer([3 3],1,272,"Name","node_109","Padding",[1 1 1 1])
    %batchNormalizationLayer("Name","node_110")
    %leakyReluLayer("Name","node_4418")
    
    %groupedConvolution2dLayer([1 1],68,4,"Name","node_111")
    %batchNormalizationLayer("Name","node_rag112")
    %leakyReluLayer("Name","node_112")
    
    %groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    %leakyReluLayer("Name","relu5")
    %batchNormalizationLayer("Name","nodevk_14")
    
    maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
   % fuzzyLayer    

    flattenLayer('Name', 'flatten')
    selfAttentionLayer(8, 64, 'Name', 'self_attention')


    fullyConnectedLayer(3096,"Name","fc7","BiasLearnRateFactor",2)
    leakyReluLayer("Name","relu7")
    batchNormalizationLayer("Name","nodjcge_14")
    dropoutLayer(0.5,"Name","drop7")
    
    fullyConnectedLayer(2,"Name","fc8","BiasLearnRateFactor",2)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augValidationimds, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
% Training 
    
    deeplemonnet = trainNetwork(auimds,layers,options);
    %model_name = strcat("deeplemonnet_", num2str(fold_idx), ".mat");
    %save(model_name, "deeplemonnet");
% Actual Labels
   actual_labels=imdsTest.Labels;

   [predicted_labels, scores] = classify(deeplemonnet,augValidationimds);
% Confusion Matrix
   figure;
   C = confusionmat(actual_labels,predicted_labels);

   confusionchart(C);
%title('Confusion Matrix: TumorResNet');
% Testing and their corresponding Labels and Posterior for each Case
   [predicted_labels, scores] = classify(deeplemonnet,auimdtests);
%%Performance Study
% Actual Labels
   actual_labels=imdsValidation1.Labels;
% Confusion Matrix
   figure;
   C = confusionmat(actual_labels,predicted_labels);
   cm = confusionchart(C);
   cm.Title = 'Testing Confusion Matrix';
%analyzeNetwork(net1)
    
end
