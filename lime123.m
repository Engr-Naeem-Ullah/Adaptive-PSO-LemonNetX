clc;
clear all;

img = imread('G:\PhD course work\lemon quality dataset paper done\bad_quality_209_enhanced.jpg');


load deeplemonnet.mat;

net1 = deeplemonnet;
%analyzeNetwork(net1)

inputSize = net1.Layers(1).InputSize(1:2);
classes = net1.Layers(end).Classes;

img = imresize(img,inputSize);
X = imresize(img,inputSize);
imshow(X);
label = classify(net1,X);
display(label)

%Classify the image, and display the three classes with the highest classification score in the image title.
[YPred,scores] = classify(net1,img);
[~,topIdx] = maxk(scores, 3);
topScores = scores(topIdx);
topClasses = classes(topIdx);

imshow(img)
titleString = compose("%s (%.2f)",topClasses,topScores');
title(sprintf(join(titleString, "; ")));

%Identify Areas of an Image the Network Uses for Classification
map = imageLIME(net1,img,YPred);

%Display the image with the LIME map overlaid.
figure
imshow(img,'InitialMagnification',150)
hold on
imagesc(map,'AlphaData',0.5)
colormap jet
colorbar

title(sprintf("Image LIME (%s)", ...
    YPred))
hold off


%To compare LIME with Grad-CAM we can produce Grad-CAM like results using
%LIME as well

map = imageLIME(net1,img,"bad_quality", ...
    "Segmentation","grid",...
    "OutputUpsampling","bicubic",...
    "NumFeatures",100,...
    "NumSamples",6000,...
    "Model","linear");

imshow(img,'InitialMagnification', 150)
hold on
imagesc(map,'AlphaData',0.5)
colormap jet

title(sprintf("Image LIME (%s - linear model)", ...
    YPred))
hold off


% Display Only the Most Important Features

%Compute the LIME map and obtain the feature map and the calculated importance of each feature.
[map,featureMap,featureImportance] = imageLIME(net1,img,YPred);



%Find the indices of the top four features.
numTopFeatures = 6;
[~,idx] = maxk(featureImportance,numTopFeatures);

% Plot a bar chart to visualize the importance of the top features
figure
bar(featureImportance(idx), 'b');
xlabel('Feature Index');
ylabel('Feature Importance');
title(sprintf('Top %d Most Important Features', numTopFeatures));

%Next, mask out the image using the LIME map so only pixels in the most important four superpixels are visible. Display the masked image.

mask = ismember(featureMap,idx);
maskedImg = uint8(mask).*img;

figure
imshow(maskedImg);

title(sprintf("Image LIME (%s - top %i features)", ...
    YPred, numTopFeatures))

