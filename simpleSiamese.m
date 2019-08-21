try
	nnet.internal.cnngpu.reluForward(1);
catch ME
end


%% RETRIEVING DATA FROM FILES

outputFolder = fullfile('geckodata');
rootFolder = fullfile(outputFolder, 'patches');

identities = {'01', '02', '03'};
indices = 1:251;

imds = imageDatastore(fullfile(rootFolder), 'IncludeSubFolders', true, 'LabelSource', 'foldernames', 'FileExtensions', {'.mat', '.jpg'}, 'ReadFcn', @customReadFcn);
imgStore = subset(imds, indices);

counts = countEachLabel(imgStore);
numClasses = 50;

first = find(imgStore.Labels == '01', 1);
second = find(imgStore.Labels == '02', 1);
third = find(imgStore.Labels == '03', 1);

[trainingSet, validationSet] = splitEachLabel(imgStore, 0.7, 'randomize');

%% DATASET PREPROCESSING

model = alexnet;

inputSize = model.Layers(1).InputSize;

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

augmentedTrainingSet = augmentedImageDatastore(inputSize(1:2), trainingSet, ...
    'ColorPreprocessing', 'gray2rgb');

augmentedValidationSet = augmentedImageDatastore(inputSize(1:2), validationSet, ...
    'ColorPreprocessing', 'gray2rgb');

% [PairsData, PairsLabels] = getPairs(augmentedTrainingSet, trainingSet.Labels);

trainingLabels = trainingSet.Labels;
validLabels = validationSet.Labels;

[tripletsData, tripletsLabels] = getTriplets(augmentedTrainingSet, trainingLabels);

%% INITIALIZING AND TRAINING NETWORK

trainOptions = trainingOptions('sgdm', ...
    'MiniBatchSize', 30, ...
    'MaxEpochs', 1000, ...
    'GradientThreshold', 2, ...
    'InitialLearnRate', 1e-6, ...
    'Momentum', 0.99, ...
    'Shuffle', 'never', ...                 % Very Important: 'Shuffle' should be set to 'never'
    'Verbose', true, ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress');

layersTransfer = model.Layers(1:end-3);

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses);
%     softmaxLayer;
%     classificationLayer;
    TripletLoss];

dummyLabelsForNet = zeros(size(tripletsData, 4), numClasses);
modelTraining = trainNetwork(tripletsData, dummyLabelsForNet, layers, trainOptions);

%% CALCULATING ACCURACY AFTER INITIAL TRAINING

featureLayer = 'fc';
trainingSetFeatures = activations(modelTraining, augmentedTrainingSet, featureLayer, 'OutputAs', 'rows', 'ExecutionEnvironment', 'gpu');
validationSetFeatures = activations(modelTraining, augmentedValidationSet, featureLayer, 'OutputAs', 'rows', 'ExecutionEnvironment', 'gpu');

trainingLabels = trainingSet.Labels;
validationLabels = validationSet.Labels;

validationPred = categorical(zeros(size(validationLabels, 1), 1));

for k = 1:size(validationLabels, 1)
    targetMat = validationSetFeatures(k, :);
    minDist = 1000.00;
    minIdx = 0;
    for l = 1:size(trainingLabels, 1)
%        if l == k
%           continue 
%        end
       tempDist = pdist2(targetMat, trainingSetFeatures(l, :), 'euclidean');
       if tempDist < minDist
          minDist = tempDist;
          minIdx = l;
       end
    end
    validationPred(k) = trainingLabels(minIdx);
end


initialAccuracy = mean(validationPred == validationLabels);

%% PREPARING HARD TRIPLETS

predictTrain = predict(modelTraining, permute(augmentedTrainingSet, [1 2 4 3]), 'ExecutionEnvironment', 'gpu');
[tripletsNewOrder, hardTripletsData] = getHardTriplets(predictTrain, augmentedTrainingSet, trainingLabels);

hardTripletsLabels = trainingLabels(tripletsNewOrder);
% hardTripletsData = permute(augmentedTrainingSet, [1 2 4 3]);

%% RETRAINING THE NETWORK ON HARD TRIPLETS

trainOptions = trainingOptions('sgdm', ...
    'MiniBatchSize', 30, ...
    'MaxEpochs', 1000, ...
    'GradientThreshold', 2, ...
    'InitialLearnRate', 1e-6, ...
    'Shuffle', 'never', ...                 % Very Important: 'Shuffle' should be set to 'never'
    'Verbose', true, ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress');

trainedModel = trainNetwork(hardTripletsData, dummyLabelsForNet, modelTraining.Layers, trainOptions);

%% CALCULATING ACCURACY AFTER RETRAINING

featureLayer = 'fc';
trainingSetFeatures = activations(trainedModel, augmentedTrainingSet, featureLayer, 'OutputAs', 'rows', 'ExecutionEnvironment', 'gpu');
validationSetFeatures = activations(trainedModel, augmentedValidationSet, featureLayer, 'OutputAs', 'rows', 'ExecutionEnvironment', 'gpu');

trainingLabels = trainingSet.Labels;
validationLabels = validationSet.Labels;

validationPred = categorical(zeros(size(validationLabels, 1), 1));

for k = 1:size(validationLabels, 1)
    targetMat = validationSetFeatures(k, :);
    minDist = 1000.00;
    minIdx = 0;
    for l = 1:size(trainingLabels, 1)
%        if l == k
%           continue 
%        end
       tempDist = pdist2(targetMat, trainingSetFeatures(l, :), 'euclidean');
       if tempDist < minDist
          minDist = tempDist;
          minIdx = l;
       end
    end
    validationPred(k) = trainingLabels(minIdx);
end


accuracy = mean(validationPred == validationLabels);

save('tempSpace');

%%
% USING ALEXNET FEATURES
% classifier = fitcecoc(trainingSetFeatures, trainingLabels);
% 
% [validationPreds, scores] = predict(classifier, validationSetFeatures);
% classifierAccuracy = mean(validationPreds == validationLabels);

%%

% USING ALEXNET
% layers = [
%     layersTransfer
%     fullyConnectedLayer(52)
%     softmaxLayer
%     classificationLayer];
% 
% model = trainNetwork(augmentedTrainingSet, layers, trainOptions);
% 
% preds = predict(model, augmentedValidationSet);
% modelAccuracy = mean(preds == validationLabels);

%% HELPER FUNCTIONS 

function data = customReadFcn(filename)
    newImg = load(filename);
    data = newImg.icolor;
end

function [ TripletData, TripletLabels ] = getTriplets(trainingSet, trainingLabels)
    totalImages = size(trainingLabels, 1);
    TripletData = zeros(227, 227, 3, 3 * totalImages);
%     TripletData = cell(totalImages, 3);
%     TripletLabels = categorical(totalImages, 3);
    TripletLabels = [trainingLabels; trainingLabels; trainingLabels];
    for k = 1:totalImages
        indv = trainingLabels(k);
        sameIndv = find(trainingLabels == trainingLabels(k));
        otherIndvs = find(trainingLabels ~= trainingLabels(k));
        sameIndvIdx = randperm(length(sameIndv), 1);
        otherIndvIdx = randperm(length(otherIndvs), 1);
        
        TripletData(:, :, :, 3 * k - 2) = getNormalized(trainingSet, k);
        TripletLabels(3 * k - 2) = trainingLabels(k);

        TripletData(:, :, :, 3 * k - 1) = getNormalized(trainingSet, sameIndv(sameIndvIdx));
        TripletLabels(3 * k - 1) = trainingLabels(sameIndv(sameIndvIdx));

        TripletData(:, :, :, 3 * k) = getNormalized(trainingSet, otherIndvs(otherIndvIdx));
        TripletLabels(3 * k) = trainingLabels(otherIndvs(otherIndvIdx));
    end
end

function [ HardTripletsOrder, HardTripletsData ] = getHardTriplets(predictTrain, trainingSet, trainingLabels)
    numSamples = size(predictTrain, 1);
    distancesMat = zeros(numSamples, numSamples, size(predictTrain, 2));
    for k = 1:size(predictTrain, 2)
        distancesMat(:, :, k) = predictTrain(:, k) - predictTrain(:, k)';
    end
    distances = sum(distancesMat.^2, 3);
    numSubSamples = 100;
    HardTripletsOrder = zeros(3 * numSamples, 1);
    HardTripletsData = zeros(227, 227, 3, 3 * numSamples);
    for k = 1:numSamples
        indv = trainingLabels(k);
        sameIndv = find(trainingLabels == trainingLabels(k));
        otherIndvs = setdiff(1:numSamples, sameIndv);
        otherIndvsSample = otherIndvs(randperm(length(otherIndvs), numSubSamples));
        [~, sameIndvIdx] = max(distances(k, sameIndv));
        [~, otherIndvIdx] = min(distances(k, otherIndvsSample));
        
        HardTripletsOrder(3 * k - 2) = k;
        HardTripletsData(:, :, :, 3 * k - 2) = getNormalized(trainingSet, k);
        
        HardTripletsOrder(3 * k - 1) = sameIndv(sameIndvIdx);
        HardTripletsData(:, :, :, 3 * k - 1) = getNormalized(trainingSet, sameIndv(sameIndvIdx));
        
        HardTripletsOrder(3 * k) = otherIndvs(otherIndvIdx);
        HardTripletsData(:, :, :, 3 * k) = getNormalized(trainingSet, otherIndvs(otherIndvIdx));
    end
end

function resImage = getNormalized(trainingSet, index)
    temp = readByIndex(trainingSet, index);
    temp = cell2mat(temp.input);
    summ = temp(:, :, 1) + temp(:, :, 2) + temp(:, :, 3);
    temp(:, :, 1) = temp(:, :, 1) ./ summ;
    temp(:, :, 2) = temp(:, :, 2) ./ summ;
    temp(:, :, 3) = temp(:, :, 3) ./ summ;
    resImage = temp;
end
