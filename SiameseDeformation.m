%% Description:
% CODE TO RUN THE NEURAL NETWORK ON GECKO DATA.
% 1. Loads the workspace containing the Field Alignment variables (qxs, qys, divVals etc.).
% 2. Creates input data for pairs of images with Field Alignment and image difference variables as input features.
% 3. Runs a Convolutional Neural Network on this Paired Input Data and gets accuracy on test pairs. 
%%

clear all;

indices = 1:251;

load('dispWorkspace6');

%%

[PairedData, PairedLabels] = getPairedData(imgStore, qxs, qys, divVals);

[trainIdxs, valIdxs, testIdxs] = dividerand(502, 0.6, 0.2, 0.2);

PairedTrainingData = PairedData(:, :, :, trainIdxs);
PairedTestData = PairedData(:, :, :, testIdxs);
PairedValidData = PairedData(:, :, :, valIdxs);

PairedTrainingLabels = PairedLabels(trainIdxs);
PairedTestLabels = PairedLabels(testIdxs);
PairedValidLabels = PairedLabels(valIdxs);

%%

layers = [
    imageInputLayer([64 64 2])
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

trainOptions = trainingOptions('sgdm', ...
    'MiniBatchSize', 30, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 2, ...
    'MaxEpochs', 80, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'cpu', ...
    'ValidationData', {PairedValidData, PairedValidLabels}, ...
    'Plots', 'training-progress');

trainedModel = trainNetwork(PairedTrainingData, PairedTrainingLabels, layers, trainOptions);

%%

YPred = classify(trainedModel, PairedTestData, 'ExecutionEnvironment', 'cpu');
accuracy = mean(YPred == PairedTestLabels);

%%

function [PairedData, PairedLabels] = getPairedData(trainingSet, qxs, qys, divVals)
    totalImages = 251;
    PairedData = zeros(64, 64, 2, 502);
    PairedLabels = zeros(502, 1);
    trainingLabels = trainingSet.Labels;
    for i = 1:totalImages
%         if mod(i, 2) ==  1
           thisLabel = trainingLabels(i);
           thisIndvIdx = i;
           sameIndvs = find(trainingLabels == thisLabel);
           sameIndvIdx = randperm(length(sameIndvs), 1);
           otherIndvs = find(trainingLabels ~= thisLabel);
           otherIndvIdx = randperm(length(otherIndvs), 1);
		
	   lapFilter = fspecial('laplacian', 0.2);
           
           img1 = imresize(rgb2gray(getNormalized(trainingSet, thisIndvIdx)), [64 64]);
           img2 = imresize(rgb2gray(getNormalized(trainingSet, sameIndvIdx)), [64 64]);
           img3 = imresize(rgb2gray(getNormalized(trainingSet, otherIndvIdx)), [64 64]);

           sm1 = getWeights(imresize(getNormalized(trainingSet, thisIndvIdx), [64 64]));
           sm2 = getWeights(imresize(getNormalized(trainingSet, sameIndvIdx), [64 64]));
           sm3 = getWeights(imresize(getNormalized(trainingSet, otherIndvIdx), [64 64]));
           

	   filImg1 = imfilter(img1, lapFilter);
	   filImg2 = imfilter(img2, lapFilter);
	   filImg3 = imfilter(img3, lapFilter);
%            k = (i + 1) / 2;
           
           % Same Label Pair
           divSame = zeros(64, 64);
	   %divSame(:, :) = (divergence(qxs(:, :, thisIndvIdx, sameIndvIdx), qys(:, :, thisIndvIdx, sameIndvIdx)) + divergence(qxs(:, :, sameIndvIdx, thisIndvIdx), qys(:, :, sameIndvIdx, thisIndvIdx))) / 2;
           if divVals(thisIndvIdx, sameIndvIdx) <= divVals(sameIndvIdx, thisIndvIdx)
               divSame(:, :) = divergence(qxs(:, :, thisIndvIdx, sameIndvIdx), qys(:, :, thisIndvIdx, sameIndvIdx));
           else
               divSame(:, :) = divergence(qxs(:, :, sameIndvIdx, thisIndvIdx), qys(:, :, sameIndvIdx, thisIndvIdx));
           end
           wvec = sm1 + sm2;
           wvec = wvec./sum(wvec);
%           wvec = 1 - wvec;
           divSame(:, :) = abs(repmat(wvec,[1 size(divSame, 2)]) .* divSame);
           diffSame = imabsdiff(img1, img2);
           %diffSame = imabsdiff(filImg1, filImg2);
	   PairedData(:, :, 1, 2 * i - 1) = divSame;
           PairedData(:, :, 2, 2 * i - 1) = diffSame;
           PairedLabels(2 * i - 1) = 1;
           
           % Other Label Pair
           divOther = zeros(64, 64);
           %divOther(:, :) = (divergence(qxs(:, :, thisIndvIdx, otherIndvIdx), qys(:, :, thisIndvIdx, otherIndvIdx)) + divergence(qxs(:, :, otherIndvIdx, thisIndvIdx), qys(:, :, otherIndvIdx, thisIndvIdx))) / 2;
	   if divVals(thisIndvIdx, otherIndvIdx) <= divVals(otherIndvIdx, thisIndvIdx)
               divOther(:, :) = divergence(qxs(:, :, thisIndvIdx, otherIndvIdx), qys(:, :, thisIndvIdx, otherIndvIdx));
           else
               divOther(:, :) = divergence(qxs(:, :, otherIndvIdx, thisIndvIdx), qys(:, :, otherIndvIdx, thisIndvIdx));
           end
           wvec = sm1 + sm3;
           wvec = wvec./sum(wvec);
%           wvec = 1 - wvec;
           divOther(:, :) = abs(repmat(wvec,[1 size(divOther, 2)]) .* divOther);
           diffOther = imabsdiff(img1, img3);
%           diffOther = imabsdiff(filImg1, filImg3);
 	   PairedData(:, :, 1, 2 * i) = divOther;
           PairedData(:, :, 2, 2 * i) = diffOther;
           
%         else
%            
%            k = i / 2; 
%             
%            thisLabel = trainingLabels(i);
%            thisIndvIdx = i;
%            otherIndvs = find(trainingLabels ~= thisLabel);
%            otherIndvIdx = randperm(length(otherIndvs), 1);
%            
%            img1 = imresize(rgb2gray(getNormalized(trainingSet, thisIndvIdx)), [64 64]);
%            img3 = imresize(rgb2gray(getNormalized(trainingSet, otherIndvIdx)), [64 64]);
%            
%            sm1 = getWeights(imresize(getNormalized(trainingSet, thisIndvIdx), [64 64]));
%            sm3 = getWeights(imresize(getNormalized(trainingSet, otherIndvIdx), [64 64]));
%            
%            % Other Label Pair
%            divOther = zeros(64, 64);
%            if divVals(thisIndvIdx, otherIndvIdx) <= divVals(otherIndvIdx, thisIndvIdx)
%                divOther(:, :) = divergence(qxs(:, :, thisIndvIdx, otherIndvIdx), qys(:, :, thisIndvIdx, otherIndvIdx));
%            else
%                divOther(:, :) = divergence(qxs(:, :, otherIndvIdx, thisIndvIdx), qys(:, :, otherIndvIdx, thisIndvIdx));
%            end
%            wvec = sm1 + sm3;
%            wvec = wvec./sum(wvec);
%            divOther(:, :) = abs(repmat(wvec,[1 size(divOther, 2)]) .* divOther);
%            diffOther = imabsdiff(img1, img3);
%            PairedData(:, :, 1, 3 * k) = divOther;
%            PairedData(:, :, 2, 3 * k) = diffOther;
%             
%         end 
    end
    PairedLabels = categorical(PairedLabels);
end

function sm1 = getWeights(image1)
    m1 = min(image1, [], 3);
    sm1 = sum(m1,2);
    lvec = 1:length(sm1);
    lvec = lvec(:);
    sm1=abs(sm1-polyval(polyfit(lvec,sm1,1),lvec));
    sm1 = sm1./sum(sm1);
end

function resImage = getNormalized(trainingSet, index)
    temp = readimage(trainingSet, index);
    summ = temp(:, :, 1) + temp(:, :, 2) + temp(:, :, 3);
    temp(:, :, 1) = temp(:, :, 1) ./ summ;
    temp(:, :, 2) = temp(:, :, 2) ./ summ;
    temp(:, :, 3) = temp(:, :, 3) ./ summ;
    resImage = temp;
end

function data = customReadFcn(filename)
    newImg = load(filename);
    data = newImg.icolor;
end
