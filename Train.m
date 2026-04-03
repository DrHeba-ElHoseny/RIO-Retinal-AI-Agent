 %% Retina_CNN_Train_Evaluate_Final.m
% ------------------------------------------------------------
% FINAL, GUARANTEED WORKING (No pretrained models needed)
% 4-class retinal fundus classification:
% Cataract, Diabetic_Retinopathy, Glaucoma, Normal
%
% Includes:
% - Load dataset from folders
% - Train/Val/Test split
% - Augmentation
% - Train a custom CNN (from scratch)
% - Evaluation table (Precision/Recall/F1 + Macro/Weighted)
% - Confusion matrix (chart + numeric)
% - ROC curves one-vs-all + AUC
% - Save model as retina_classification_model.mat (agentModel)
%
% Requirements:
% - Deep Learning Toolbox
% ------------------------------------------------------------

clear; clc; close all;
rng(42);

%% =======================
% USER SETTINGS
% ========================
datasetPath = "F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\RetinalDataset";  % <<< CHANGE THIS
inputSize   = [224 224 3];                  % you can reduce to [128 128 3] if PC is slow
trainRatio  = 0.70;
valRatio    = 0.15;
testRatio   = 0.15;

miniBatchSize     = 16;
maxEpochs         = 25;
initialLearnRate  = 1e-3;   % from scratch often benefits from higher LR
validationFreq    = 50;

%% =======================
% 1) LOAD DATASET
% ========================
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp("Dataset loaded:");
disp(countEachLabel(imds));

% Enforce fixed label order (recommended)
desiredOrder = ["Cataract","Diabetic_Retinopathy","Glaucoma","Normal"];
imds.Labels = categorical(string(imds.Labels), desiredOrder, desiredOrder);

classNames = categories(imds.Labels);
numClasses = numel(classNames);
if numClasses ~= 4
    error("Expected 4 classes, found %d. Check folder names.", numClasses);
end

%% =======================
% 2) SPLIT: TRAIN / VAL / TEST
% ========================
[imdsTrain, imdsTmp] = splitEachLabel(imds, trainRatio, 'randomized');
valShareOfTmp = valRatio / (valRatio + testRatio);
[imdsVal, imdsTest]  = splitEachLabel(imdsTmp, valShareOfTmp, 'randomized');

disp("Split counts:");
disp("Train:"); disp(countEachLabel(imdsTrain));
disp("Val  :"); disp(countEachLabel(imdsVal));
disp("Test :"); disp(countEachLabel(imdsTest));

%% =======================
% 3) AUGMENTATION + DATASTORES
% ========================
augmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXReflection',true, ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing','gray2rgb');

augVal   = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing','gray2rgb');

augTest  = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing','gray2rgb');

%% =======================
% 4) BUILD CUSTOM CNN (WORKS EVERYWHERE)
% ========================
layers = [
    imageInputLayer(inputSize, 'Name','input', 'Normalization','zerocenter')

    convolution2dLayer(3, 32, 'Padding','same', 'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool1')

    convolution2dLayer(3, 64, 'Padding','same', 'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool2')

    convolution2dLayer(3, 128, 'Padding','same', 'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool3')

    convolution2dLayer(3, 256, 'Padding','same', 'Name','conv4')
    batchNormalizationLayer('Name','bn4')
    reluLayer('Name','relu4')

    globalAveragePooling2dLayer('Name','gap')

    dropoutLayer(0.30, 'Name','dropout')

    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

%% =======================
% 5) TRAIN
% ========================
options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', maxEpochs, ...
    'InitialLearnRate', initialLearnRate, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', validationFreq, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

disp("Training started...");
trainedNet = trainNetwork(augTrain, layers, options);
disp("Training done.");

%% =======================
% 6) TEST PREDICTION (labels + scores)
% ========================
[YPred, scores] = classify(trainedNet, augTest);
YTrue = imdsTest.Labels;

overallAcc = mean(YPred == YTrue);
fprintf("\nOverall Test Accuracy = %.4f\n", overallAcc);

%% =======================
% 7) CONFUSION MATRIX (chart + numeric)
% ========================
figure('Name','Confusion Matrix');
cmChart = confusionchart(YTrue, YPred);
cmChart.Title = 'Confusion Matrix (Test Set)';
cmChart.RowSummary = 'row-normalized';
cmChart.ColumnSummary = 'column-normalized';

[cm, ~] = confusionmat(YTrue, YPred, 'Order', categorical(classNames));
disp("Confusion Matrix (rows=True, cols=Pred) in classNames order:");
disp(array2table(cm, ...
    'VariableNames', strcat("Pred_", string(classNames)), ...
    'RowNames', strcat("True_", string(classNames))));

%% =======================
% 8) EVALUATION TABLE (Precision/Recall/F1 + Macro/Weighted)
% ========================
metricsTable = computeClassificationMetrics(cm, classNames);
macroRow    = computeMacroAverage(metricsTable);
weightedRow = computeWeightedAverage(metricsTable);

summaryTable = [metricsTable; macroRow; weightedRow];

disp("Evaluation Table (per-class + Macro/Weighted):");
disp(summaryTable);

%% =======================
% 9) ROC CURVES (One-vs-All) + AUC
% ========================
YTrueStr = string(YTrue);
classStr = string(classNames);

figure('Name','ROC Curves (One-vs-All)');
hold on; grid on;
aucList = zeros(numClasses,1);

for i = 1:numClasses
    posClass = classStr(i);
    yBin = double(YTrueStr == posClass);  % 1 for positive, 0 for others
    s = scores(:, i);                    % score/prob for that class

    [Xroc, Yroc, ~, AUC] = perfcurve(yBin, s, 1);
    aucList(i) = AUC;

    plot(Xroc, Yroc, 'LineWidth', 1.5);
end

xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves (One-vs-All) on Test Set');
legend(compose("%s (AUC=%.3f)", classStr, aucList), 'Location', 'SouthEast');
hold off;

aucTable = table(classNames, aucList, 'VariableNames', {'Class','AUC'});
disp("AUC per class (One-vs-All):");
disp(aucTable);

%% =======================
% 10) SAVE MODEL FOR YOUR AI AGENT
% ========================
agentModel.net = trainedNet;
agentModel.classNames = classNames;
save("retina_classification_model.mat", "agentModel");
disp("Saved model: retina_classification_model.mat");

%% =======================
% LOCAL FUNCTIONS
% ========================
function T = computeClassificationMetrics(cm, classNames)
    support = sum(cm,2);
    TP = diag(cm);
    FP = sum(cm,1)' - TP;
    FN = sum(cm,2)  - TP;
    TN = sum(cm(:)) - (TP + FP + FN);

    precision   = safeDiv(TP, (TP + FP));
    recall      = safeDiv(TP, (TP + FN));
    f1          = safeDiv(2 * precision .* recall, (precision + recall));
    specificity = safeDiv(TN, (TN + FP));
    perClassAcc = safeDiv(TP + TN, TP + TN + FP + FN);

    T = table(string(classNames(:)), support, precision, recall, f1, specificity, perClassAcc, ...
        'VariableNames', {'Class','Support','Precision','Recall','F1','Specificity','PerClassAccuracy'});
end

function row = computeMacroAverage(T)
    row = T(1,:);
    row.Class = "MacroAvg";
    row.Support = sum(T.Support);
    row.Precision = mean(T.Precision, 'omitnan');
    row.Recall    = mean(T.Recall, 'omitnan');
    row.F1        = mean(T.F1, 'omitnan');
    row.Specificity = mean(T.Specificity, 'omitnan');
    row.PerClassAccuracy = mean(T.PerClassAccuracy, 'omitnan');
end

function row = computeWeightedAverage(T)
    w = T.Support / sum(T.Support);
    row = T(1,:);
    row.Class = "WeightedAvg";
    row.Support = sum(T.Support);
    row.Precision = sum(w .* T.Precision, 'omitnan');
    row.Recall    = sum(w .* T.Recall, 'omitnan');
    row.F1        = sum(w .* T.F1, 'omitnan');
    row.Specificity = sum(w .* T.Specificity, 'omitnan');
    row.PerClassAccuracy = sum(w .* T.PerClassAccuracy, 'omitnan');
end

function y = safeDiv(a,b)
    y = zeros(size(a), 'like', single(1));
    idx = (b ~= 0);
    y(idx) = a(idx) ./ b(idx);
    y(~idx) = NaN;
end
