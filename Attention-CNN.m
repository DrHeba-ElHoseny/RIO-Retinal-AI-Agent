%% =========================================================
%  Retinal Disease Classification with SE Attention (MATLAB)
%  - Backbone: ResNet-18
%  - SE attention head (channel gating) + classifier
%  - Optimizer: AdamW (or Adam fallback)
%  - During training: Confusion Matrix + ROC (Validation)
%  - Metrics per epoch: Accuracy, Precision, Recall, Jaccard
%% =========================================================
clc; clear; close all;

%% ============ 1) DATASET PATH (EDIT) ============
datasetRoot = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\1';

%% ============ 2) LOAD DATA ============
imds = imageDatastore(datasetRoot, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

classNames = categories(imds.Labels);
numClasses = numel(classNames);

disp("Classes found:"); disp(classNames);
tbl = countEachLabel(imds); disp(tbl);

%% ============ 3) SPLIT TRAIN/VAL/TEST ============
[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.70, 'randomized');
[imdsVal, imdsTest]   = splitEachLabel(imdsTemp, 0.50, 'randomized');

fprintf("Train: %d | Val: %d | Test: %d\n", numel(imdsTrain.Files), numel(imdsVal.Files), numel(imdsTest.Files));

%% ============ 4) BACKBONE ============
net = resnet18;
inputSize = net.Layers(1).InputSize; % [224 224 3]

%% ============ 5) AUGMENTATION ============
augmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-8 8], ...
    'RandYTranslation',[-8 8], ...
    'RandXReflection',true, ...
    'RandScale',[0.95 1.05]);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);
augTest  = augmentedImageDatastore(inputSize(1:2), imdsTest);

%% ============ 6) BUILD SE-ATTENTION HEAD ============
lgraph = layerGraph(net);

% Remove original classification head (ResNet-18)
layersToRemove = {'fc1000','prob','ClassificationLayer_predictions'};
lgraph = removeLayers(lgraph, layersToRemove);

featureChannels = 512;  % ResNet-18 pooled features channels
seRatio = 16;
seHidden = max(8, floor(featureChannels / seRatio)); % 32

seLayers = [
    fullyConnectedLayer(seHidden,'Name','se_fc1', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','se_relu')
    fullyConnectedLayer(featureChannels,'Name','se_fc2', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    sigmoidLayer('Name','se_sigmoid')
];

mulLayer = multiplicationLayer(2,'Name','se_multiply');

newHead = [
    fullyConnectedLayer(numClasses,'Name','new_fc', ...
        'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name','new_softmax')
    classificationLayer('Name','new_classoutput')
];

lgraph = addLayers(lgraph, seLayers);
lgraph = addLayers(lgraph, mulLayer);
lgraph = addLayers(lgraph, newHead);

lgraph = connectLayers(lgraph, 'pool5', 'se_fc1');
lgraph = connectLayers(lgraph, 'pool5', 'se_multiply/in1');
lgraph = connectLayers(lgraph, 'se_sigmoid', 'se_multiply/in2');
lgraph = connectLayers(lgraph, 'se_multiply', 'new_fc');

% (اختياري) افتحيه مرة للتأكد ثم علّقي السطر
% analyzeNetwork(lgraph);

%% ============ 7) LIVE METRICS SETUP (figures + logging) ============
% Figures to update each epoch
hFigCM  = figure('Name','Validation Confusion Matrix (Live)','NumberTitle','off');
hFigROC = figure('Name','Validation ROC (Live)','NumberTitle','off');
hFigMet = figure('Name','Validation Metrics (Live)','NumberTitle','off');

% Storage for metric history
histEpoch = [];
histAcc   = [];
histPrec  = [];
histRec   = [];
histJac   = [];

% For OutputFcn access
valFilesCount = numel(imdsVal.Files);
miniBatch = 16;
valFreq = max(1, floor(numel(imdsTrain.Files)/miniBatch)); % same as training options

%% ============ 8) TRAINING OPTIONS ============
% إذا adamw غير مدعوم في إصدارك، غيّريه إلى 'adam'
solverName = 'adamw';  % 'adamw' or 'adam'

opts = trainingOptions(solverName, ...
    'MiniBatchSize', miniBatch, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 3e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod', 6, ...
    'LearnRateDropFactor', 0.3, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', valFreq, ...
    'L2Regularization', 2e-4, ...
    'GradientThreshold', 1.0, ...
    'Verbose', true, ...
    'Plots','training-progress', ...
    'OutputFcn', @liveMetricsOutputFcn);

%% ============ 9) TRAIN ============
trainedNet = trainNetwork(augTrain, lgraph, opts);

%% ============ 10) FINAL TEST EVALUATION ============
[YPredTest, scoresTest] = classify(trainedNet, augTest);
YTrueTest = imdsTest.Labels;

[accT, precT, recT, jacT, cmT] = computeMetricsFromLabels(YTrueTest, YPredTest, classNames);

fprintf("\n===== FINAL TEST METRICS =====\n");
fprintf("Accuracy  : %.2f %%\n", accT*100);
fprintf("Precision : %.4f (macro)\n", precT);
fprintf("Recall    : %.4f (macro)\n", recT);
fprintf("Jaccard   : %.4f (macro)\n", jacT);

figure('Name','Test Confusion Matrix','NumberTitle','off');
confusionchart(YTrueTest, YPredTest);
title(sprintf("TEST Confusion Matrix | Acc %.2f%%", accT*100));

% Save artifacts for agent stage
results.model       = trainedNet;
results.classNames  = classNames;
results.testTrue    = YTrueTest;
results.testPred    = YPredTest;
results.testScores  = scoresTest;
results.testCM      = cmT;
results.testAcc     = accT;
results.testPrec    = precT;
results.testRec     = recT;
results.testJaccard = jacT;

results.trainHist.epoch = histEpoch;
results.trainHist.acc   = histAcc;
results.trainHist.prec  = histPrec;
results.trainHist.rec   = histRec;
results.trainHist.jac   = histJac;

savePath = fullfile(datasetRoot, "SE_Attention_ResNet18_results_with_metrics.mat");
save(savePath, "results");
disp("✅ Saved results to:"); disp(savePath);

%% =========================================================
%  OutputFcn: runs during training to update live plots
%% =========================================================
function stop = liveMetricsOutputFcn(info)
    stop = false;

    % Use base workspace variables (MATLAB nested function alternative)
    % We'll pull needed vars using evalin for simplicity in single-file code.
    persistent lastEpoch;

    if isempty(lastEpoch)
        lastEpoch = 0;
    end

    % Only update at end of each epoch (more stable & faster)
    if strcmp(info.State, "iteration")
        return;
    end

    if strcmp(info.State, "epoch")
        % Avoid duplicate updates
        if info.Epoch == lastEpoch
            return;
        end
        lastEpoch = info.Epoch;

        trainedNet = info.TrainedNetwork;

        augVal   = evalin('base','augVal');
        imdsVal  = evalin('base','imdsVal');
        classNames = evalin('base','classNames');

        hFigCM  = evalin('base','hFigCM');
        hFigROC = evalin('base','hFigROC');
        hFigMet = evalin('base','hFigMet');

        % Predict on validation
        [YValPred, scoresVal] = classify(trainedNet, augVal);
        YValTrue = imdsVal.Labels;

        % Compute metrics
        [acc, prec, rec, jac, cm] = computeMetricsFromLabels(YValTrue, YValPred, classNames);

        % ---- Update Confusion Matrix ----
        figure(hFigCM); clf;
        confusionchart(YValTrue, YValPred);
        title(sprintf("Validation Confusion Matrix | Epoch %d | Acc %.2f%%", info.Epoch, acc*100));
        drawnow;

        % ---- Update ROC Curves (One-vs-All) ----
        % Requires perfcurve (Statistics and Machine Learning Toolbox)
        figure(hFigROC); clf; hold on;
        title(sprintf("Validation ROC (One-vs-All) | Epoch %d", info.Epoch));
        xlabel("False Positive Rate"); ylabel("True Positive Rate");

        hasPerfcurve = exist('perfcurve','file') == 2;
        if hasPerfcurve
            for i = 1:numel(classNames)
                posClass = classNames{i};
                yTrueBin = (YValTrue == posClass);
                scorePos = scoresVal(:, i);
                [X, Y, ~, AUC] = perfcurve(yTrueBin, scorePos, true);
                plot(X, Y, 'LineWidth', 1.5);
                leg{i} = sprintf("%s (AUC=%.3f)", string(posClass), AUC); %#ok<AGROW>
            end
            legend(leg, 'Location','southeast');
        else
            text(0.05,0.5,"perfcurve not available. Install Statistics & ML Toolbox to plot ROC.",'FontSize',12);
        end
        grid on; hold off;
        drawnow;

        % ---- Update Metrics trend ----
        % Append history in base workspace arrays
        histEpoch = evalin('base','histEpoch');
        histAcc   = evalin('base','histAcc');
        histPrec  = evalin('base','histPrec');
        histRec   = evalin('base','histRec');
        histJac   = evalin('base','histJac');

        histEpoch(end+1) = info.Epoch;
        histAcc(end+1)   = acc;
        histPrec(end+1)  = prec;
        histRec(end+1)   = rec;
        histJac(end+1)   = jac;

        assignin('base','histEpoch',histEpoch);
        assignin('base','histAcc',histAcc);
        assignin('base','histPrec',histPrec);
        assignin('base','histRec',histRec);
        assignin('base','histJac',histJac);

        figure(hFigMet); clf;
        plot(histEpoch, histAcc,  '-o', 'LineWidth',1.5); hold on;
        plot(histEpoch, histPrec, '-o', 'LineWidth',1.5);
        plot(histEpoch, histRec,  '-o', 'LineWidth',1.5);
        plot(histEpoch, histJac,  '-o', 'LineWidth',1.5);
        grid on;
        xlabel("Epoch");
        ylabel("Metric Value");
        title("Validation Metrics over Epochs");
        legend({'Accuracy','Precision (macro)','Recall (macro)','Jaccard (macro)'}, 'Location','best');
        drawnow;

        % Print metrics to command window
        fprintf("Epoch %d | Val Acc %.2f%% | Prec %.4f | Rec %.4f | Jac %.4f\n", ...
            info.Epoch, acc*100, prec, rec, jac);
    end

    if strcmp(info.State, "done")
        % final state
        return;
    end
end

%% =========================================================
%  Metrics from labels (Accuracy, Precision, Recall, Jaccard)
%  - Macro-averaged precision/recall/jaccard across classes
%% =========================================================
function [acc, precMacro, recMacro, jacMacro, cm] = computeMetricsFromLabels(YTrue, YPred, classNames)
    cm = confusionmat(YTrue, YPred, 'Order', categorical(classNames));

    TP = diag(cm);
    FP = sum(cm,1)' - TP;
    FN = sum(cm,2)  - TP;

    acc = sum(TP) / max(1,sum(cm(:)));

    precision = TP ./ max(1,(TP + FP));
    recall    = TP ./ max(1,(TP + FN));
    jaccard   = TP ./ max(1,(TP + FP + FN));

    precMacro = mean(precision, 'omitnan');
    recMacro  = mean(recall,    'omitnan');
    jacMacro  = mean(jaccard,   'omitnan');
end
