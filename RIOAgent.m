 function RIOAgent
% RIOAgent.m  (ONE FILE UI - FIXED EVALUATION for MATLAB R2021a)
% ==============================================================
% RIO Fine-Tuning + Inference AI Agent (UI)
% - Stage A + Stage B training
% - Evaluation (Confusion + ROC/AUC + Metrics) works reliably
% - Export CSV (A+B) + Comparison
% - Single image inference + policy + logging
%
% Fixes applied:
% - Removed dynamic field names S.(['cm_' tag]) to avoid R2021a errors
% - Uses stable storage: S.eval.StageA / S.eval.StageB
% - Export checks evaluation existence before comparison
% ==============================================================

clc; close all;
rng(42);

%% -------------------------
% App State (Agent memory)
%% -------------------------
S = struct();
S.datasetPath = "";
S.runDir = "";
S.classOrder = ["Cataract","Diabetic_Retinopathy","Glaucoma","Normal"];

S.netA = [];
S.netB = [];
S.classNames = string.empty;
S.imdsTrain = [];
S.imdsVal = [];
S.imdsTest = [];
S.augTrain = [];
S.augVal = [];
S.augTest = [];

% Evaluation storage (FIX)
S.eval = struct();
S.eval.StageA = struct('cm',[],'auc',[],'acc',NaN,'metrics',[]);
S.eval.StageB = struct('cm',[],'auc',[],'acc',NaN,'metrics',[]);

% Defaults (editable from UI)
S.cfg.inputSize = [224 224 3];
S.cfg.trainRatio = 0.70;
S.cfg.valRatio   = 0.15;
S.cfg.testRatio  = 0.15;
S.cfg.miniBatchSize = 16;
S.cfg.validationFreq = 50;

S.cfg.stageA.maxEpochs = 30;
S.cfg.stageA.lr = 1e-3;
S.cfg.stageA.dropPeriod = 6;
S.cfg.stageA.dropFactor = 0.2;

S.cfg.stageB.maxEpochs = 12;
S.cfg.stageB.lr = 1e-4;

S.cfg.dropout = 0.35;
S.cfg.l2 = 1e-4;

% QC thresholds (for inference)
S.qc.lowBright = 60;
S.qc.highBright = 200;
S.qc.minContrast = 25;
S.qc.minSharpness = 50;

% Policy thresholds (for inference)
S.pol.thrDefault = 0.75;
S.pol.thrGlaucoma = 0.80;
S.pol.thrMinReview = 0.60;

S.modelFile = "";
S.logFile = "";

%% -------------------------
% Build UI
%% -------------------------
fig = uifigure('Name','RIO Fine-Tuning + Inference AI Agent', 'Position',[100 80 1200 720]);

gl = uigridlayout(fig,[1 1]);
gl.RowHeight = {'1x'}; gl.ColumnWidth = {'1x'};

tabs = uitabgroup(gl);

tabTrain = uitab(tabs,'Title','Fine-Tuning Agent');
tabInfer = uitab(tabs,'Title','Single Image Inference');

%% =========================
% TAB 1: Fine-tuning Agent UI
%% =========================
g1 = uigridlayout(tabTrain,[8 12]);
g1.RowHeight = {34,34,34,34,'1x','1x',34,34};
g1.ColumnWidth = {160,160,160,160,160,160,'1x','1x','1x','1x','1x','1x'};

btnSelectDS = uibutton(g1,'Text','Select Dataset Folder','ButtonPushedFcn',@onSelectDataset);
btnSelectDS.Layout.Row = 1; btnSelectDS.Layout.Column = [1 2];

btnRunDir = uibutton(g1,'Text','Select Output Folder','ButtonPushedFcn',@onSelectRunDir);
btnRunDir.Layout.Row = 1; btnRunDir.Layout.Column = [3 4];

lblDS = uilabel(g1,'Text','Dataset: (not selected)');
lblDS.Layout.Row = 1; lblDS.Layout.Column = [5 12];

lblOut = uilabel(g1,'Text','Output: (auto)');
lblOut.Layout.Row = 2; lblOut.Layout.Column = [5 12];

btnPrepare = uibutton(g1,'Text','Prepare (Split + Augment)','ButtonPushedFcn',@onPrepare);
btnPrepare.Layout.Row = 2; btnPrepare.Layout.Column = [1 2];

btnStageA = uibutton(g1,'Text','Run Stage A','ButtonPushedFcn',@onRunStageA, 'Enable','off');
btnStageA.Layout.Row = 2; btnStageA.Layout.Column = [3 4];

btnStageB = uibutton(g1,'Text','Run Stage B','ButtonPushedFcn',@onRunStageB, 'Enable','off');
btnStageB.Layout.Row = 2; btnStageB.Layout.Column = [5 6];

btnEvalA = uibutton(g1,'Text','Eval Stage A','ButtonPushedFcn',@onEvalA, 'Enable','off');
btnEvalA.Layout.Row = 3; btnEvalA.Layout.Column = [1 2];

btnEvalB = uibutton(g1,'Text','Eval Stage B','ButtonPushedFcn',@onEvalB, 'Enable','off');
btnEvalB.Layout.Row = 3; btnEvalB.Layout.Column = [3 4];

btnExport = uibutton(g1,'Text','Export CSV (A+B)','ButtonPushedFcn',@onExport, 'Enable','off');
btnExport.Layout.Row = 3; btnExport.Layout.Column = [5 6];

% Settings panel (compact)
p = uipanel(g1,'Title','Quick Settings');
p.Layout.Row = [1 4]; p.Layout.Column = [7 12];
gp = uigridlayout(p,[4 6]);
gp.RowHeight = {22,22,22,22};
gp.ColumnWidth = {100,80,100,80,100,80};

uilabel(gp,'Text','Input');    ddInput = uidropdown(gp,'Items',{'224','128'},'Value','224');
uilabel(gp,'Text','Batch');    edBatch = uieditfield(gp,'numeric','Value',S.cfg.miniBatchSize,'Limits',[1 256]);
uilabel(gp,'Text','EpA');      edEpA   = uieditfield(gp,'numeric','Value',S.cfg.stageA.maxEpochs,'Limits',[1 500]);
uilabel(gp,'Text','LR A');     edLRA   = uieditfield(gp,'numeric','Value',S.cfg.stageA.lr,'Limits',[1e-6 1]);
uilabel(gp,'Text','EpB');      edEpB   = uieditfield(gp,'numeric','Value',S.cfg.stageB.maxEpochs,'Limits',[0 500]);
uilabel(gp,'Text','LR B');     edLRB   = uieditfield(gp,'numeric','Value',S.cfg.stageB.lr,'Limits',[1e-6 1]);

ddInput.Layout.Row=1; ddInput.Layout.Column=2;
edBatch.Layout.Row=1; edBatch.Layout.Column=4;
edEpA.Layout.Row=1; edEpA.Layout.Column=6;

edLRA.Layout.Row=2; edLRA.Layout.Column=2;
edEpB.Layout.Row=2; edEpB.Layout.Column=4;
edLRB.Layout.Row=2; edLRB.Layout.Column=6;

% Axes + tables
axCM = uiaxes(g1); axCM.Layout.Row = [5 6]; axCM.Layout.Column = [1 6];
title(axCM,'Confusion Matrix (heatmap)');

axROC = uiaxes(g1); axROC.Layout.Row = [5 6]; axROC.Layout.Column = [7 12];
title(axROC,'ROC Curves');

tblMetrics = uitable(g1); tblMetrics.Layout.Row = 7; tblMetrics.Layout.Column = [1 12];
tblMetrics.ColumnName = {'Class','Support','Precision','Recall','F1','Specificity','PerClassAcc'};
tblMetrics.Data = cell(0,7);

txtStatus = uilabel(g1,'Text','Status: Select dataset folder.'); 
txtStatus.Layout.Row = 8; txtStatus.Layout.Column = [1 12];

%% =========================
% TAB 2: Single Image Inference UI
%% =========================
g2 = uigridlayout(tabInfer,[7 12]);
g2.RowHeight = {34,34,34,'1x','1x',34,34};
g2.ColumnWidth = {160,160,160,160,160,160,'1x','1x','1x','1x','1x','1x'};

btnLoadModel = uibutton(g2,'Text','Load Model (.mat)','ButtonPushedFcn',@onLoadModel);
btnLoadModel.Layout.Row = 1; btnLoadModel.Layout.Column = [1 2];

btnPickImg = uibutton(g2,'Text','Select Image','ButtonPushedFcn',@onPickImage,'Enable','off');
btnPickImg.Layout.Row = 1; btnPickImg.Layout.Column = [3 4];

lblModel = uilabel(g2,'Text','Model: (not loaded)');
lblModel.Layout.Row = 1; lblModel.Layout.Column = [5 12];

lblImg = uilabel(g2,'Text','Image: (not selected)');
lblImg.Layout.Row = 2; lblImg.Layout.Column = [5 12];

axImg = uiaxes(g2); axImg.Layout.Row = [4 5]; axImg.Layout.Column = [1 6];
title(axImg,'Input Image');

tblScores = uitable(g2); tblScores.Layout.Row = [4 5]; tblScores.Layout.Column = [7 12];
tblScores.ColumnName = {'Class','Score','Score%'}; tblScores.Data = cell(0,3);

lblPred = uilabel(g2,'Text','Prediction: -'); lblPred.Layout.Row = 6; lblPred.Layout.Column = [1 6];
lblQual = uilabel(g2,'Text','Quality: -');    lblQual.Layout.Row = 6; lblQual.Layout.Column = [7 12];
lblDec  = uilabel(g2,'Text','Policy Decision: -'); lblDec.Layout.Row = 7; lblDec.Layout.Column = [1 12];

%% -------------------------
% Callbacks (Agent actions)
%% -------------------------
    function syncSettingsFromUI()
        v = str2double(ddInput.Value);
        S.cfg.inputSize = [v v 3];
        S.cfg.miniBatchSize = edBatch.Value;
        S.cfg.stageA.maxEpochs = edEpA.Value;
        S.cfg.stageA.lr = edLRA.Value;
        S.cfg.stageB.maxEpochs = edEpB.Value;
        S.cfg.stageB.lr = edLRB.Value;
    end

    function onSelectDataset(~,~)
        folder = uigetdir(pwd,'Select dataset root folder (contains class subfolders)');
        if isequal(folder,0), return; end
        S.datasetPath = string(folder);
        lblDS.Text = "Dataset: " + S.datasetPath;
        txtStatus.Text = "Status: Dataset selected. Select output folder (optional) then Prepare.";
    end

    function onSelectRunDir(~,~)
        folder = uigetdir(pwd,'Select output folder (run folder will be created inside)');
        if isequal(folder,0), return; end
        base = string(folder);
        runID = "RIO_Run_" + string(datetime("now",'Format','yyyyMMdd_HHmmss'));
        S.runDir = string(fullfile(base,runID));
        if ~isfolder(S.runDir), mkdir(S.runDir); end
        S.logFile = string(fullfile(S.runDir,"RIO_SingleImage_Log.csv"));
        lblOut.Text = "Output: " + S.runDir;
        txtStatus.Text = "Status: Output folder set.";
    end

    function ensureRunDir()
        if strlength(S.runDir)==0
            runID = "RIO_Run_" + string(datetime("now",'Format','yyyyMMdd_HHmmss'));
            S.runDir = string(fullfile(pwd,runID));
            if ~isfolder(S.runDir), mkdir(S.runDir); end
            S.logFile = string(fullfile(S.runDir,"RIO_SingleImage_Log.csv"));
            lblOut.Text = "Output: " + S.runDir;
        end
    end

    function onPrepare(~,~)
        syncSettingsFromUI();
        ensureRunDir();

        if ~isfolder(S.datasetPath)
            uialert(fig,'Please select a valid dataset folder.','Dataset missing'); return;
        end

        txtStatus.Text = "Status: Preparing datastore, split, augmentation...";
        drawnow;

        imds = imageDatastore(S.datasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
        imds.Labels = categorical(string(imds.Labels), S.classOrder, S.classOrder);

        cn = string(categories(imds.Labels));
        if numel(cn) ~= 4
            uialert(fig,'Expected 4 class folders. Check folder names.','Class mismatch'); return;
        end
        S.classNames = cn;

        [imdsTrain, imdsTmp] = splitEachLabel(imds, S.cfg.trainRatio, 'randomized');
        valShareOfTmp = S.cfg.valRatio / (S.cfg.valRatio + S.cfg.testRatio);
        [imdsVal, imdsTest] = splitEachLabel(imdsTmp, valShareOfTmp, 'randomized');

        S.imdsTrain = imdsTrain; S.imdsVal = imdsVal; S.imdsTest = imdsTest;

        augmenter = imageDataAugmenter('RandRotation',[-10 10],'RandXReflection',true,...
            'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

        S.augTrain = augmentedImageDatastore(S.cfg.inputSize, imdsTrain,'DataAugmentation',augmenter,'ColorPreprocessing','gray2rgb');
        S.augVal   = augmentedImageDatastore(S.cfg.inputSize, imdsVal,'ColorPreprocessing','gray2rgb');
        S.augTest  = augmentedImageDatastore(S.cfg.inputSize, imdsTest,'ColorPreprocessing','gray2rgb');

        btnStageA.Enable = 'on';
        txtStatus.Text = "Status: Prepared. You can run Stage A.";
    end

    function buildCNN()
        tbl = countEachLabel(S.imdsTrain);
        counts = tbl.Count;
        w = sum(counts) ./ (numel(counts)*counts);
        w = w / mean(w);

        numClasses = numel(S.classNames);

        S.layers = [
            imageInputLayer(S.cfg.inputSize,'Name','input','Normalization','zerocenter')

            convolution2dLayer(3,32,'Padding','same','Name','conv1')
            batchNormalizationLayer('Name','bn1')
            reluLayer('Name','relu1')
            maxPooling2dLayer(2,'Stride',2,'Name','pool1')

            convolution2dLayer(3,64,'Padding','same','Name','conv2')
            batchNormalizationLayer('Name','bn2')
            reluLayer('Name','relu2')
            maxPooling2dLayer(2,'Stride',2,'Name','pool2')

            convolution2dLayer(3,128,'Padding','same','Name','conv3')
            batchNormalizationLayer('Name','bn3')
            reluLayer('Name','relu3')
            maxPooling2dLayer(2,'Stride',2,'Name','pool3')

            convolution2dLayer(3,256,'Padding','same','Name','conv4')
            batchNormalizationLayer('Name','bn4')
            reluLayer('Name','relu4')

            globalAveragePooling2dLayer('Name','gap')
            dropoutLayer(S.cfg.dropout,'Name','drop')

            fullyConnectedLayer(numClasses,'Name','fc')
            softmaxLayer('Name','sm')
            classificationLayer('Name','out','Classes',categorical(S.classNames),'ClassWeights',w)
        ];
    end

    function onRunStageA(~,~)
        syncSettingsFromUI();
        ensureRunDir();
        if isempty(S.augTrain), uialert(fig,'Click Prepare first.','Not ready'); return; end

        txtStatus.Text = "Status: Building CNN + running Stage A...";
        drawnow;

        buildCNN();

        ckptA = fullfile(S.runDir,"checkpoints_stageA"); if ~isfolder(ckptA), mkdir(ckptA); end
        optsA = trainingOptions('adam',...
            'MiniBatchSize',S.cfg.miniBatchSize,...
            'MaxEpochs',S.cfg.stageA.maxEpochs,...
            'InitialLearnRate',S.cfg.stageA.lr,...
            'LearnRateSchedule','piecewise',...
            'LearnRateDropFactor',S.cfg.stageA.dropFactor,...
            'LearnRateDropPeriod',S.cfg.stageA.dropPeriod,...
            'L2Regularization',S.cfg.l2,...
            'Shuffle','every-epoch',...
            'ValidationData',S.augVal,...
            'ValidationFrequency',S.cfg.validationFreq,...
            'CheckpointPath',ckptA,...
            'Verbose',true,...
            'Plots','training-progress');

        S.netA = trainNetwork(S.augTrain, S.layers, optsA);

        btnEvalA.Enable = 'on';
        btnStageB.Enable = 'on';
        txtStatus.Text = "Status: Stage A done. You can Eval Stage A or run Stage B.";
    end

    function onRunStageB(~,~)
        syncSettingsFromUI();
        ensureRunDir();
        if isempty(S.netA), uialert(fig,'Run Stage A first.','Not ready'); return; end

        txtStatus.Text = "Status: Running Stage B (polish)...";
        drawnow;

        ckptB = fullfile(S.runDir,"checkpoints_stageB"); if ~isfolder(ckptB), mkdir(ckptB); end
        optsB = trainingOptions('adam',...
            'MiniBatchSize',S.cfg.miniBatchSize,...
            'MaxEpochs',S.cfg.stageB.maxEpochs,...
            'InitialLearnRate',S.cfg.stageB.lr,...
            'L2Regularization',S.cfg.l2,...
            'Shuffle','every-epoch',...
            'ValidationData',S.augVal,...
            'ValidationFrequency',S.cfg.validationFreq,...
            'CheckpointPath',ckptB,...
            'Verbose',true,...
            'Plots','training-progress');

        try
            layersB = S.netA.Layers;  % carries learned weights
            S.netB = trainNetwork(S.augTrain, layersB, optsB);
        catch ME
            S.netB = S.netA;
            uialert(fig, "Stage B continuation not supported in your MATLAB setup. Using Stage A as final.\n\n" + ME.message, ...
                'Stage B Fallback');
        end

        btnEvalB.Enable = 'on';
        btnExport.Enable = 'on';

        agentModel = struct();
        agentModel.net = S.netB;
        agentModel.classNames = S.classNames;
        save(fullfile(S.runDir,"retina_classification_model.mat"),"agentModel");
        S.modelFile = string(fullfile(S.runDir,"retina_classification_model.mat"));
        lblModel.Text = "Model: " + S.modelFile;
        btnPickImg.Enable = 'on';

        txtStatus.Text = "Status: Stage B done. Model saved. Eval Stage B then Export.";
    end

    function [cm, auc, acc, metricsData] = evalCompute(net)
        YTrue = S.imdsTest.Labels;
        [YPred, scores] = classify(net, S.augTest);
        acc = mean(YPred == YTrue);

        [cm,~] = confusionmat(YTrue, YPred, 'Order', categorical(S.classNames));

        TP = diag(cm); FP = sum(cm,1)' - TP; FN = sum(cm,2) - TP; TN = sum(cm(:)) - (TP+FP+FN);
        precision = TP ./ (TP + FP); precision((TP+FP)==0)=NaN;
        recall    = TP ./ (TP + FN); recall((TP+FN)==0)=NaN;
        f1        = (2*precision.*recall) ./ (precision+recall); f1((precision+recall)==0)=NaN;
        spec      = TN ./ (TN + FP); spec((TN+FP)==0)=NaN;
        pacc      = (TP+TN) ./ (TP+TN+FP+FN); pacc((TP+TN+FP+FN)==0)=NaN;
        support   = sum(cm,2);

        macro = [mean(precision,'omitnan') mean(recall,'omitnan') mean(f1,'omitnan') mean(spec,'omitnan') mean(pacc,'omitnan')];
        w = support/sum(support);
        weighted = [sum(w.*precision,'omitnan') sum(w.*recall,'omitnan') sum(w.*f1,'omitnan') sum(w.*spec,'omitnan') sum(w.*pacc,'omitnan')];

        metricsData = [cellstr(S.classNames(:)), num2cell(support), num2cell(precision), num2cell(recall), num2cell(f1), num2cell(spec), num2cell(pacc)];
        metricsData = [metricsData; {'MacroAvg', sum(support), macro(1), macro(2), macro(3), macro(4), macro(5)}];
        metricsData = [metricsData; {'WeightedAvg', sum(support), weighted(1), weighted(2), weighted(3), weighted(4), weighted(5)}];

        % ROC/AUC
        YTrueStr = string(YTrue);
        auc = zeros(numel(S.classNames),1);
        for i=1:numel(S.classNames)
            yBin = double(YTrueStr == S.classNames(i));
            [~,~,~,AUC] = perfcurve(yBin, scores(:,i), 1);
            auc(i)=AUC;
        end
    end

    function evalAndShow(net, tag)
        [cm, auc, acc, metricsData] = evalCompute(net);

        % Show CM heatmap
        % Show CM heatmap (FIX for R2021a tick labels)
% cla(axCM);
% imagesc(axCM, cm);
% axis(axCM,'tight');  % ensures full matrix is shown
% 
% K = numel(S.classNames);
% axCM.XLim = [0.5 K+0.5];
% axCM.YLim = [0.5 K+0.5];
% 
% axCM.XTick = 1:K;
% axCM.YTick = 1:K;
% 
% % IMPORTANT: convert labels to cellstr for R2021a
% axCM.XTickLabel = cellstr(S.classNames);
% axCM.YTickLabel = cellstr(S.classNames);
% 
% xlabel(axCM,'Predicted');
% ylabel(axCM,'True');
% title(axCM, [char(tag) ' Confusion (counts)']);
% colorbar(axCM);
% 
% --- Confusion Matrix in UI (Counts + Numbers Overlay) ---
cla(axCM);

cmCounts = cm;

% Row-normalized (كل صف مجموع=1)
rowSum = sum(cmCounts,2);
cmRow = cmCounts ./ max(rowSum,1);

imagesc(axCM, cmRow);           % عرض كنِسَب (0..1)
colormap(axCM, parula);         % الافتراضي
caxis(axCM,[0 1]);              % تثبيت مقياس الألوان
colorbar(axCM);

K = numel(S.classNames);
axCM.XLim = [0.5 K+0.5];
axCM.YLim = [0.5 K+0.5];
axCM.XTick = 1:K; axCM.YTick = 1:K;
axCM.XTickLabel = cellstr(S.classNames);
axCM.YTickLabel = cellstr(S.classNames);

xlabel(axCM,'Predicted');
ylabel(axCM,'True');
title(axCM, [char(tag) ' Confusion (Row-Normalized)']);

% كتابة الأرقام (counts + percent) داخل كل خلية
for r = 1:K
    for c = 1:K
        cnt = cmCounts(r,c);
        pct = 100*cmRow(r,c);
        txt = sprintf('%d\n(%.1f%%)', cnt, pct);
        text(axCM, c, r, txt, 'HorizontalAlignment','center', ...
            'FontSize',10, 'Color','w', 'FontWeight','bold');
    end
end

        % Show ROC curves (need scores -> recompute for curves)
        cla(axROC);
        hold(axROC,'on'); grid(axROC,'on');
        YTrue = S.imdsTest.Labels;
        [~, scores] = classify(net, S.augTest);
        YTrueStr = string(YTrue);

        for i=1:numel(S.classNames)
            yBin = double(YTrueStr == S.classNames(i));
            [Xroc, Yroc] = perfcurve(yBin, scores(:,i), 1);
            plot(axROC, Xroc, Yroc, 'LineWidth', 1.5);
        end
        xlabel(axROC,'FPR'); ylabel(axROC,'TPR'); title(axROC, char(tag) + " ROC (One-vs-All)");
 legend(axROC, cellstr(S.classNames), 'Location','SouthEast');
        hold(axROC,'off');

        % Metrics table
        tblMetrics.Data = metricsData;

        % Store safely (FIXED)
        tagChar = char(tag);
        if strcmpi(tagChar,'StageA')
            S.eval.StageA.cm = cm; S.eval.StageA.auc = auc; S.eval.StageA.acc = acc; S.eval.StageA.metrics = metricsData;
        elseif strcmpi(tagChar,'StageB')
            S.eval.StageB.cm = cm; S.eval.StageB.auc = auc; S.eval.StageB.acc = acc; S.eval.StageB.metrics = metricsData;
        end

        txtStatus.Text = "Status: " + string(tag) + " evaluation done.";
        btnExport.Enable = 'on';
    end

    function onEvalA(~,~)
        if isempty(S.netA), uialert(fig,'Run Stage A first.','Not ready'); return; end
        evalAndShow(S.netA, "StageA");
    end

    function onEvalB(~,~)
        if isempty(S.netB), uialert(fig,'Run Stage B first.','Not ready'); return; end
        evalAndShow(S.netB, "StageB");
    end

    function onExport(~,~)
        ensureRunDir();

        if isempty(S.netA) && isempty(S.netB)
            uialert(fig,'No models found. Run Stage A/B first.','Not ready'); return;
        end

        % Require evaluation for export (to avoid missing fields)
        if isempty(S.eval.StageA.cm) && ~isempty(S.netA)
            uialert(fig,'Please click "Eval Stage A" first, then Export.','Missing StageA eval'); return;
        end
        if isempty(S.eval.StageB.cm) && ~isempty(S.netB)
            uialert(fig,'Please click "Eval Stage B" first, then Export.','Missing StageB eval'); return;
        end

        if ~isempty(S.netA), exportOne("StageA", S.netA, S.eval.StageA); end
        if ~isempty(S.netB), exportOne("StageB", S.netB, S.eval.StageB); end

        % Comparison if both exist
        if ~isempty(S.netA) && ~isempty(S.netB) && ~isempty(S.eval.StageA.auc) && ~isempty(S.eval.StageB.auc)
            Comp = table();
            Comp.Metric = ["OverallAccuracy"; "AUC_"+S.classNames(:)];
            Comp.StageA = [S.eval.StageA.acc; S.eval.StageA.auc];
            Comp.StageB = [S.eval.StageB.acc; S.eval.StageB.auc];
            Comp.Delta  = Comp.StageB - Comp.StageA;
            writetable(Comp, fullfile(S.runDir,"StageA_vs_StageB_Comparison.csv"));
        end

        uialert(fig,"CSV exported to: " + S.runDir,'Export Done');
    end

    function exportOne(tag, net, E)
        % Confusion + Metrics + AUC exported from computed eval E
        cm = E.cm;
        Conf = array2table(cm,'VariableNames',strcat("Pred_",S.classNames),'RowNames',strcat("True_",S.classNames));
        writetable(table(E.acc,'VariableNames',{'OverallAccuracy'}), fullfile(S.runDir, tag+"_OverallAccuracy.csv"));
        writetable(Conf, fullfile(S.runDir, tag+"_ConfusionMatrix.csv"),'WriteRowNames',true);

        % Build metrics table from E.metrics
        % Convert metricsData (cell array) to table for CSV
        md = E.metrics;
        Tm = cell2table(md, 'VariableNames', {'Class','Support','Precision','Recall','F1','Specificity','PerClassAccuracy'});
        writetable(Tm, fullfile(S.runDir, tag+"_EvaluationMetrics.csv"));

        Ta = table(S.classNames(:), E.auc(:), 'VariableNames', {'Class','AUC'});
        writetable(Ta, fullfile(S.runDir, tag+"_AUC_Table.csv"));
    end

%% -------- Inference tab callbacks ----------
    function onLoadModel(~,~)
        [f,pth] = uigetfile("*.mat","Select retina_classification_model.mat");
        if isequal(f,0), return; end
        S.modelFile = string(fullfile(pth,f));
        tmp = load(S.modelFile);
        if ~isfield(tmp,'agentModel') || ~isfield(tmp.agentModel,'net')
            uialert(fig,'File does not contain agentModel.net','Invalid model'); return;
        end
        S.netB = tmp.agentModel.net;
        S.classNames = string(tmp.agentModel.classNames);
        lblModel.Text = "Model: " + S.modelFile;
        btnPickImg.Enable = 'on';

        mdlDir = fileparts(S.modelFile);
        S.logFile = string(fullfile(mdlDir,"RIO_SingleImage_Log.csv"));
        uialert(fig,'Model loaded. Now select an image.','Ready');
    end

    function onPickImage(~,~)
        [f,pth] = uigetfile({'*.png;*.jpg;*.jpeg;*.tif;*.tiff','Images'}, 'Select image');
        if isequal(f,0), return; end
        imgPath = string(fullfile(pth,f));
        lblImg.Text = "Image: " + imgPath;

        I = imread(imgPath);
        if ndims(I)==2, I = repmat(I,[1 1 3]); end
        if ndims(I)==3 && size(I,3)>3, I = I(:,:,1:3); end

        inH=224; inW=224;
        try
            L1 = S.netB.Layers(1);
            if isprop(L1,'InputSize')
                sz=L1.InputSize; inH=sz(1); inW=sz(2);
            end
        catch
        end
        Iin = imresize(I,[inH inW]);
        imshow(Iin,'Parent',axImg);

        % QC
        G = rgb2gray(Iin);
        meanB = mean(G(:));
        stdC  = std(double(G(:)));
        lap = imfilter(double(G), [0 1 0; 1 -4 1; 0 1 0], 'replicate');
        sharp = var(lap(:));

        flags = strings(0,1);
        if meanB < S.qc.lowBright, flags(end+1)="Low brightness"; end %#ok<AGROW>
        if meanB > S.qc.highBright, flags(end+1)="High brightness"; end %#ok<AGROW>
        if stdC  < S.qc.minContrast, flags(end+1)="Low contrast"; end %#ok<AGROW>
        if sharp < S.qc.minSharpness, flags(end+1)="Blurry"; end %#ok<AGROW>
        if isempty(flags), q="Quality OK"; else, q=strjoin(flags,"; "); end
        qualityOK = strcmp(q,"Quality OK");

        % Predict
        [yp, scores] = classify(S.netB, Iin);
        scores = squeeze(scores); if isrow(scores), scores=scores(:); end
        [pMax, ~] = max(scores);
        predClass = string(yp);

        acceptThr = S.pol.thrDefault;
        if predClass=="Glaucoma", acceptThr = S.pol.thrGlaucoma; end

        if (pMax >= acceptThr) && qualityOK
            dec = "ACCEPT";
        elseif (pMax >= S.pol.thrMinReview)
            dec = "REVIEW";
        else
            dec = "RETAKE/HUMAN";
        end

        lblPred.Text = "Prediction: " + predClass + " | Conf: " + sprintf("%.2f%%",pMax*100);
        lblQual.Text = "Quality: " + q;
        lblDec.Text  = "Policy Decision: " + dec;

        T = [cellstr(S.classNames(:)), num2cell(scores), num2cell(scores*100)];
        tblScores.Data = T;

        % Log CSV
        if strlength(S.logFile)==0
            S.logFile = string(fullfile(pwd,"RIO_SingleImage_Log.csv"));
        end

        row = table(string(datetime("now",'Format','yyyy-MM-dd HH:mm:ss')), imgPath, predClass, pMax, string(q), string(dec), ...
            'VariableNames', {'Time','ImagePath','PredictedClass','TopScore','QualityNotes','Decision'});

        if isfile(S.logFile)
            old = readtable(S.logFile,'TextType','string');
            out = [old; row];
            writetable(out, S.logFile);
        else
            writetable(row, S.logFile);
        end
    end

end
