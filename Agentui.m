%% Retinal_RIO_SingleImageAgent_OneFile.m
% ============================================================
% Retinal Intelligence Orchestrator (RIO) - Single Image Inference Agent
% ============================================================
% هدف الملف:
% - تحميل نموذج التصنيف (retina_classification_model.mat) إن وجد
%   أو استخدام متغير trainedNet الموجود بالـ Workspace (إن وجد)
% - استقبال صورة جديدة (غير موجودة بالـ dataset)
% - Quality Gate (فحص جودة سريع: إضاءة/تباين/حدة)
% - قرار Policy: ACCEPT / REVIEW / RETAKE-HUMAN
% - إخراج:
%   1) Predicted class
%   2) Confidence + جدول احتمالات لكل الفئات (Excel-ready)
%   3) ملاحظات جودة
%   4) حفظ Log في CSV
%
% مراعاة مشاكل نسخة MATLAB:
% - لا يستخدم local functions (لتجنب مشاكل تعريف الدوال في الـ scripts)
% - لا يعتمد على layerGraph / dlnetwork
% - يتعامل مع SeriesNetwork/DAGNetwork/الموديل المحفوظ
% ============================================================

clc; close all;

%% =======================
% USER INPUTS (EDIT)
% =======================
% 1) ضع مسار الصورة الجديدة هنا:
newImagePath = "F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\DRTest.jpeg";  % <<< CHANGE THIS

% 2) (اختياري) اسم ملف النموذج المحفوظ:
modelMatFile = "retina_classification_model.mat"; % في نفس فولدر المشروع أو ضع مسار كامل

% 3) (اختياري) True label للصورة لحساب Correct=0/1 (لا يمكن حساب recall لصورة واحدة)
% اتركيها "" لو غير معروف
trueLabel = "";  % مثال: "Glaucoma"

% 4) سياسات الثقة (يمكن تعديلها)
thrDefault  = 0.75;  % قبول عام
thrGlaucoma = 0.80;  % stricter للجلوكوما
thrMinReview = 0.60; % أقل من هذا: retake/human

% 5) ملف لوج للصور الجديدة
logFile = "RIO_SingleImage_Log.csv";

%% =======================
% 0) CHECK IMAGE PATH
% =======================
if ~isfile(newImagePath)
    error("Image file not found: %s", newImagePath);
end

%% =======================
% 1) LOAD MODEL SAFELY
% =======================
% الأولوية:
% (A) لو trainedNet موجود بالـ workspace -> استخدمه
% (B) لو modelMatFile موجود -> load agentModel.net
% وإلا -> error

netFinal = [];
classNames = [];

if evalin('base',"exist('trainedNet','var')") == 1
    try
        netFinal = evalin('base','trainedNet');
        disp("Loaded model from workspace variable: trainedNet");
    catch
        % ignore and try loading from file
    end
end

if isempty(netFinal)
    if isfile(modelMatFile)
        S = load(modelMatFile);
        if isfield(S,'agentModel') && isfield(S.agentModel,'net')
            netFinal = S.agentModel.net;
            disp("Loaded model from file: " + modelMatFile);
            if isfield(S.agentModel,'classNames')
                classNames = string(S.agentModel.classNames);
            end
        else
            error("File exists but agentModel.net not found inside: %s", modelMatFile);
        end
    else
        error("No model found. Provide trainedNet in workspace OR ensure %s exists.", modelMatFile);
    end
end

% لو classNames لسه فاضي، حاول نقرأها من الشبكة أو من الـ workspace
if isempty(classNames)
    if evalin('base',"exist('classNames','var')") == 1
        classNames = string(evalin('base','classNames'));
    else
        % fallback: infer from final classification layer if possible
        classNames = [];
        try
            L = netFinal.Layers;
            % ابحث عن classification layer
            idxCL = find(arrayfun(@(x) contains(class(x), 'Classification','IgnoreCase',true), L), 1, 'last');
            if ~isempty(idxCL)
                % غالبًا لا يوجد Classes property في بعض الإصدارات
                % لذا نتركه فاضي ونستخرجه بعد classify من scores length
                classNames = [];
            end
        catch
            classNames = [];
        end
    end
end

%% =======================
% 2) READ IMAGE + PREPARE INPUT
% =======================
% read
I = imread(newImagePath);

% ensure uint8/uint16 acceptable
if ~isnumeric(I)
    error("Unsupported image format. Ensure image is numeric.");
end

% ensure 3 channels
if ndims(I) == 2
    I = repmat(I, [1 1 3]);
elseif ndims(I) == 3 && size(I,3) > 3
    I = I(:,:,1:3);
elseif ndims(I) ~= 3
    error("Unsupported image dimensions: %s", mat2str(size(I)));
end

% determine input size safely
inputH = 224; inputW = 224;
try
    L1 = netFinal.Layers(1);
    if isprop(L1,'InputSize')
        sz = L1.InputSize;
        inputH = sz(1); inputW = sz(2);
    end
catch
    % keep defaults
end

Iin = imresize(I, [inputH inputW]);

% make grayscale for quality checks
try
    Igray = rgb2gray(Iin);
catch
    % fallback manual conversion
    Igray = uint8(0.2989*double(Iin(:,:,1)) + 0.5870*double(Iin(:,:,2)) + 0.1140*double(Iin(:,:,3)));
end

%% =======================
% 3) QUALITY GATE (FAST HEURISTICS)
% =======================
meanBright = mean(Igray(:));
stdContrast = std(double(Igray(:)));

% sharpness via Laplacian variance
lapKernel = [0 1 0; 1 -4 1; 0 1 0];
lap = imfilter(double(Igray), lapKernel, 'replicate');
sharpness = var(lap(:));

% thresholds (tunable)
isTooDark     = meanBright < 60;
isTooBright   = meanBright > 200;
isLowContrast = stdContrast < 25;
isBlurry      = sharpness < 50;

qualityFlags = strings(0,1);
if isTooDark,     qualityFlags(end+1) = "Low brightness (too dark)"; end %#ok<SAGROW>
if isTooBright,   qualityFlags(end+1) = "High brightness (too bright)"; end %#ok<SAGROW>
if isLowContrast, qualityFlags(end+1) = "Low contrast"; end %#ok<SAGROW>
if isBlurry,      qualityFlags(end+1) = "Blurry image (low sharpness)"; end %#ok<SAGROW>

if isempty(qualityFlags)
    qualityComment = "Quality OK";
else
    qualityComment = strjoin(qualityFlags, "; ");
end

qualityOK = strcmp(qualityComment, "Quality OK");

%% =======================
% 4) CLASSIFY + SCORES
% =======================
% classify may return categorical label and scores vector
[yp, scores] = classify(netFinal, Iin);

% normalize score shape
scores = squeeze(scores);
if isrow(scores), scores = scores(:); end

% if classNames unknown, build placeholder names
if isempty(classNames)
    K = numel(scores);
    classNames = "Class_" + string(1:K);
end

% ensure same length
K = numel(scores);
if numel(classNames) ~= K
    % try to fix by trunc/pad
    if numel(classNames) > K
        classNames = classNames(1:K);
    else
        classNames = [classNames; "Class_"+string((numel(classNames)+1):K)'];
    end
end

[pMax, idxMax] = max(scores);
predClass = string(yp);

% choose acceptance threshold (class-aware)
acceptThr = thrDefault;
if predClass == "Glaucoma"
    acceptThr = thrGlaucoma;
end

%% =======================
% 5) POLICY DECISION
% =======================
if (pMax >= acceptThr) && qualityOK
    decision = "ACCEPT";
    notes = "High confidence + good quality.";
elseif (pMax >= thrMinReview)
    decision = "REVIEW";
    notes = "Borderline confidence and/or quality issues. Consider re-capture or expert review. If Glaucoma suspected, consider OCT or disc/cup assessment.";
else
    decision = "RETAKE/HUMAN";
    notes = "Low confidence. Recommend re-capture and/or manual verification.";
end

%% =======================
% 6) DISPLAY RESULTS (User-friendly)
% =======================
disp("============================================================");
disp("RIO - Single Image Inference Result");
disp("============================================================");
disp("Image: " + newImagePath);
disp("Predicted Class: " + predClass);
disp("Top Confidence: " + sprintf("%.2f%%", pMax*100));
disp("Quality Notes: " + qualityComment);
disp("Policy Decision: " + decision);
disp("Decision Notes: " + notes);

% Excel-ready table of class probabilities
ScoresTable = table(string(classNames(:)), scores(:), scores(:)*100, ...
    'VariableNames', {'Class','Score','ScorePercent'});
disp("---- Class probabilities (Excel-ready) ----");
disp(ScoresTable);

%% =======================
% 7) OPTIONAL: SINGLE-CASE CORRECT FLAG (if trueLabel provided)
% =======================
isCorrect = NaN;
if strlength(string(trueLabel)) > 0
    trueLabel = string(trueLabel);
    isCorrect = double(predClass == trueLabel);
    disp("True Label: " + trueLabel);
    disp("Correct (1=yes,0=no): " + isCorrect);
else
    disp("True Label: (not provided) -> Accuracy/Recall cannot be computed for a single case.");
end

%% =======================
% 8) SAVE LOG ROW (CSV)
% =======================
% Create a compact report row
timeStamp = string(datetime("now",'Format','yyyy-MM-dd HH:mm:ss'));
logRow = table(timeStamp, string(newImagePath), predClass, pMax, ...
    string(qualityComment), string(decision), string(notes), string(trueLabel), isCorrect, ...
    'VariableNames', {'Time','ImagePath','PredictedClass','TopScore','QualityNotes','Decision','DecisionNotes','TrueLabel','Correct'});

% Append or create
if isfile(logFile)
    try
        old = readtable(logFile, 'TextType','string');
        out = [old; logRow];
    catch
        % if old file has different schema, create a new one with timestamp
        fallbackName = "RIO_SingleImage_Log_" + string(datetime("now",'Format','yyyyMMdd_HHmmss')) + ".csv";
        writetable(logRow, fallbackName);
        disp("Existing log schema mismatch. Saved new log: " + fallbackName);
        out = [];
    end
    if ~isempty(out)
        writetable(out, logFile);
        disp("Appended to log: " + logFile);
    end
else
    writetable(logRow, logFile);
    disp("Created log: " + logFile);
end

%% =======================
% 9) SAVE SCORES TABLE FOR THIS CASE (Excel-ready)
% =======================
% Save per-image probability table (optional)
caseScoresFile = "RIO_Scores_" + string(datetime("now",'Format','yyyyMMdd_HHmmss')) + ".csv";
try
    % Add metadata columns
    meta = table(repmat(timeStamp,K,1), repmat(string(newImagePath),K,1), repmat(predClass,K,1), repmat(pMax,K,1), ...
        'VariableNames', {'Time','ImagePath','PredictedClass','TopScore'});
    outScores = [meta, ScoresTable];
    writetable(outScores, caseScoresFile);
    disp("Saved case scores CSV: " + caseScoresFile);
catch
    disp("Could not save case scores CSV (non-critical).");
end

%% =======================
% 10) SHOW IMAGE (optional)
% =======================
try
    figure('Name','RIO - Input Image'); imshow(Iin);
    title(sprintf("Pred: %s | Conf: %.2f%% | %s", predClass, pMax*100, decision));
catch
    % ignore
end
