% RetinaClassificationAgent.m
% ------------------------------------------------------------
% AI Agent (Classification-Only) for Retinal Fundus Images
% Classes: Cataract, Diabetic Retinopathy, Glaucoma, Normal
%
% What this agent does:
% 1) Validates input image
% 2) Applies optional resizing/normalization for the network
% 3) Runs classification using a loaded network
% 4) Computes probabilities + confidence level
% 5) Applies a clear policy to output a decision:
%    - ACCEPT (high confidence)
%    - REVIEW (mid confidence or borderline quality)
%    - RETAKE (very low quality / invalid image)
%
% How to use (quick):
%   report = RetinaClassificationAgent("image.jpg", "model.mat");
%
% Model requirements:
% - modelFile (.mat) must contain either:
%   A) a variable named 'net' (DAGNetwork/SeriesNetwork/dlnetwork)
%      and optionally 'classNames' (string/categorical array)
%   OR
%   B) a variable named 'agentModel' struct with fields:
%      agentModel.net
%      agentModel.classNames  (optional)
%
% Notes:
% - This file is self-contained: one .m file ready to paste into MATLAB.
% - If you already have a preprocessing agent, call it before this agent
%   and pass the preprocessed image matrix I directly.
% ------------------------------------------------------------

function report = RetinaClassificationAgent(inputImage, modelFile, varargin)
% report = RetinaClassificationAgent(inputImage, modelFile)
%
% INPUTS:
%   inputImage : (1) file path (string/char) OR (2) image matrix HxWxC
%   modelFile  : path to .mat file containing the model (see requirements)
%
% OPTIONAL name-value args:
%   "QualityScore" : numeric [0..1] from your preprocessing/quality agent
%   "Meta"         : struct metadata (patientID, sessionID, etc.)
%   "Verbose"      : true/false
%
% OUTPUT:
%   report : struct containing prediction, probabilities, confidence, policy decision

    % -----------------------------
    % Parse inputs
    % -----------------------------
    p = inputParser;
    p.addRequired("inputImage");
    p.addRequired("modelFile", @(x) isstring(x) || ischar(x));
    p.addParameter("QualityScore", [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    p.addParameter("Meta", struct(), @(x) isstruct(x));
    p.addParameter("Verbose", false, @(x) islogical(x) && isscalar(x));
    p.parse(inputImage, modelFile, varargin{:});
    args = p.Results;

    cfg = defaultAgentConfig();

    % -----------------------------
    % Load/Read image
    % -----------------------------
    [I, imgInfo] = loadImage(args.inputImage);

    % Basic validation
    if isempty(I)
        report = buildReportInvalid("Invalid image input (empty).", args.Meta, cfg);
        return;
    end
    if ndims(I) < 2
        report = buildReportInvalid("Invalid image input (unexpected dimensions).", args.Meta, cfg);
        return;
    end

    % Ensure RGB (HxWx3)
    I = ensureRGB(I);

    % -----------------------------
    % Optional: lightweight quality checks (no extra model)
    % If you already have a quality score from preprocessing agent,
    % pass it via "QualityScore". Otherwise we compute a heuristic.
    % -----------------------------
    if isempty(args.QualityScore)
        quality = heuristicQuality(I);
    else
        quality.score = clamp01(args.QualityScore);
        quality.method = "external";
        quality.flags  = string.empty(1,0);
    end

    % Quality policy gate
    if quality.score < cfg.Q_MIN
        report = buildReportRetake(quality, args.Meta, cfg, imgInfo);
        return;
    end

    % -----------------------------
    % Load model
    % -----------------------------
    model = loadAgentModel(modelFile);

    % Validate model
    if isempty(model.net)
        report = buildReportInvalid("Model file did not contain a valid network variable.", args.Meta, cfg);
        return;
    end

    % Resolve class names
    classNames = resolveClassNames(model);
    if numel(classNames) ~= 4
        % Still proceed, but warn in report
        classNames = string(classNames(:));
    end

    % -----------------------------
    % Preprocess for network input
    % -----------------------------
    [X, inputSize] = prepareForNetwork(I, model.net);

    % -----------------------------
    % Run classification
    % -----------------------------
    [probs, labelTop, pTop] = runClassification(model.net, X, classNames);

    % -----------------------------
    % Confidence bucket
    % -----------------------------
    confLevel = confidenceLevel(pTop, cfg);

    % -----------------------------
    % Policy decision
    % -----------------------------
    policy = applyPolicy(cfg, quality, confLevel);

    % -----------------------------
    % Build report
    % -----------------------------
    report = buildReport(cfg, imgInfo, inputSize, quality, classNames, probs, labelTop, pTop, confLevel, policy, args.Meta);

    % -----------------------------
    % Optional verbose prints
    % -----------------------------
    if args.Verbose
        fprintf("\n--- RetinaClassificationAgent ---\n");
        fprintf("Top label: %s\n", report.prediction.label);
        fprintf("Top prob : %.4f\n", report.prediction.probability);
        fprintf("Conf     : %s\n", report.prediction.confidenceLevel);
        fprintf("Decision : %s\n", report.decision.action);
        fprintf("Reason   : %s\n", report.decision.reason);
        fprintf("Quality  : %.3f (%s)\n", report.quality.score, report.quality.method);
    end
end

% =====================================================================
% CONFIG
% =====================================================================
function cfg = defaultAgentConfig()
    % Quality thresholds (0..1)
    cfg.Q_MIN = 0.45;     % below => RETAKE
    cfg.Q_OK  = 0.70;     % below this, can still work but may REVIEW

    % Classification probability thresholds
    cfg.P_HIGH = 0.85;    % >= => HIGH confidence
    cfg.P_LOW  = 0.65;    % <  => LOW confidence

    % Policy preference:
    % if quality is borderline (Q_MIN..Q_OK) and not high confidence => REVIEW
    cfg.borderlineQualityTriggersReview = true;

    % Expected classes (can be overridden by model if provided)
    cfg.expectedClasses = ["Cataract","Diabetic Retinopathy","Glaucoma","Normal"];
end

% =====================================================================
% IMAGE LOADING & PREP
% =====================================================================
function [I, info] = loadImage(inputImage)
    info = struct();
    I = [];
    try
        if isstring(inputImage) || ischar(inputImage)
            path = string(inputImage);
            if ~isfile(path)
                info.source = path;
                info.type = "path";
                info.exists = false;
                return;
            end
            I = imread(path);
            info.source = path;
            info.type = "path";
            info.exists = true;
        else
            I = inputImage;
            info.source = "in_memory";
            info.type = "matrix";
            info.exists = true;
        end

        info.size = size(I);
        info.class = class(I);
    catch
        I = [];
    end
end

function I = ensureRGB(I)
    % Convert grayscale to RGB; handle RGBA by dropping alpha
    if ismatrix(I)
        I = repmat(I, 1, 1, 3);
    elseif ndims(I) == 3
        if size(I,3) == 1
            I = repmat(I, 1, 1, 3);
        elseif size(I,3) >= 4
            I = I(:,:,1:3);
        end
    end

    % Convert to uint8 if needed for standard image ops
    if ~isa(I,"uint8")
        I = im2uint8(I);
    end
end

function q = heuristicQuality(I)
    % Simple, fast heuristics: sharpness + brightness range
    % Outputs q.score in [0..1]
    q.method = "heuristic";
    q.flags = string.empty(1,0);

    gray = rgb2gray(I);
    grayD = im2double(gray);

    % Sharpness proxy: variance of Laplacian
    h = fspecial('laplacian', 0.2);
    lap = imfilter(grayD, h, 'replicate');
    sharp = var(lap(:));           % small => blurry

    % Brightness: avoid too dark/bright
    meanB = mean(grayD(:));
    stdB  = std(grayD(:));

    % Normalize to [0..1] using soft bounds
    sharpN = clamp01((sharp - 0.0002) / (0.005 - 0.0002));   % tune if needed
    meanN  = 1 - min(abs(meanB - 0.5)/0.5, 1);               % best near 0.5
    stdN   = clamp01((stdB - 0.10) / (0.25 - 0.10));         % tune if needed

    score = 0.45*sharpN + 0.35*meanN + 0.20*stdN;
    score = clamp01(score);

    % Flags
    if sharpN < 0.25, q.flags(end+1) = "blurry"; end
    if meanB < 0.20, q.flags(end+1) = "too_dark"; end
    if meanB > 0.85, q.flags(end+1) = "too_bright"; end

    q.score = score;
end

function [X, inputSize] = prepareForNetwork(I, net)
    % Determine expected input size from network
    inputSize = inferInputSize(net);  % [H W C]

    % Resize if necessary
    if ~isempty(inputSize)
        targetH = inputSize(1); targetW = inputSize(2);
        if size(I,1) ~= targetH || size(I,2) ~= targetW
            Irs = imresize(I, [targetH targetW]);
        else
            Irs = I;
        end
    else
        Irs = I;
        inputSize = [size(I,1) size(I,2) size(I,3)];
    end

    % Normalize to single in [0..1]
    X = im2single(Irs);
end

function sz = inferInputSize(net)
    % Tries to infer input size for common MATLAB network types
    sz = [];
    try
        if isa(net, "DAGNetwork") || isa(net, "SeriesNetwork")
            inLayer = net.Layers(1);
            if isprop(inLayer, "InputSize")
                sz = inLayer.InputSize; % [H W C]
            end
        elseif isa(net, "dlnetwork")
            % Attempt to infer from first learnable/input layer if possible
            l = net.Layers;
            for i = 1:numel(l)
                if isprop(l(i), "InputSize")
                    sz = l(i).InputSize;
                    break;
                end
            end
        end
    catch
        sz = [];
    end
end

% =====================================================================
% MODEL LOADING
% =====================================================================
function model = loadAgentModel(modelFile)
    model = struct('net', [], 'classNames', []);
    S = load(modelFile);

    if isfield(S, "agentModel") && isstruct(S.agentModel)
        if isfield(S.agentModel, "net")
            model.net = S.agentModel.net;
        end
        if isfield(S.agentModel, "classNames")
            model.classNames = S.agentModel.classNames;
        end
        return;
    end

    % Try common variable names
    if isfield(S, "net")
        model.net = S.net;
    elseif isfield(S, "trainedNet")
        model.net = S.trainedNet;
    elseif isfield(S, "network")
        model.net = S.network;
    end

    if isfield(S, "classNames")
        model.classNames = S.classNames;
    elseif isfield(S, "classes")
        model.classNames = S.classes;
    end
end

function classNames = resolveClassNames(model)
    % If provided in .mat use it; else attempt from net; else default expected names
    if ~isempty(model.classNames)
        classNames = string(model.classNames);
        return;
    end

    try
        % Some classification networks store class names in final layer
        if isa(model.net, "DAGNetwork") || isa(model.net, "SeriesNetwork")
            lastL = model.net.Layers(end);
            if isprop(lastL, "Classes")
                classNames = string(lastL.Classes);
                return;
            end
        end
    catch
        % ignore
    end

    classNames = ["Cataract","Diabetic Retinopathy","Glaucoma","Normal"];
end

% =====================================================================
% CLASSIFICATION
% =====================================================================
function [probs, labelTop, pTop] = runClassification(net, X, classNames)
    % Outputs:
    % probs: 1xK probabilities aligned with classNames
    % labelTop: string
    % pTop: scalar

    scores = [];

    try
        if isa(net, "DAGNetwork") || isa(net, "SeriesNetwork")
            % classify + predict are both supported
            % Prefer predict to get scores and compute probs ourselves
            raw = predict(net, X);
            scores = raw(:)';
        elseif isa(net, "dlnetwork")
            dlX = dlarray(X, "SSCB"); % HxWxCxB
            raw = predict(net, dlX);
            raw = extractdata(raw);
            scores = raw(:)';
        else
            % Unknown network type; try predict
            raw = predict(net, X);
            scores = raw(:)';
        end
    catch
        % Fallback: use classify if predict fails (but may not return scores)
        try
            label = classify(net, X);
            % If we cannot get probabilities, create a one-hot style output
            probs = zeros(1, numel(classNames), "single");
            idx = find(string(classNames) == string(label), 1);
            if isempty(idx), idx = 1; end
            probs(idx) = 1;
            [pTop, idTop] = max(probs);
            labelTop = string(classNames(idTop));
            return;
        catch ME
            error("Classification failed: %s", ME.message);
        end
    end

    % Compute probabilities (robust softmax)
    probs = stableSoftmax(scores);

    % Align length with class names (if mismatch, truncate/pad)
    K = numel(classNames);
    if numel(probs) ~= K
        probs = probs(:)';
        if numel(probs) > K
            probs = probs(1:K);
            probs = probs / sum(probs);
        else
            probs = [probs, zeros(1, K-numel(probs), "like", probs)];
            probs = probs / max(sum(probs), eps("single"));
        end
    end

    [pTop, idTop] = max(probs);
    labelTop = string(classNames(idTop));
end

function p = stableSoftmax(x)
    x = single(x);
    x = x - max(x);
    ex = exp(x);
    s = sum(ex);
    if s <= 0
        p = ones(size(x), "single") ./ numel(x);
    else
        p = ex ./ s;
    end
end

function level = confidenceLevel(pTop, cfg)
    if pTop >= cfg.P_HIGH
        level = "HIGH";
    elseif pTop >= cfg.P_LOW
        level = "MID";
    else
        level = "LOW";
    end
end

% =====================================================================
% POLICY ENGINE (Classification-only)
% =====================================================================
function decision = applyPolicy(cfg, quality, confLevel)
    decision = struct();
    decision.action = "REVIEW";
    decision.reason = "Default policy";

    if quality.score < cfg.Q_MIN
        decision.action = "RETAKE";
        decision.reason = "Image quality below minimum threshold";
        return;
    end

    if quality.score < cfg.Q_OK && cfg.borderlineQualityTriggersReview
        if confLevel ~= "HIGH"
            decision.action = "REVIEW";
            decision.reason = "Borderline quality and classification not high confidence";
            return;
        end
    end

    if confLevel == "HIGH"
        decision.action = "ACCEPT";
        decision.reason = "High confidence classification";
    elseif confLevel == "MID"
        decision.action = "REVIEW";
        decision.reason = "Mid confidence classification (human review recommended)";
    else
        % LOW
        if quality.score < cfg.Q_OK
            decision.action = "RETAKE";
            decision.reason = "Low confidence with borderline quality (retake recommended)";
        else
            decision.action = "REVIEW";
            decision.reason = "Low confidence classification (human review required)";
        end
    end
end

% =====================================================================
% REPORT BUILDERS
% =====================================================================
function report = buildReport(cfg, imgInfo, inputSize, quality, classNames, probs, labelTop, pTop, confLevel, policy, meta)
    report = struct();

    report.timestamp = string(datetime("now", "TimeZone", "Asia/Riyadh", "Format", "yyyy-MM-dd HH:mm:ss z"));
    report.agent = struct( ...
        'name', "RetinaClassificationAgent", ...
        'version', "1.0", ...
        'task', "classification_only", ...
        'classes', string(classNames), ...
        'thresholds', struct( ...
            'Q_MIN', cfg.Q_MIN, ...
            'Q_OK', cfg.Q_OK, ...
            'P_LOW', cfg.P_LOW, ...
            'P_HIGH', cfg.P_HIGH ...
        ) ...
    );

    report.image = struct( ...
        'source', getfieldOr(imgInfo, "source", "unknown"), ...
        'type',   getfieldOr(imgInfo, "type", "unknown"), ...
        'originalSize', getfieldOr(imgInfo, "size", []), ...
        'originalClass', getfieldOr(imgInfo, "class", ""), ...
        'networkInputSize', inputSize ...
    );

    report.quality = struct( ...
        'score', quality.score, ...
        'method', quality.method, ...
        'flags', quality.flags ...
    );

    report.prediction = struct( ...
        'label', labelTop, ...
        'probability', single(pTop), ...
        'confidenceLevel', confLevel, ...
        'probabilities', table(classNames(:), probs(:), 'VariableNames', {'Class','Probability'}) ...
    );

    report.decision = policy;

    report.meta = meta;
end

function report = buildReportInvalid(msg, meta, cfg)
    report = struct();
    report.timestamp = string(datetime("now", "TimeZone", "Asia/Riyadh", "Format", "yyyy-MM-dd HH:mm:ss z"));
    report.agent = struct('name',"RetinaClassificationAgent",'version',"1.0",'task',"classification_only",'thresholds',cfg);
    report.status = "INVALID";
    report.message = string(msg);
    report.meta = meta;
end

function report = buildReportRetake(quality, meta, cfg, imgInfo)
    report = struct();
    report.timestamp = string(datetime("now", "TimeZone", "Asia/Riyadh", "Format", "yyyy-MM-dd HH:mm:ss z"));
    report.agent = struct( ...
        'name', "RetinaClassificationAgent", ...
        'version', "1.0", ...
        'task', "classification_only", ...
        'thresholds', struct('Q_MIN',cfg.Q_MIN,'Q_OK',cfg.Q_OK,'P_LOW',cfg.P_LOW,'P_HIGH',cfg.P_HIGH) ...
    );
    report.status = "OK";
    report.image = struct('source', getfieldOr(imgInfo,"source","unknown"), 'type', getfieldOr(imgInfo,"type","unknown"));

    report.quality = quality;

    report.prediction = struct( ...
        'label', "", ...
        'probability', 0, ...
        'confidenceLevel', "N/A", ...
        'probabilities', table() ...
    );

    report.decision = struct( ...
        'action', "RETAKE", ...
        'reason', "Image quality below minimum threshold", ...
        'recommendation', "Re-capture the image with proper focus/illumination and sufficient field-of-view." ...
    );

    report.meta = meta;
end

% =====================================================================
% UTILITIES
% =====================================================================
function y = clamp01(x)
    y = max(0, min(1, x));
end

function v = getfieldOr(S, fieldName, defaultVal)
    if isstruct(S) && isfield(S, fieldName)
        v = S.(fieldName);
    else
        v = defaultVal;
    end
end
