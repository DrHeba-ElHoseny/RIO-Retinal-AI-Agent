 %% =========================================================
%  Evaluate Image Quality BEFORE vs AFTER (Two Folders)
%  Robust reading: Recursive + supports upper/lower extensions
%  Outputs Excel with:
%   - ImageName, Dimensions, File size
%   - 9 metrics BEFORE/AFTER + deltas:
%     1 Mean, 2 Std, 3 Entropy, 4 LaplacianVar, 5 Tenengrad,
%     6 SNRproxy, 7 EdgeDensity, 8 LocalContrast, 9 QualityFactor
%
%  ✅ EDIT ONLY: beforeDir, afterDir, MAX_IMAGES (optional)
%% =========================================================

clc; clear; close all;

%% =========================
%  USER INPUT (EDIT HERE)
%% =========================
beforeDir  = 'F:\path\to\BEFORE';   % <-- put BEFORE folder path
afterDir   = 'F:\path\to\AFTER';    % <-- put AFTER folder path
MAX_IMAGES = Inf;                   % Inf = all matched, or set e.g., 40

% Output Excel path (saved inside AFTER folder by default)
outXlsx = fullfile(afterDir, 'quality_metrics_before_after.xlsx');

%% =========================
%  Validate folders
%% =========================
if ~isfolder(beforeDir)
    error("BEFORE folder not found: %s", beforeDir);
end
if ~isfolder(afterDir)
    error("AFTER folder not found: %s", afterDir);
end

%% =========================
%  Recursive listing (robust)
%% =========================
extPattern = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff', ...
              '*.JPG','*.JPEG','*.PNG','*.BMP','*.TIF','*.TIFF'};

beforeList = []; afterList = [];
for i = 1:numel(extPattern)
    beforeList = [beforeList; dir(fullfile(beforeDir, '**', extPattern{i}))]; %#ok<AGROW>
    afterList  = [afterList;  dir(fullfile(afterDir,  '**', extPattern{i}))]; %#ok<AGROW>
end

fprintf("BEFORE images found (recursive): %d\n", numel(beforeList));
fprintf("AFTER  images found (recursive): %d\n", numel(afterList));

if isempty(beforeList)
    error("No images found in BEFORE (even recursively). Check path or extensions.");
end
if isempty(afterList)
    error("No images found in AFTER (even recursively). Check path or extensions.");
end

% Full paths
beforePaths = fullfile(string({beforeList.folder}), string({beforeList.name}));
afterPaths  = fullfile(string({afterList.folder}),  string({afterList.name}));

% Filenames for matching
beforeNames = string({beforeList.name});
afterNames  = string({afterList.name});

% Match common filenames
commonNames = intersect(beforeNames, afterNames);
commonNames = sort(commonNames);

fprintf("Matched filenames (common): %d\n", numel(commonNames));
if isempty(commonNames)
    error("No matching filenames between BEFORE and AFTER folders. Ensure same filenames.");
end

% Apply MAX_IMAGES
if isfinite(MAX_IMAGES)
    commonNames = commonNames(1:min(MAX_IMAGES, numel(commonNames)));
end

fprintf("Evaluating %d matched image pairs...\n", numel(commonNames));

%% =========================
%  Compute metrics and build table rows
%% =========================
rows = {};

for k = 1:numel(commonNames)
    imgName = commonNames(k);

    idxB = find(beforeNames == imgName, 1, 'first');
    idxA = find(afterNames  == imgName, 1, 'first');

    if isempty(idxB) || isempty(idxA)
        fprintf("⚠️ Missing pair index for: %s\n", imgName);
        continue;
    end

    beforePath = beforePaths(idxB);
    afterPath  = afterPaths(idxA);

    % Read images
    try
        imgB = imread(beforePath);
        imgA = imread(afterPath);
    catch
        fprintf("⚠️ Cannot read pair: %s\n", imgName);
        continue;
    end

    % Ensure RGB then use green channel
    if size(imgB,3)==1, imgB = cat(3,imgB,imgB,imgB); end
    if size(imgA,3)==1, imgA = cat(3,imgA,imgA,imgA); end

    gB = imgB(:,:,2);
    gA = imgA(:,:,2);

    % Dimensions (HxW)
    H = size(gB,1); W = size(gB,2);

    % File sizes (KB)
    sB = dir(beforePath); sizeKB_B = sB.bytes / 1024;
    sA = dir(afterPath);  sizeKB_A = sA.bytes / 1024;

    % 9 metrics BEFORE/AFTER
    mB = computeMetrics(gB);
    mA = computeMetrics(gA);

    rows(end+1,:) = { ...
        char(imgName), H, W, ...
        sizeKB_B, sizeKB_A, ...
        mB.meanI, mA.meanI, (mA.meanI - mB.meanI), ...
        mB.stdI,  mA.stdI,  (mA.stdI  - mB.stdI), ...
        mB.ent,   mA.ent,   (mA.ent   - mB.ent), ...
        mB.lapVar,mA.lapVar,(mA.lapVar- mB.lapVar), ...
        mB.ten,   mA.ten,   (mA.ten   - mB.ten), ...
        mB.snr,   mA.snr,   (mA.snr   - mB.snr), ...
        mB.edgeD, mA.edgeD, (mA.edgeD - mB.edgeD), ...
        mB.localC, mA.localC, (mA.localC - mB.localC), ...
        mB.qf,     mA.qf,     (mA.qf     - mB.qf), ...
        char(beforePath), char(afterPath) ...
    }; %#ok<AGROW>

    if mod(k,10)==0
        fprintf("Processed %d / %d\n", k, numel(commonNames));
    end
end

%% =========================
%  Export to Excel
%% =========================
colNames = { ...
    'ImageName','Height','Width', ...
    'FileSizeKB_Before','FileSizeKB_After', ...
    'Mean_Before','Mean_After','dMean', ...
    'Std_Before','Std_After','dStd', ...
    'Entropy_Before','Entropy_After','dEntropy', ...
    'LaplacianVar_Before','LaplacianVar_After','dLaplacianVar', ...
    'Tenengrad_Before','Tenengrad_After','dTenengrad', ...
    'SNRproxy_Before','SNRproxy_After','dSNRproxy', ...
    'EdgeDensity_Before','EdgeDensity_After','dEdgeDensity', ...
    'LocalContrast_Before','LocalContrast_After','dLocalContrast', ...
    'QualityFactor_Before','QualityFactor_After','dQualityFactor', ...
    'BeforeFullPath','AfterFullPath' ...
};

T = cell2table(rows, 'VariableNames', colNames);
writetable(T, outXlsx, 'FileType','spreadsheet');

disp("✅ Excel saved to:");
disp(outXlsx);

%% =========================================================
% Local Functions
%% =========================================================
function M = computeMetrics(g)
    % g: uint8 2D (green channel)
    gd = im2double(g);

    % 1) Mean intensity
    meanI = mean(gd(:));

    % 2) Std intensity (contrast proxy)
    stdI  = std(gd(:));

    % 3) Entropy
    ent = entropy(g);

    % 4) Laplacian variance (sharpness)
    lap = imfilter(gd, fspecial('laplacian', 0), 'replicate');
    lapVar = var(lap(:));

    % 5) Tenengrad (Sobel energy)
    gx = imfilter(gd, fspecial('sobel')'/8, 'replicate');
    gy = imfilter(gd, fspecial('sobel')/8,  'replicate');
    ten = mean(gx(:).^2 + gy(:).^2);

    % 6) SNR proxy: mean / std(residual)
    smooth = imgaussfilt(gd, 1.0);
    resid  = gd - smooth;
    noiseStd = std(resid(:));
    snr = meanI / (noiseStd + 1e-8);

    % 7) Edge density (Canny)
    edges = edge(g, 'Canny');
    edgeD = mean(edges(:));

    % 8) Local contrast: mean local std (7x7)
    localStd = stdfilt(g, true(7));
    localC = mean(im2double(localStd(:)));

    % 9) Quality Factor (composite index)
    % Normalize components with safe clipping
    nSharp = normClip(ten,    0.0000, 0.0200);
    nLocC  = normClip(localC, 0.0000, 0.2000);
    nEnt   = normClip(ent,    0.0,    8.0);
    nSNR   = normClip(snr,    0.0,   30.0);

    w1=0.35; w2=0.30; w3=0.20; w4=0.15;
    qf = w1*nSharp + w2*nLocC + w3*nEnt + w4*nSNR;

    M.meanI  = meanI;
    M.stdI   = stdI;
    M.ent    = ent;
    M.lapVar = lapVar;
    M.ten    = ten;
    M.snr    = snr;
    M.edgeD  = edgeD;
    M.localC = localC;
    M.qf     = qf;
end

function x = normClip(v, vmin, vmax)
    if vmax <= vmin
        x = 0;
        return;
    end
    v = min(max(v, vmin), vmax);
    x = (v - vmin) / (vmax - vmin);
end
