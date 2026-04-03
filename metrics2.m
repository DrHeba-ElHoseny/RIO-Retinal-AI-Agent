%% =========================================================
%  Evaluate Image Quality BEFORE vs AFTER (Two Folders)
%  Robust reading: Recursive + supports upper/lower extensions
%
%  Metrics (IN THIS EXACT ORDER):
%   1) Average Gradient
%   2) Local Contrast
%   3) Standard Deviation
%   4) Edge Intensity
%   5) Entropy
%
%  Output Excel includes:
%   - ImageName, Dimensions, FileSize before/after
%   - Each metric before/after + delta
%
%  ✅ EDIT ONLY: beforeDir, afterDir, MAX_IMAGES
%% =========================================================

clc; clear; close all;

%% =========================
%  USER INPUT (EDIT HERE)
%% =========================
beforeDir  = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\1\Normal-Before';   % <-- BEFORE folder path
afterDir   = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\1\Normal-After';    % <-- AFTER folder path
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

    % ===== Metrics in required order =====
    mB = computeMetricsOrdered(gB);
    mA = computeMetricsOrdered(gA);

    rows(end+1,:) = { ...
        char(imgName), H, W, ...
        sizeKB_B, sizeKB_A, ...
        mB.avgGrad, mA.avgGrad, (mA.avgGrad - mB.avgGrad), ...
        mB.localC,  mA.localC,  (mA.localC  - mB.localC), ...
        mB.stdI,    mA.stdI,    (mA.stdI    - mB.stdI), ...
        mB.edgeInt, mA.edgeInt, (mA.edgeInt - mB.edgeInt), ...
        mB.ent,     mA.ent,     (mA.ent     - mB.ent), ...
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
    'AverageGradient_Before','AverageGradient_After','dAverageGradient', ...
    'LocalContrast_Before','LocalContrast_After','dLocalContrast', ...
    'StdDev_Before','StdDev_After','dStdDev', ...
    'EdgeIntensity_Before','EdgeIntensity_After','dEdgeIntensity', ...
    'Entropy_Before','Entropy_After','dEntropy', ...
    'BeforeFullPath','AfterFullPath' ...
};

T = cell2table(rows, 'VariableNames', colNames);
writetable(T, outXlsx, 'FileType','spreadsheet');

disp("✅ Excel saved to:");
disp(outXlsx);

%% =========================================================
% Local Functions
%% =========================================================
function M = computeMetricsOrdered(g)
    % g: uint8 2D (green channel)
    gd = im2double(g);

    % 1) Average Gradient
    % Use gradient magnitude from Sobel filters, then average
    sx = fspecial('sobel')/8;
    sy = sx';
    gx = imfilter(gd, sx, 'replicate');
    gy = imfilter(gd, sy, 'replicate');
    gradMag = hypot(gx, gy);
    avgGrad = mean(gradMag(:));

    % 2) Local Contrast (mean local std 7x7)
    localStd = stdfilt(g, true(7));
    localC = mean(im2double(localStd(:)));

    % 3) Standard Deviation
    stdI = std(gd(:));

    % 4) Edge Intensity
    % Mean gradient magnitude on edge pixels (Canny)
    edges = edge(g, 'Canny');
    if any(edges(:))
        edgeInt = mean(gradMag(edges));
    else
        edgeInt = 0;
    end

    % 5) Entropy
    ent = entropy(g);

    M.avgGrad = avgGrad;
    M.localC  = localC;
    M.stdI    = stdI;
    M.edgeInt = edgeInt;
    M.ent     = ent;
end
