 %% =========================================================
%  Ocular Preprocessing AI Agent (MATLAB)
%  Multi-Disease Fundus Dataset
%
%  Pipeline:
%   1) Green Channel Extraction
%   2) CLAHE (illumination correction)
%   3) Bilateral Filtering (denoising)
%   4) Adaptive Gamma Correction (INLINE)
%
%  Input:
%   F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\1
%
%  Output:
%   results cataract
%   results DR
%   results Glaucoma
%   results Normal
%
%% =========================================================

clc; clear; close all;

%% ======================
% Paths Configuration
% ======================
datasetRoot = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\1';
parentPath  = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset';

disp(['Dataset root: ', datasetRoot]);

%% ======================
% Detect disease folders
% ======================
d = dir(datasetRoot);
isSub = [d.isdir] & ~ismember({d.name},{'.','..'});
classFolders = d(isSub);

if isempty(classFolders)
    error('❌ No disease subfolders found. Put each disease in a separate folder.');
end

fprintf('Found %d disease classes:\n', numel(classFolders));
for i = 1:numel(classFolders)
    fprintf(' - %s\n', classFolders(i).name);
end

%% ======================
% Image extensions
% ======================
extensions = {'*.jpg','*.jpeg','*.png','*.bmp'};

%% ======================
% Main Agent Loop
% ======================
for c = 1:numel(classFolders)

    className = classFolders(c).name;        % Disease name
    classPath = fullfile(datasetRoot, className);

    % Create disease-specific results folder
    resultsRoot = fullfile(parentPath, ['results ' className]);
    if ~exist(resultsRoot, 'dir')
        mkdir(resultsRoot);
    end

    fprintf('\nProcessing class: %s\nSaving to: %s\n', className, resultsRoot);

    % Collect images
    imageFiles = [];
    for e = 1:numel(extensions)
        imageFiles = [imageFiles; dir(fullfile(classPath, extensions{e}))]; %#ok<AGROW>
    end

    fprintf('Images found: %d\n', numel(imageFiles));

    savedCount = 0;

    for k = 1:numel(imageFiles)

        imgPath = fullfile(classPath, imageFiles(k).name);

        % -------------------------
        % Ingestion Agent
        % -------------------------
        try
            img = imread(imgPath);
        catch
            fprintf('⚠️ Cannot read image: %s\n', imgPath);
            continue;
        end

        % Ensure RGB
        if size(img,3) == 1
            img = cat(3, img, img, img);
        end

        % -------------------------
        % Stage 1: Green Channel
        % -------------------------
        greenChannel = img(:,:,2);

        % -------------------------
        % Stage 2: CLAHE
        % -------------------------
        claheImg = adapthisteq(greenChannel, ...
                               'ClipLimit', 0.02, ...
                               'NumTiles', [8 8]);

        % -------------------------
        % Stage 3: Bilateral Filter
        % -------------------------
        denoisedImg = imbilatfilt(claheImg);

        % -------------------------
        % Stage 4: Adaptive Gamma (INLINE)
        % -------------------------
        imgDouble = im2double(denoisedImg);
        meanIntensity = mean(imgDouble(:));

        if meanIntensity < 0.4
            gamma = 1.2;      % dark image
        elseif meanIntensity < 0.6
            gamma = 1.1;      % slightly dark
        else
            gamma = 1.0;      % normal
        end

        gammaCorrectedImg = im2uint8(imgDouble .^ gamma);

        % -------------------------
        % Save Result
        % -------------------------
        savePath = fullfile(resultsRoot, imageFiles(k).name);
        imwrite(gammaCorrectedImg, savePath);

        savedCount = savedCount + 1;

    end

    fprintf('✅ Class %s completed | Saved images: %d\n', className, savedCount);

end

disp('========================================');
disp('🎉 ALL CLASSES PROCESSED SUCCESSFULLY');
disp('========================================');
