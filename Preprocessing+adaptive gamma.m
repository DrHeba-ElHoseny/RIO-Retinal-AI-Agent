%% =========================================================
%  Fundus / Cataract Dataset Preprocessing (4 Stages)
%  Stage 1: Green Channel Extraction
%  Stage 2: CLAHE
%  Stage 3: Bilateral Filtering
%  Stage 4: Adaptive Gamma Correction (INLINE)
%
%  Input  : F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\1
%  Output : F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\results cataract
%
%  Save as: preprocess_cataract_dataset.m
%% =========================================================

clc; clear; close all;

%% ======================
% Paths Configuration
% ======================
datasetPath = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\normal';
parentPath  = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset';
resultsPath = fullfile(parentPath, 'results normalGamma');

if ~exist(resultsPath, 'dir')
    mkdir(resultsPath);
end
disp(['Results will be saved in: ', resultsPath]);

%% ======================
% Read Image Files
% ======================
extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp'};
imageFiles = [];

for i = 1:length(extensions)
    imageFiles = [imageFiles; dir(fullfile(datasetPath, extensions{i}))]; %#ok<AGROW>
end

fprintf('Total images found: %d\n', length(imageFiles));

%% ======================
% Preprocessing Loop
% ======================
for k = 1:length(imageFiles)

    % Full image path
    imgPath = fullfile(datasetPath, imageFiles(k).name);

    % Read image
    img = imread(imgPath);

    % Convert grayscale to RGB if needed
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
        gamma = 1.2;        % dark images
    elseif meanIntensity < 0.6
        gamma = 1.1;        % slightly dark
    else
        gamma = 1.0;        % normal brightness
    end

    gammaCorrectedImg = imgDouble .^ gamma;
    gammaCorrectedImg = im2uint8(gammaCorrectedImg);

    % -------------------------
    % Save processed image
    % -------------------------
    savePath = fullfile(resultsPath, imageFiles(k).name);
    imwrite(gammaCorrectedImg, savePath);

end

disp('✅ Preprocessing completed successfully.');
