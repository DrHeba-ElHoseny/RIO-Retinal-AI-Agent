%% =========================================================
%  Fundus / Cataract Dataset Preprocessing
%  Green Channel + CLAHE + Bilateral Filtering
%  Author: Dr. Heba El-Hoseny (Project Pipeline)
% ==========================================================

clc; clear; close all;

%% ======================
% Paths Configuration
% ======================

% Dataset path (Windows – supports Arabic & spaces)
datasetPath = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\Quality Test After Preprocessing\DR-Before';

% Parent directory
parentPath = 'F:\%% Research Newwwww\Dr Yasser-Jeddah Project\archive\dataset\Quality Test After Preprocessing\DR-After';

% Results folder
resultsPath = fullfile(parentPath, 'results DR3');

% Create results folder if it does not exist
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
    imageFiles = [imageFiles; dir(fullfile(datasetPath, extensions{i}))];
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

    % =========================
    % Preprocessing Pipeline
    % =========================

    % 1) Extract Green Channel
    greenChannel = img(:,:,3);

    % 2) CLAHE for illumination correction
    claheImg = adapthisteq(greenChannel, ...
                           'ClipLimit', 0.02, ...
                           'NumTiles', [8 8]);

    % 3) Edge-preserving denoising
    denoisedImg = imbilatfilt(claheImg);

    % =========================
    % Save processed image
    % =========================

    savePath = fullfile(resultsPath, imageFiles(k).name);
    imwrite(denoisedImg, savePath);

end

disp('✅ Preprocessing completed successfully.');

%% ======================
% End of Script
% ======================
