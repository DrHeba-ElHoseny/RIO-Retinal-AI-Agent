function ocular_preprocess_ui_agent()
% ==========================================================
% Ocular Preprocess Agent UI (Single-file MATLAB App)
% - Multi-class folders
% - Pipeline: Green -> CLAHE -> Bilateral -> Adaptive Gamma
% - Optional: Reject low-quality images (blur / too dark)
% - Saves per disease: results <className> in parent folder
% ==========================================================

%% ----------------------------
% UI + App State
% ----------------------------
app = struct();
app.datasetRoot = "";
app.parentPath  = "";
app.classes     = {};
app.exts        = {'*.jpg','*.jpeg','*.png','*.bmp'};

% Quality thresholds (you can tune)
app.q_blur_thr = 80;   % Laplacian variance threshold
app.q_dark_thr = 35;   % mean gray threshold (0..255)

% Create UI
createUI();

% ----------------------------
% UI Construction
% ----------------------------
    function createUI()
        app.fig = uifigure('Name','Ocular Preprocess AI Agent','Position',[80 80 1250 720]);

        % Top controls
        app.btnSelect = uibutton(app.fig,'push','Text','Select Dataset Folder','Position',[20 675 180 30],...
            'ButtonPushedFcn',@onSelectDataset);

        app.btnRun = uibutton(app.fig,'push','Text','Run Agent','Position',[210 675 120 30],...
            'ButtonPushedFcn',@onRunAgent,'Enable','off');

        app.chkReject = uicheckbox(app.fig,'Text','Reject low-quality images','Position',[350 679 200 22],...
            'Value',true);

        app.lblStatus = uilabel(app.fig,'Text','Status: Please select dataset folder.','Position',[20 640 800 22]);

        app.pbar = uiprogressdlg(app.fig,'Title','Progress','Message','Idle','Cancelable','on');
        app.pbar.Value = 0;
        close(app.pbar); % start hidden

        % Left panel: classes + images
        uilabel(app.fig,'Text','Disease / Class','Position',[20 600 200 22],'FontWeight','bold');
        app.ddClass = uidropdown(app.fig,'Items',{},'Position',[20 575 310 28],...
            'ValueChangedFcn',@onClassChanged,'Enable','off');

        uilabel(app.fig,'Text','Images','Position',[20 540 200 22],'FontWeight','bold');
        app.lbImages = uilistbox(app.fig,'Items',{},'Position',[20 280 310 260],...
            'ValueChangedFcn',@onImageSelected,'Enable','off');

        % Middle: Before/After axes
        uilabel(app.fig,'Text','Before','Position',[360 600 100 22],'FontWeight','bold');
        app.axBefore = uiaxes(app.fig,'Position',[360 360 420 240]);
        axis(app.axBefore,'off');

        uilabel(app.fig,'Text','After','Position',[800 600 100 22],'FontWeight','bold');
        app.axAfter = uiaxes(app.fig,'Position',[800 360 420 240]);
        axis(app.axAfter,'off');

        % Bottom: Log table
        uilabel(app.fig,'Text','Processing Log','Position',[360 320 200 22],'FontWeight','bold');

        cols = {'Class','File','MeanIntensity','Gamma','Status','Reason','SavedPath'};
        app.tblLog = uitable(app.fig,'Position',[360 20 860 300],...
            'ColumnName',cols,'Data',cell(0,numel(cols)));

        % Tips
        app.lblTips = uilabel(app.fig,'Text',...
            "Tip: Dataset root should contain subfolders (classes). Each class contains images.",...
            'Position',[560 675 650 22]);
    end

%% ----------------------------
% Callbacks
% ----------------------------
    function onSelectDataset(~,~)
        folder = uigetdir(pwd,'Select Dataset Root Folder (contains class subfolders)');
        if folder == 0
            return;
        end
        app.datasetRoot = string(folder);
        app.parentPath  = string(fileparts(app.datasetRoot));

        % Detect classes (subfolders)
        d = dir(app.datasetRoot);
        isSub = [d.isdir] & ~ismember({d.name},{'.','..'});
        app.classes = {d(isSub).name};

        if isempty(app.classes)
            app.lblStatus.Text = "Status: No subfolders found. Put each disease/class in a separate folder.";
            app.ddClass.Enable = "off";
            app.lbImages.Enable = "off";
            app.btnRun.Enable = "off";
            return;
        end

        app.ddClass.Items = app.classes;
        app.ddClass.Value = app.classes{1};
        app.ddClass.Enable = "on";
        app.lbImages.Enable = "on";
        app.btnRun.Enable = "on";

        app.lblStatus.Text = "Status: Dataset loaded. Select class, preview images, or click Run Agent.";

        % Populate images for first class
        refreshImageList(app.ddClass.Value);
    end

    function onClassChanged(~,~)
        if strlength(app.datasetRoot)==0, return; end
        refreshImageList(app.ddClass.Value);
    end

    function onImageSelected(~,~)
        if strlength(app.datasetRoot)==0, return; end
        className = app.ddClass.Value;
        fileName  = app.lbImages.Value;
        if isempty(fileName), return; end

        beforePath = fullfile(app.datasetRoot, className, fileName);
        afterPath  = fullfile(app.parentPath, "results " + string(className), fileName);

        % Show Before
        if exist(beforePath,'file')
            imgB = imread(beforePath);
            imshow(imgB,'Parent',app.axBefore);
            title(app.axBefore,'Before');
        else
            cla(app.axBefore);
            title(app.axBefore,'Before (missing)');
        end

        % Show After if exists
        if exist(afterPath,'file')
            imgA = imread(afterPath);
            imshow(imgA,'Parent',app.axAfter);
            title(app.axAfter,'After');
        else
            cla(app.axAfter);
            title(app.axAfter,'After (not processed yet)');
        end
    end

    function onRunAgent(~,~)
        if strlength(app.datasetRoot)==0
            app.lblStatus.Text = "Status: Select dataset folder first.";
            return;
        end

        % Gather class folders
        d = dir(app.datasetRoot);
        isSub = [d.isdir] & ~ismember({d.name},{'.','..'});
        classFolders = d(isSub);

        % Count total images for progress
        totalImages = 0;
        for c = 1:numel(classFolders)
            classPath = fullfile(app.datasetRoot, classFolders(c).name);
            totalImages = totalImages + countImages(classPath);
        end
        if totalImages == 0
            app.lblStatus.Text = "Status: No images found in class folders.";
            return;
        end

        % Progress dialog
        app.pbar = uiprogressdlg(app.fig,'Title','Running Agent','Message','Starting...','Cancelable','on');
        app.pbar.Value = 0;

        % Clear log
        app.tblLog.Data = cell(0,7);

        processed = 0;
        rejectOn = app.chkReject.Value;

        for c = 1:numel(classFolders)
            className = string(classFolders(c).name);
            classPath = fullfile(app.datasetRoot, className);

            % Create disease-specific results folder
            resultsRoot = fullfile(app.parentPath, "results " + className);
            if ~exist(resultsRoot,'dir')
                mkdir(resultsRoot);
            end

            files = listImages(classPath);
            for k = 1:numel(files)
                if app.pbar.CancelRequested
                    app.pbar.Message = "Canceled.";
                    close(app.pbar);
                    app.lblStatus.Text = "Status: Canceled by user.";
                    return;
                end

                fileName = string(files{k});
                imgPath  = fullfile(classPath, fileName);

                % Ingestion
                try
                    img = imread(imgPath);
                catch
                    addLogRow(className, fileName, NaN, NaN, "ERROR", "cannot_read", "");
                    processed = processed + 1;
                    updateProgress();
                    continue;
                end

                if size(img,3)==1
                    img = cat(3,img,img,img);
                end

                % ===== Optional Reject Low-quality =====
                if rejectOn
                    [accept, reason] = qualityCheck(img, app.q_blur_thr, app.q_dark_thr);
                    if ~accept
                        addLogRow(className, fileName, NaN, NaN, "REJECT", reason, "");
                        processed = processed + 1;
                        updateProgress();
                        continue;
                    end
                end

                % ===== Preprocessing Pipeline (4 stages) =====
                % Stage 1: Green channel
                greenChannel = img(:,:,2);

                % Stage 2: CLAHE
                claheImg = adapthisteq(greenChannel,'ClipLimit',0.02,'NumTiles',[8 8]);

                % Stage 3: Bilateral filter
                denoisedImg = imbilatfilt(claheImg);

                % Stage 4: Adaptive Gamma (INLINE)
                imgDouble = im2double(denoisedImg);
                meanI = mean(imgDouble(:));

                if meanI < 0.4
                    gamma = 1.2;
                elseif meanI < 0.6
                    gamma = 1.1;
                else
                    gamma = 1.0;
                end

                outImg = im2uint8(imgDouble .^ gamma);

                % Save
                savePath = fullfile(resultsRoot, fileName);
                imwrite(outImg, savePath);

                addLogRow(className, fileName, meanI, gamma, "SAVED", "-", savePath);

                processed = processed + 1;
                updateProgress();
            end
        end

        close(app.pbar);
        app.lblStatus.Text = "Status: ✅ Completed. Results saved in 'results <class>' folders.";
        % refresh image preview lists
        refreshImageList(app.ddClass.Value);

        % Helper: update progress
        function updateProgress()
            app.pbar.Value = min(processed/totalImages, 1);
            app.pbar.Message = sprintf("Processed %d / %d", processed, totalImages);
            drawnow;
        end
    end

%% ----------------------------
% Helper functions (local)
% ----------------------------
    function refreshImageList(className)
        classPath = fullfile(app.datasetRoot, className);
        files = listImages(classPath);
        if isempty(files)
            app.lbImages.Items = {};
            cla(app.axBefore); cla(app.axAfter);
            return;
        end
        app.lbImages.Items = files;
        app.lbImages.Value = files{1};
        onImageSelected();
    end

    function n = countImages(folderPath)
        n = 0;
        for i=1:numel(app.exts)
            n = n + numel(dir(fullfile(folderPath, app.exts{i})));
        end
    end

    function files = listImages(folderPath)
        tmp = [];
        for i=1:numel(app.exts)
            tmp = [tmp; dir(fullfile(folderPath, app.exts{i}))]; %#ok<AGROW>
        end
        if isempty(tmp)
            files = {};
            return;
        end
        files = {tmp.name};
        files = sort(files);
    end

    function addLogRow(className, fileName, meanI, gamma, status, reason, savedPath)
        row = {char(className), char(fileName), meanI, gamma, char(status), char(reason), char(savedPath)};
        data = app.tblLog.Data;
        data(end+1,:) = row; %#ok<AGROW>
        app.tblLog.Data = data;
        drawnow limitrate;
    end

    function [accept, reason] = qualityCheck(imgRGB, blur_thr, dark_thr)
        % Convert to gray
        gray = rgb2gray(imgRGB);

        % Blur score (variance of Laplacian)
        lap = imfilter(double(gray), fspecial('laplacian', 0), 'replicate');
        blur_score = var(lap(:));

        % Mean intensity (0..255)
        mean_gray = mean(gray(:));

        reasons = strings(0);

        if blur_score < blur_thr
            reasons(end+1) = "blur";
        end
        if mean_gray < dark_thr
            reasons(end+1) = "too_dark";
        end

        accept = isempty(reasons);
        if accept
            reason = "ok";
        else
            reason = strjoin(reasons, "+");
        end
    end
end
