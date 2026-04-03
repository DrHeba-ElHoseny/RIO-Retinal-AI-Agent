 %% =======================
% 7) STAGE B TRAINING (Polish) - continue from Stage A weights (ROBUST)
% ========================
if ~isfolder(ckptB), mkdir(ckptB); end

optionsB = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', stageB.maxEpochs, ...
    'InitialLearnRate', stageB.initialLR, ...
    'L2Regularization', l2Reg, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', validationFreq, ...
    'CheckpointPath', ckptB, ...
    'Verbose', true, ...
    'Plots','training-progress');

disp("=== Stage B: Polish fine-tuning started ===");

% Try to continue training using the trained weights from netA
try
    layersB = netA.Layers;  % carries learned weights (SeriesNetwork workflow)
    % Safety check: ensure it's a valid layer array
    if ~isa(layersB, 'nnet.cnn.layer.Layer')
        error("netA.Layers is not a valid layer array in this MATLAB version.");
    end

    trainedNet = trainNetwork(augTrain, layersB, optionsB);

catch ME
    warning("Stage B continuation via netA.Layers failed: %s", ME.message);

    % Fallback (still improves results): do another training pass with LR schedule in one stage
    % You can skip Stage B safely if needed.
    disp("Skipping Stage B and using netA as final model.");
    trainedNet = netA;
end

disp("=== Stage B: Done ===");
