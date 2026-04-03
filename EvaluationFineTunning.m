 %% Eval_StageA_StageB_INLINE_NoFunctions.m
% ------------------------------------------------------------
% NO local functions. NO separate .m helpers.
% تقييم StageA و StageB (fine-tuned) + تصدير CSV للإكسل
%
% Required in workspace:
%   netA, trainedNet, augTest, imdsTest, classNames
% ------------------------------------------------------------

clc; close all;

% ---- Check required variables
reqVars = {'netA','trainedNet','augTest','imdsTest','classNames'};
for k = 1:numel(reqVars)
    if ~evalin('base', sprintf("exist('%s','var')", reqVars{k}))
        error("Missing variable in workspace: %s. Run training script first (Stage A & B).", reqVars{k});
    end
end

% ---- Pull variables from base workspace
netA       = evalin('base','netA');
trainedNet = evalin('base','trainedNet');
augTest    = evalin('base','augTest');
imdsTest   = evalin('base','imdsTest');
classNames = evalin('base','classNames');

YTrue = imdsTest.Labels;
classNames = string(classNames);
K = numel(classNames);

% Make sure figures are visible
set(0,'DefaultFigureVisible','on');

%% =========================
% STAGE A EVALUATION (INLINE)
% =========================
tag = "StageA";

[YPredA, scoresA] = classify(netA, augTest);
accA = mean(YPredA == YTrue);

% Confusion matrix numeric
[cmA, ~] = confusionmat(YTrue, YPredA, 'Order', categorical(classNames));

% Confusion chart
figure('Name', tag + " - Confusion Matrix");
cA = confusionchart(YTrue, YPredA);
cA.Title = tag + " - Confusion Matrix (Test Set)";
cA.RowSummary = 'row-normalized';
cA.ColumnSummary = 'column-normalized';

% Confusion table (Excel-ready)
ConfA = array2table(cmA, ...
    'VariableNames', strcat("Pred_", classNames), ...
    'RowNames', strcat("True_", classNames));
disp("StageA Confusion Matrix (numeric):"); disp(ConfA);

% Metrics per class
TP = diag(cmA);
FP = sum(cmA,1)' - TP;
FN = sum(cmA,2)  - TP;
TN = sum(cmA(:)) - (TP + FP + FN);

precision = TP ./ (TP + FP);  precision((TP+FP)==0) = NaN;
recall    = TP ./ (TP + FN);  recall((TP+FN)==0) = NaN;
f1        = (2*precision.*recall) ./ (precision + recall);  f1((precision+recall)==0) = NaN;
specificity = TN ./ (TN + FP); specificity((TN+FP)==0) = NaN;
perClassAcc = (TP + TN) ./ (TP + TN + FP + FN); perClassAcc((TP+TN+FP+FN)==0) = NaN;
support = sum(cmA,2);

EvalA = table(classNames(:), support, precision, recall, f1, specificity, perClassAcc, ...
    'VariableNames', {'Class','Support','Precision','Recall','F1','Specificity','PerClassAccuracy'});

% Macro/Weighted
macroA = [mean(precision,'omitnan'), mean(recall,'omitnan'), mean(f1,'omitnan'), mean(specificity,'omitnan'), mean(perClassAcc,'omitnan')];
w = support / sum(support);
weightedA = [sum(w.*precision,'omitnan'), sum(w.*recall,'omitnan'), sum(w.*f1,'omitnan'), sum(w.*specificity,'omitnan'), sum(w.*perClassAcc,'omitnan')];

MacroRowA = table("MacroAvg", sum(support), macroA(1), macroA(2), macroA(3), macroA(4), macroA(5), ...
    'VariableNames', {'Class','Support','Precision','Recall','F1','Specificity','PerClassAccuracy'});
WeightedRowA = table("WeightedAvg", sum(support), weightedA(1), weightedA(2), weightedA(3), weightedA(4), weightedA(5), ...
    'VariableNames', {'Class','Support','Precision','Recall','F1','Specificity','PerClassAccuracy'});

EvalA_full = [EvalA; MacroRowA; WeightedRowA];
disp("StageA Evaluation Table:"); disp(EvalA_full);

% ROC + AUC (One-vs-All)
YTrueStr = string(YTrue);
aucA = zeros(K,1);

figure('Name', tag + " - ROC Curves");
hold on; grid on;
for i=1:K
    yBin = double(YTrueStr == classNames(i));
    s = scoresA(:,i);
    [Xroc, Yroc, ~, AUC] = perfcurve(yBin, s, 1);
    aucA(i) = AUC;
    plot(Xroc, Yroc, 'LineWidth', 1.5);
end
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(tag + " - ROC Curves (One-vs-All)");
legend(compose("%s (AUC=%.3f)", classNames, aucA), 'Location','SouthEast');
hold off;

AUCTableA = table(classNames(:), aucA, 'VariableNames', {'Class','AUC'});
disp("StageA AUC Table:"); disp(AUCTableA);

% Export CSV (Excel-ready)
writetable(table(accA,'VariableNames',{'OverallAccuracy'}), "StageA_OverallAccuracy.csv");
writetable(ConfA, "StageA_ConfusionMatrix.csv", "WriteRowNames", true);
writetable(EvalA_full, "StageA_EvaluationMetrics.csv");
writetable(AUCTableA, "StageA_AUC_Table.csv");
disp("StageA CSV exported.");

%% =========================
% STAGE B EVALUATION (INLINE)
% =========================
tag = "StageB";

[YPredB, scoresB] = classify(trainedNet, augTest);
accB = mean(YPredB == YTrue);

[cmB, ~] = confusionmat(YTrue, YPredB, 'Order', categorical(classNames));

figure('Name', tag + " - Confusion Matrix");
cB = confusionchart(YTrue, YPredB);
cB.Title = tag + " - Confusion Matrix (Test Set)";
cB.RowSummary = 'row-normalized';
cB.ColumnSummary = 'column-normalized';

ConfB = array2table(cmB, ...
    'VariableNames', strcat("Pred_", classNames), ...
    'RowNames', strcat("True_", classNames));
disp("StageB Confusion Matrix (numeric):"); disp(ConfB);

TP = diag(cmB);
FP = sum(cmB,1)' - TP;
FN = sum(cmB,2)  - TP;
TN = sum(cmB(:)) - (TP + FP + FN);

precision = TP ./ (TP + FP);  precision((TP+FP)==0) = NaN;
recall    = TP ./ (TP + FN);  recall((TP+FN)==0) = NaN;
f1        = (2*precision.*recall) ./ (precision + recall);  f1((precision+recall)==0) = NaN;
specificity = TN ./ (TN + FP); specificity((TN+FP)==0) = NaN;
perClassAcc = (TP + TN) ./ (TP + TN + FP + FN); perClassAcc((TP+TN+FP+FN)==0) = NaN;
support = sum(cmB,2);

EvalB = table(classNames(:), support, precision, recall, f1, specificity, perClassAcc, ...
    'VariableNames', {'Class','Support','Precision','Recall','F1','Specificity','PerClassAccuracy'});

macroB = [mean(precision,'omitnan'), mean(recall,'omitnan'), mean(f1,'omitnan'), mean(specificity,'omitnan'), mean(perClassAcc,'omitnan')];
w = support / sum(support);
weightedB = [sum(w.*precision,'omitnan'), sum(w.*recall,'omitnan'), sum(w.*f1,'omitnan'), sum(w.*specificity,'omitnan'), sum(w.*perClassAcc,'omitnan')];

MacroRowB = table("MacroAvg", sum(support), macroB(1), macroB(2), macroB(3), macroB(4), macroB(5), ...
    'VariableNames', {'Class','Support','Precision','Recall','F1','Specificity','PerClassAccuracy'});
WeightedRowB = table("WeightedAvg", sum(support), weightedB(1), weightedB(2), weightedB(3), weightedB(4), weightedB(5), ...
    'VariableNames', {'Class','Support','Precision','Recall','F1','Specificity','PerClassAccuracy'});

EvalB_full = [EvalB; MacroRowB; WeightedRowB];
disp("StageB Evaluation Table:"); disp(EvalB_full);

aucB = zeros(K,1);
figure('Name', tag + " - ROC Curves");
hold on; grid on;
for i=1:K
    yBin = double(YTrueStr == classNames(i));
    s = scoresB(:,i);
    [Xroc, Yroc, ~, AUC] = perfcurve(yBin, s, 1);
    aucB(i) = AUC;
    plot(Xroc, Yroc, 'LineWidth', 1.5);
end
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(tag + " - ROC Curves (One-vs-All)");
legend(compose("%s (AUC=%.3f)", classNames, aucB), 'Location','SouthEast');
hold off;

AUCTableB = table(classNames(:), aucB, 'VariableNames', {'Class','AUC'});
disp("StageB AUC Table:"); disp(AUCTableB);

writetable(table(accB,'VariableNames',{'OverallAccuracy'}), "StageB_OverallAccuracy.csv");
writetable(ConfB, "StageB_ConfusionMatrix.csv", "WriteRowNames", true);
writetable(EvalB_full, "StageB_EvaluationMetrics.csv");
writetable(AUCTableB, "StageB_AUC_Table.csv");
disp("StageB CSV exported.");

%% =========================
% COMPARISON TABLE (StageA vs StageB)
% =========================
Comp = table();
Comp.Metric = ["OverallAccuracy"; "MacroF1"; "WeightedF1"; "AUC_"+classNames(:)];
Comp.StageA = [accA; MacroRowA.F1; WeightedRowA.F1; aucA];
Comp.StageB = [accB; MacroRowB.F1; WeightedRowB.F1; aucB];
Comp.Delta  = Comp.StageB - Comp.StageA;

disp("StageA vs StageB Comparison:"); disp(Comp);
writetable(Comp, "StageA_vs_StageB_Comparison.csv");
disp("Comparison CSV exported: StageA_vs_StageB_Comparison.csv");
