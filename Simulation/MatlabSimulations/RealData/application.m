%% === Single run on real PM25 data (no skipping) ===
clear; clc;

% ------------------ LOAD real data ------------------
tmp = load("out_from_R.mat");  % load .mat file
out = tmp.out;                 % extract the inner struct

% ------------------ Build ModelInfo ------------------
clear global ModelInfo
global ModelInfo
ModelInfo = struct();

% HF_train numeric array: columns = [t, s1, s2, fH]
ModelInfo.X_H = out.HF_train(:,1:3);
ModelInfo.y_H = out.HF_train(:,4);

% LF numeric array: columns = [t, s1, s2, fL]
ModelInfo.X_L = out.LF(:,1:3);
ModelInfo.y_L = out.LF(:,4);

ModelInfo.cov_type   = "RBF";
ModelInfo.combination = "multiplicative";
ModelInfo.jitter     = 1e-6;

% HF_test numeric array: columns = [t, s1, s2, fH]
X_test = out.HF_test(:,1:3);
y_test = out.HF_test(:,4);

hyp_init = rand(11,1); % initial guess

% fminunc options
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','iter', ...
    'TolFun',1e-12, ...
    'TolX',1e-12, ...
    'MaxFunctionEvaluations',2000);

% Storage for results (single-run)
result = struct();
result.y_test = y_test;
result.out = out;

% ------------------ Classic fit ------------------
try
    [hyp_classic, fval_c, ef_c] = fminunc(@likelihood2Dsp, hyp_init, options);
    ModelInfo.hyp = hyp_classic;
    yhat_classic = predict2Dsp(X_test);
    result.hyp_classic = hyp_classic;
    result.yhat_classic = yhat_classic;
    result.fval_classic = fval_c;
    result.exitflag_classic = ef_c;
catch ME
    warning('Classic fit threw an error: %s', ME.message);
    result.hyp_classic = NaN(size(hyp_init));
    result.yhat_classic = NaN(size(y_test));
    result.fval_classic = NaN;
    result.exitflag_classic = -999;
end

% ------------------ Robust fit ------------------
try
    [hyp_robust, fval_r, ef_r] = fminunc(@robustObjective2Dsp, hyp_init, options);
    ModelInfo.hyp = hyp_robust;
    yhat_robust = predict2Dsp(X_test);
    result.hyp_robust = hyp_robust;
    result.yhat_robust = yhat_robust;
    result.fval_robust = fval_r;
    result.exitflag_robust = ef_r;
catch ME
    warning('Robust fit threw an error: %s', ME.message);
    result.hyp_robust = NaN(size(hyp_init));
    result.yhat_robust = NaN(size(y_test));
    result.fval_robust = NaN;
    result.exitflag_robust = -999;
end

% ------------------ Metrics ------------------
if isfield(result,'yhat_classic')
    mae_classic = mean(abs(result.yhat_classic - result.y_test), 'omitnan');
    rmse_classic = sqrt(mean((result.yhat_classic - result.y_test).^2, 'omitnan'));
else
    mae_classic = NaN; rmse_classic = NaN;
end

if isfield(result,'yhat_robust')
    mae_robust = mean(abs(result.yhat_robust - result.y_test), 'omitnan');
    rmse_robust = sqrt(mean((result.yhat_robust - result.y_test).^2, 'omitnan'));
else
    mae_robust = NaN; rmse_robust = NaN;
end

result.metrics = table(mae_classic, rmse_classic, mae_robust, rmse_robust, ...
    'VariableNames', {'mae_classic','rmse_classic','mae_robust','rmse_robust'});

% ------------------ Save & report ------------------
save('results_real_run.mat', 'result', '-v7.3');
fprintf('Single run finished. Metrics:\n');
disp(result.metrics);
