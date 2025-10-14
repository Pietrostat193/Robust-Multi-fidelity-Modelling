% === Single run on real PM2.5 data (corrected indexing) ===
% This script runs three different Gaussian Process (GP) models 
% (Likelihood, Robust 1, Robust 2) on HF and LF air quality data.
clear; clc;

%% 2) Load data
tmp      = load("out_from_R.mat");   % Contains 'out' struct from R processing
out      = tmp.out;
% R data layout (used in this script): [Time, Site_ID, x, y, Value]
HF_data  = out.HF;                   
LF_data  = out.LF;

%% 3) Base hyper-parameters / model settings
clear global ModelInfo
global ModelInfo
ModelInfo = struct();
ModelInfo.cov_type     = "RBF";         % Covariance function type
ModelInfo.kernel       = "RBF";         % Kernel function
ModelInfo.MeanFunction = "zero";        % Mean function type
ModelInfo.RhoFunction  = "constant";    % Rho function type
ModelInfo.combination  = "multiplicative"; % Combination strategy (e.g., multiplicative)
ModelInfo.jitter       = 1e-6;          % Numerical stability factor
ModelInfo.nn_size      = 100;           % Neural network size (if applicable)

% NOTE: Spatial subsampling (Section 4) is commented out/skipped to keep this simple
LF_sub = out.LF; % Use the full LF data

%% 5) Time subsample: Using all common timestamps (TEMPORAL SELECTION DISABLED)
% The temporal subsampling logic is commented out to use all available time points.
LF_sub = out.LF; 
HF_subtime  = HF_data;
LF_subtime  = LF_sub;

% Define keep_times to include all times present in the training data 
% FIX: Time index is COLUMN 1 in R data, NOT column 2 (Site ID).
keep_times = unique(LF_subtime(:,1)); 

%% 6) Build ModelInfo (training data)
% The R data order is [Time (1), Site_ID (2), x (3), y (4), Value (5)]

% HF Training Data (Inputs: Site_ID, time, x | Output: value)
% FIX: Reorder to match the intended logic [HF_site, time, x] using columns [2, 1, 3]
ModelInfo.X_H = HF_subtime(:, [2, 1, 3]); % Inputs: [HF_site, time, x] 
ModelInfo.y_H = HF_subtime(:, 5);         % Output: value (Column 5)

% LF Training Data (Inputs: Time, Site_ID, x, y | Output: value)
% Columns 1:4 contain [Time, Site_ID, x, y]
ModelInfo.X_L = LF_subtime(:, 1:4);       % Inputs: [Time, Site_ID, x, y] 
ModelInfo.y_L = LF_subtime(:, 5);         % Output: value (Column 5)

%% 7) Test set -- NOW on LF (matching LF input format)
% Assuming out.LF_test has layout [Time, Site_ID, x, y, value]
if isfield(out, 'LF_test')
    % FIX: Time index is COLUMN 1 in R data.
    LFtest_times = out.LF_test(:,1); 
    
    % Filter test set to only include times present in the training set
    test_mask    = ismember(LFtest_times, keep_times); 
    
    % Test Inputs: Columns 1:4 contain [Time, Site_ID, x, y]
    X_test       = out.LF_test(test_mask, 1:4);   
    % Test Outputs: Column 5 contains Value
    y_test       = out.LF_test(test_mask, 5);
else
    error('out.LF_test not found. Please provide LF_test in "out".');
end

% Store for result bundle
result = struct();
result.y_test = y_test;
result.out    = out;
result.keep_times = keep_times;

%% 8) Optimisation setup
% NOTE: The hyp_init length (11) is an assumption for the 2Dsp model.
hyp_init = rand(11,1); 
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','iter', ...
    'TolFun',1e-12, ...
    'TolX',1e-12, ...
    'MaxFunctionEvaluations',2000);

%% 9) Fit: likelihood2Dsp (classic)
try
    [hyp_lik, fval_lik, ef_lik] = fminunc(@likelihood2Dsp, hyp_init, options);
    ModelInfo.hyp   = hyp_lik;
    % Assuming predict2Dsp uses ModelInfo.hyp
    yhat_lik        = predict2Dsp(X_test); 
    result.hyp_lik        = hyp_lik;
    result.yhat_lik       = yhat_lik;
    result.fval_lik       = fval_lik;
    result.exitflag_lik   = ef_lik;
catch ME
    warning('Likelihood fit error: %s', ME.message);
    result.hyp_lik        = NaN(size(hyp_init));
    result.yhat_lik       = NaN(size(y_test));
    result.fval_lik       = NaN;
    result.exitflag_lik   = -999;
end

%% 10) Fit: robustObjective2Dsp (robust #1)
try
    [hyp_robust1, fval_r1, ef_r1] = fminunc(@robustObjective2Dsp, hyp_init, options);
    ModelInfo.hyp    = hyp_robust1;
    yhat_robust1     = predict2Dsp(X_test);
    result.hyp_robust1      = hyp_robust1;
    result.yhat_robust1     = yhat_robust1;
    result.fval_robust1     = fval_r1;
    result.exitflag_robust1 = ef_r1;
catch ME
    warning('Robust1 fit error: %s', ME.message);
    result.hyp_robust1      = NaN(size(hyp_init));
    result.yhat_robust1     = NaN(size(y_test));
    result.fval_robust1     = NaN;
    result.exitflag_robust1 = -999;
end

%% 11) Fit: robustObjective2Dsp2 (robust #2)
try
    [hyp_robust2, fval_r2, ef_r2] = fminunc(@robustObjective2Dsp2, hyp_init, options);
    ModelInfo.hyp    = hyp_robust2;
    yhat_robust2     = predict2Dsp(X_test);
    result.hyp_robust2      = hyp_robust2;
    result.yhat_robust2     = yhat_robust2;
    result.fval_robust2     = fval_r2;
    result.exitflag_robust2 = -999;
catch ME
    warning('Robust2 fit error: %s', ME.message);
    result.hyp_robust2      = NaN(size(hyp_init));
    result.yhat_robust2     = NaN(size(y_test));
    result.fval_robust2     = NaN;
    result.exitflag_robust2 = -999;
end

%% 12) Metrics for all three models (MAE & RMSE)
% Ensure variables are double for numerical stability and correctness
y_test_d = double(result.y_test);

% Likelihood
if isfield(result, 'yhat_lik')
    yhat_lik_d = double(result.yhat_lik);
    mae_lik  = mean(abs(yhat_lik_d - y_test_d), 'omitnan');
    rmse_lik = sqrt(mean((yhat_lik_d - y_test_d).^2, 'omitnan'));
else
    mae_lik = NaN; rmse_lik = NaN;
end

% Robust 1
if isfield(result, 'yhat_robust1')
    yhat_r1_d = double(result.yhat_robust1);
    mae_r1  = mean(abs(yhat_r1_d - y_test_d), 'omitnan');
    rmse_r1 = sqrt(mean((yhat_r1_d - y_test_d).^2, 'omitnan'));
else
    mae_r1 = NaN; rmse_r1 = NaN;
end

% Robust 2
if isfield(result, 'yhat_robust2')
    yhat_r2_d = double(result.yhat_robust2);
    mae_r2  = mean(abs(yhat_r2_d - y_test_d), 'omitnan');
    rmse_r2 = sqrt(mean((yhat_r2_d - y_test_d).^2, 'omitnan'));
else
    mae_r2 = NaN; rmse_r2 = NaN;
end

result.metrics = table(mae_lik, rmse_lik, mae_r1, rmse_r1, mae_r2, rmse_r2, ...
    'VariableNames', {'mae_likelihood','rmse_likelihood', ...
                      'mae_robust1','rmse_robust1', ...
                      'mae_robust2','rmse_robust2'});
% -------------------------------------------------------------------------

%% 13) Prediction CSV Export
% X_test columns: [Time, Site_ID, x, y] as per R data.
T_predictions = table(...
    X_test(:,2), ... % Site_ID (Column 2)
    X_test(:,1), ... % Time_Index (Column 1)
    X_test(:,3), ... % X_Coord (Column 3)
    X_test(:,4), ... % Y_Coord (Column 4)
    result.y_test, ...
    result.yhat_lik, ...
    result.yhat_robust1, ...
    result.yhat_robust2, ...
    'VariableNames', {'Site_ID', 'Time_Index', 'X_Coord', 'Y_Coord', 'True_Value', ...
                      'Pred_Likelihood', 'Pred_Robust1', 'Pred_Robust2'} ...
);
% Write the table to a CSV file
output_filename = 'predictions_output_corrected.csv';
writetable(T_predictions, output_filename);
fprintf('Predictions saved to: %s\n', output_filename);
% -------------------------------------------------------------------------

%% 14) Save & report
save('results_real_run_corrected.mat', 'result', '-v7.3');
fprintf('Single run finished. Metrics:\n');
disp(result.metrics);
