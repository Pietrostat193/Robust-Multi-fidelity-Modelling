%% delta_sensitivity_experiment.m
% Simulation experiment: sensitivity of robust predictions to Huber delta.
% Compares 5 models over 3 distortion scenarios, 100 runs each.
% Requires your utilities:
%   simulate_data, distort_sample, likelihood2Dsp, predict2Dsp
% and Optimization Toolbox (fminunc).

clear; clc; close all;

%% --- Paths (adjust to your machine) ---
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\Simulation_utilities")
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\ClassicMFGP_Utilities")
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\RobustMF_Utilities")

%% --- Solver options ---
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'TolFun',1e-12, ...
    'TolX',1e-12, ...
    'MaxFunctionEvaluations', 4000);

%% --- Global ModelInfo (used by your utilities) ---
clear global ModelInfo
global ModelInfo

%% --- Experiment design ---
N_RUNS = 100;                  % repetitions per scenario

% Distortion scenarios (LF distorted on test subset only)
scenarios = struct( ...
  'name', {"m=2,a=0.1","m=4,a=0.3","m=20,a=0.7"}, ...
  'm',    {2, 4, 20}, ...
  'a',    {0.1, 0.3, 0.7} ...
);

% Five model conditions
models = struct( ...
  'name', {"classic","robust_d_small","robust_d_medium","robust_d_big","robust_d_auto"}, ...
  'type', {"classic","robust_fixed","robust_fixed","robust_fixed","robust_auto"}, ...
  'delta', {NaN, 0.05, 0.35, 1.50, NaN} ...   % tune if desired; covers small→large
);

% Robust knobs (same as your single-run script)
ModelInfo.c           = 2.0;     % inflate delta to avoid over-robustness
ModelInfo.delta_min   = 1e-3;    % small floor
ModelInfo.cov_type    = "RBF";
ModelInfo.combination = "multiplicative";
ModelInfo.jitter      = 1e-6;

% Hyperparameter init length (match your kernel setup)
HYP_DIM = 11;

%% --- Storage for results (long format) ---
results = struct([]);   % append on success only
row = 0;

%% --- Main loops: scenario × run × model ---
base_seed = 1209375;

for s = 1:numel(scenarios)
    m = scenarios(s).m;
    a = scenarios(s).a;
    fprintf('\n=== Scenario %d/%d: %s ===\n', s, numel(scenarios), scenarios(s).name);

    for r = 1:N_RUNS   % (was 8:N_RUNS in your snippet)
        % Seed this run (keeps runs reproducible per scenario)
        rng(base_seed + 1000*s + r);

        % --- Simulate data (80% stations train) ---
        out = simulate_data(base_seed + 1000*s + r, 0.8);

        % --- Distort LF on test subset only ---
        LF_dist = out.LF;
        distorted_subset = distort_sample(out.LF(out.test_row_idx, :), a, m, base_seed + 2000*s + r);
        LF_dist(out.test_row_idx, :) = distorted_subset;
        out.distorted = LF_dist;

        % --- Build ModelInfo inputs/outputs ---
        ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
        ModelInfo.y_H = out.HF_train.fH;

        ModelInfo.X_L = [out.distorted.t, out.distorted.s1, out.distorted.s2];
        ModelInfo.y_L = out.distorted.fL;

        % Test set
        X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
        y_test = out.HF_test.fH;

        % Random initial hyperparameters (fixed across models within run)
        rng(1);                                    % same init per run (fair)
        hyp_init = rand(HYP_DIM, 1);

        % --- Fit & evaluate each model ---
        for j = 1:numel(models)

            % Reset robust flags each time
            ModelInfo.use_fixed_delta = false;

            % Prepare per-model bookkeeping
            model_name = models(j).name;
            model_type = models(j).type;
            delta_base = NaN;
            delta_eff  = NaN;
            yhat = [];

            try
                switch model_type
                    case "classic"
                        % CLASSIC fit
                        [hyp_j, ~] = fminunc(@likelihood2Dsp, hyp_init, options);
                        ModelInfo.hyp = hyp_j;
                        yhat = predict2Dsp(X_test);

                    case "robust_fixed"
                        % ROBUST with fixed delta
                        ModelInfo.use_fixed_delta = true;
                        ModelInfo.delta = models(j).delta;   % base delta
                        delta_base = ModelInfo.delta;
                        delta_eff  = max(ModelInfo.c * ModelInfo.delta, ModelInfo.delta_min);

                        [hyp_j, ~] = fminunc(@robustObjective2Dsp, hyp_init, options);
                        ModelInfo.hyp = hyp_j;
                        yhat = predict2Dsp(X_test);

                    case "robust_auto"
                        % ROBUST auto (MAD-based)
                        ModelInfo.use_fixed_delta = false;

                        [hyp_j, ~] = fminunc(@robustObjective2Dsp, hyp_init, options);
                        ModelInfo.hyp = hyp_j;
                        yhat = predict2Dsp(X_test);

                        % log the effective delta implied by final fit
                        delta_eff = compute_delta_auto_eff();

                    otherwise
                        error('Unknown model type: %s', model_type);
                end

                % Basic sanity check on predictions
                if numel(yhat) ~= numel(y_test)
                    error('Prediction length mismatch: got %d, expected %d.', numel(yhat), numel(y_test));
                end

            catch ME
                warning('Fit FAILED — Scenario=%s | Run=%d | Model=%s | Msg: %s', ...
                        scenarios(s).name, r, model_name, ME.message);
                continue;  % skip storing this (append on success only)
            end

            % --- If we get here, the fit & predict succeeded ---
            row = row + 1;
            MAE  = mean(abs(yhat - y_test));
            RMSE = sqrt(mean((yhat - y_test).^2));

            % Record
            results(row).scenario   = scenarios(s).name;
            results(row).m          = m;
            results(row).a          = a;
            results(row).run        = r;
            results(row).model      = model_name;
            results(row).delta_base = delta_base;
            results(row).delta_eff  = delta_eff;
            results(row).MAE        = MAE;
            results(row).RMSE       = RMSE;

        end % model loop

        if mod(r, max(1, round(N_RUNS/10))) == 0
            fprintf('  Run %3d/%3d done.\n', r, N_RUNS);
        end
    end % run loop
end % scenario loop

%% --- Convert results to table & summarise ---
if isempty(results)
    error('No successful fits recorded. Check paths and utilities.');
end

T = struct2table(results);

% Summary: mean +/- std and 95% CI per scenario × model
G = findgroups(T.scenario, T.model);
summ = table;
summ.scenario = splitapply(@(x) x(1), T.scenario, G);
summ.model    = splitapply(@(x) x(1), T.model,    G);
summ.n        = splitapply(@numel,     T.MAE,     G);
summ.MAE_mean = splitapply(@mean,      T.MAE,     G);
summ.MAE_std  = splitapply(@std,       T.MAE,     G);
summ.RMSE_mean= splitapply(@mean,      T.RMSE,    G);
summ.RMSE_std = splitapply(@std,       T.RMSE,    G);

% 95% CI via normal approx
z = 1.96;
summ.MAE_ci  = z * (summ.MAE_std  ./ sqrt(summ.n));
summ.RMSE_ci = z * (summ.RMSE_std ./ sqrt(summ.n));

disp('--- Summary (mean ± 95% CI) ---');
for i = 1:height(summ)
    fprintf('%-12s | %-16s | MAE %.4f ± %.4f | RMSE %.4f ± %.4f (n=%d)\n', ...
        summ.scenario{i}, summ.model{i}, ...
        summ.MAE_mean(i), summ.MAE_ci(i), ...
        summ.RMSE_mean(i), summ.RMSE_ci(i), summ.n(i));
end

%% --- Save to disk ---
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
writetable(T,    sprintf('delta_sensitivity_runs_%s.csv',    timestamp));
writetable(summ, sprintf('delta_sensitivity_summary_%s.csv', timestamp));

%% --- Quick plots (per scenario) ---
models_order = string({models.name});  % ensure string array for categories

for s = 1:numel(scenarios)
    % filter this scenario
    Tsc = T(strcmp(T.scenario, scenarios(s).name), :);

    % consistent categorical order
    g = categorical(string(Tsc.model), models_order, 'Ordinal', true);

    % MAE box
    figName = "MAE box: " + string(scenarios(s).name);   % string scalar
    figure('Name', char(figName));                       % Name expects scalar; char() is safe
    boxchart(g, Tsc.MAE);
    ylabel('MAE'); title(figName); grid on;

    % RMSE box
    figName = "RMSE box: " + string(scenarios(s).name);
    figure('Name', char(figName));
    boxchart(g, Tsc.RMSE);
    ylabel('RMSE'); title(figName); grid on;
end


%% ===========================
%% Helpers (same logic you used)
%% ===========================

function HL = robustObjective2Dsp(hyp)
% Robust objective with Huber loss.
% Uses either fixed delta from ModelInfo (use_fixed_delta==true)
% or auto-delta via MAD (as in your original sketch).
    global ModelInfo

    % current hyp for prediction and to populate ModelInfo internals
    ModelInfo.hyp = hyp;

    % Predict on HF training inputs to compute residuals
    y_hat_H = predict2Dsp(ModelInfo.X_H);
    resid_H = ModelInfo.y_H - y_hat_H;

    % Precision for held-out points
    if isfield(ModelInfo,'W_inv') && ~isempty(ModelInfo.W_inv)
        K = numel(resid_H);
        prec = diag(ModelInfo.W_inv(end-K+1:end, end-K+1:end)); % safer indexing
    else
        prec = ones(size(resid_H)); % fallback: unit precision
    end

    % Standardized residuals
    r = resid_H .* sqrt(prec);

    % Choose delta
    if isfield(ModelInfo,'use_fixed_delta') && ModelInfo.use_fixed_delta ...
            && isfield(ModelInfo,'delta') && ~isempty(ModelInfo.delta)
        delta_or_info = ModelInfo;  % huberLoss2 will read fields (c, delta_min, delta)
    else
        % Auto via MAD
        scale = median(abs(r)) / 0.6745;
        delta_auto = 1.345 * scale;
        tmp = ModelInfo; tmp.delta = delta_auto; % keep c & floor
        delta_or_info = tmp;
    end

    % Huberized loss
    loss_vals = huberLoss2(r, delta_or_info);
    HL = sum(loss_vals);
end

function L = huberLoss2(r, delta_or_info)
% Huber loss with optional inflation/floor via ModelInfo fields.
% Effective threshold: delta_eff = max(c * delta, delta_min)
    if isnumeric(delta_or_info)
        delta = delta_or_info; c = 1.0; delta_min = 0.0;
    else
        MI = delta_or_info;
        if ~isfield(MI,'delta'),     error('ModelInfo.delta is required'); end
        if ~isfield(MI,'c'),         MI.c = 2.0;       end
        if ~isfield(MI,'delta_min'), MI.delta_min = 1e-3; end
        delta = MI.delta; c = MI.c; delta_min = MI.delta_min;
    end

    delta_eff = max(c * delta, delta_min);
    a = abs(r);
    L = zeros(size(r));
    idx = (a <= delta_eff);

    % Quadratic region
    L(idx)  = 0.5 * r(idx).^2;
    % Linear region
    L(~idx) = delta_eff .* (a(~idx) - 0.5*delta_eff);
end

function w = huberWeights(r, delta_or_info)
% IRLS weights matching the above Huber:
%   w(r) = 1, if |r| <= delta_eff;  w(r) = delta_eff/|r|, otherwise.
    if isnumeric(delta_or_info)
        delta = delta_or_info; c = 1.0; delta_min = 0.0;
    else
        MI = delta_or_info;
        if ~isfield(MI,'delta'),     error('ModelInfo.delta is required'); end
        if ~isfield(MI,'c'),         MI.c = 2.0;       end
        if ~isfield(MI,'delta_min'), MI.delta_min = 1e-3; end
        delta = MI.delta; c = MI.c; delta_min = MI.delta_min;
    end
    delta_eff = max(c * delta, delta_min);

    a = abs(r);
    w = ones(size(r));
    out = (a > delta_eff) & (a > 0);
    w(out) = delta_eff ./ a(out);
end

function delta_eff = compute_delta_auto_eff()
% Compute the effective delta implied by the final robust auto fit
% (repeats the MAD step on current standardized residuals).
    global ModelInfo
    y_hat_H = predict2Dsp(ModelInfo.X_H);
    resid_H = ModelInfo.y_H - y_hat_H;

    if isfield(ModelInfo,'W_inv') && ~isempty(ModelInfo.W_inv)
        K = numel(resid_H);
        prec = diag(ModelInfo.W_inv(end-K+1:end, end-K+1:end));
    else
        prec = ones(size(resid_H));
    end
    r = resid_H .* sqrt(prec);

    scale = median(abs(r)) / 0.6745;
    delta_auto = 1.345 * scale;
    delta_eff = max(ModelInfo.c * delta_auto, ModelInfo.delta_min);
end
