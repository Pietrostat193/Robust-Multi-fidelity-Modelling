%% delta_sensitivity_demo.m
% Sensitivity of robust predictions to Huber delta (single-file script)

clear; clc;

%% --- Paths (yours) ---
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\Simulation_utilities")
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\7_Classic_MFGP_utility")
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\RobustMF_Utilities")

%% --- Solver options ---
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'TolFun',1e-12, ...
    'TolX',1e-12, ...
    'MaxFunctionEvaluations',2000);

%% --- One scenario (you can loop later if you want) ---
m_list = [2, 5, 10];
a_list = [0.1, 0.3, 0.5];

a    = 0.7;
m    = 20;
seed = 1209375;

% Simulate data (80% stations train)
out = simulate_data(seed, 0.8);

% Distort LF on test subset only
LF_dist = out.LF;
distorted_subset = distort_sample(out.LF(out.test_row_idx,:), a, m, seed);
LF_dist(out.test_row_idx,:) = distorted_subset;
out.distorted = LF_dist;

%% --- Build ModelInfo (global) ---
clear global ModelInfo
global ModelInfo
ModelInfo = struct();

% Inputs/outputs
ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
ModelInfo.y_H = out.HF_train.fH;
ModelInfo.X_L = [out.distorted.t, out.distorted.s1, out.distorted.s2];
ModelInfo.y_L = out.distorted.fL;

% GP/meta fields used by your utilities
ModelInfo.cov_type    = "RBF";
ModelInfo.combination = "multiplicative";
ModelInfo.jitter      = 1e-6;

% Robust knobs (defaults)
ModelInfo.c         = 2.0;    % inflate delta to avoid over-robustness
ModelInfo.delta_min = 1e-3;   % small floor
ModelInfo.use_fixed_delta = false;  % default = auto (MAD)

% Test set
X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
y_test = out.HF_test.fH;

% Random init
rng(1);
hyp_init = rand(11,1);

%% --- Fit classic once ---
[hyp_classic, ~] = fminunc(@likelihood2Dsp, hyp_init, options);
ModelInfo.hyp = hyp_classic;
yhat_classic  = predict2Dsp(X_test);

%% --- Robust: sweep fixed delta values ---
% Choose a grid (covers small -> large; feel free to tweak)
delta_grid = [0.001 0.02 0.05 0.1 0.2 0.35 0.5 0.75 1.0 1.5 4.0];

% Storage
yhat_robust_mat = zeros(numel(y_test), numel(delta_grid));
hyp_robust_all  = zeros(numel(hyp_init), numel(delta_grid));
delta_eff_list  = zeros(size(delta_grid));

for k = 1:numel(delta_grid)
    % Fix delta for this run
    ModelInfo.use_fixed_delta = true;
    ModelInfo.delta = delta_grid(k);       % this is the *base* delta
    % record effective delta (after inflation/floor)
    delta_eff_list(k) = max(ModelInfo.c * ModelInfo.delta, ModelInfo.delta_min);

    % Re-start from same init (or warm start from previous if you prefer)
    [hyp_robust_k, ~] = fminunc(@robustObjective2Dsp, hyp_init, options);
    hyp_robust_all(:,k) = hyp_robust_k;

    % Predictions with this robust fit
    ModelInfo.hyp = hyp_robust_k;
    yhat_robust_mat(:,k) = predict2Dsp(X_test);
end

%% --- (Optional) One auto-delta run (MAD-based), to compare with your original)
ModelInfo.use_fixed_delta = false;
[hyp_robust_auto, ~] = fminunc(@robustObjective2Dsp, hyp_init, options);
ModelInfo.hyp = hyp_robust_auto;
yhat_robust_auto = predict2Dsp(X_test);

%% --- Plot: predictions vs true, showing sensitivity to delta ---
figure; hold on; grid on;
plot(y_test, 'k-', 'LineWidth', 1.25);                       % ground truth
plot(yhat_classic, 'Color', [0.4 0.4 0.4], 'LineStyle','--');% classic

% Plot a subset, but keep legend informative (you can plot all if you like)
max_to_plot = min(numel(delta_grid), 10);
for k = 1:max_to_plot
    plot(yhat_robust_mat(:,k), 'LineWidth', 1.0);
end
plot(yhat_robust_auto, 'LineWidth', 1.5);                    % auto-delta

% Legend labels
leg = ["True", "Classic"];
for k = 1:max_to_plot
    leg(end+1) = sprintf("Robust (\\delta=%.3g; \\delta_eff=%.3g)", ...
        delta_grid(k), delta_eff_list(k));
end
leg(end+1) = "Robust (auto-MAD)";
legend(leg, 'Location','bestoutside');
title('Prediction sensitivity to Huber \delta');
xlabel('Test point index'); ylabel('f_H prediction');

%% --- (Optional) Quick MAE/RMSE vs \delta plots ---
mae = @(a,b) mean(abs(a-b));
rmse = @(a,b) sqrt(mean((a-b).^2));

mae_classic = mae(yhat_classic, y_test);
rmse_classic = rmse(yhat_classic, y_test);

mae_fixed = zeros(size(delta_grid));
rmse_fixed = zeros(size(delta_grid));
for k = 1:numel(delta_grid)
    mae_fixed(k) = mae(yhat_robust_mat(:,k), y_test);
    rmse_fixed(k) = rmse(yhat_robust_mat(:,k), y_test);
end
mae_auto  = mae(yhat_robust_auto, y_test);
rmse_auto = rmse(yhat_robust_auto, y_test);

figure; 
subplot(2,1,1);
semilogx(delta_grid, mae_fixed, '-o', 'LineWidth', 1.0); hold on; grid on;
yline(mae_classic, '--'); yline(mae_auto, ':', 'LineWidth', 1.5);
xlabel('\delta (fixed)'); ylabel('MAE');
legend('Robust (fixed \delta)', 'Classic', 'Robust (auto-MAD)', 'Location','best');
title('MAE vs Huber \delta');

subplot(2,1,2);
semilogx(delta_grid, rmse_fixed, '-o', 'LineWidth', 1.0); hold on; grid on;
yline(rmse_classic, '--'); yline(rmse_auto, ':', 'LineWidth', 1.5);
xlabel('\delta (fixed)'); ylabel('RMSE');
legend('Robust (fixed \delta)', 'Classic', 'Robust (auto-MAD)', 'Location','best');
title('RMSE vs Huber \delta');

%% ===========================
%% Helpers (single-file)
%% ===========================

function HL = robustObjective2Dsp(hyp)
% Robust objective with Huber loss. Uses either fixed delta from ModelInfo
% (ModelInfo.use_fixed_delta==true) or auto-delta via MAD (your original).
    global ModelInfo

    % current hyp for prediction and to populate ModelInfo internals
    ModelInfo.hyp = hyp;

    % Predict on HF training inputs to compute residuals
    y_hat_H = predict2Dsp(ModelInfo.X_H);
    resid_H = ModelInfo.y_H - y_hat_H;

    % Precision for held-out points
    idx0 = numel(ModelInfo.y_L) + 1;
    % Guard in case W_inv is not ready (depends on your pipeline)
    if isfield(ModelInfo,'W_inv') && ~isempty(ModelInfo.W_inv)
        prec = diag(ModelInfo.W_inv(idx0:end, idx0:end));
    else
        % Fallback: unit precision
        prec = ones(size(resid_H));
    end

    % Standardized residuals
    r = resid_H .* sqrt(prec);

    % Choose delta
    if isfield(ModelInfo,'use_fixed_delta') && ModelInfo.use_fixed_delta ...
            && isfield(ModelInfo,'delta') && ~isempty(ModelInfo.delta)
        delta_or_info = ModelInfo;  % huberLoss2 will read fields (c, delta_min, delta)
    else
        % Auto via MAD (your original)
        % MAD: median(|Z|) = 0.6745 * sigma   =>  sigma_hat = median(|r|)/0.6745
        scale = median(abs(r)) / 0.6745;
        delta_auto = 1.345 * scale;
        % Pass numeric delta (huberLoss2 will still apply c & floor when ModelInfo exists)
        % Here we pass numeric, which uses c=1 and no floor. To keep your
        % "avoid over-robustness" behavior even in auto mode, wrap in ModelInfo:
        tmp = ModelInfo; tmp.delta = delta_auto; % keep c & delta_min
        delta_or_info = tmp;
    end

    % Huberized loss
    loss_vals = huberLoss2(r, delta_or_info);
    HL = sum(loss_vals);
end

function L = huberLoss2(r, delta_or_info)
% Huber loss with optional inflation/floor via ModelInfo fields.
% Usage:
%   L = huberLoss2(r, delta_scalar)
%   L = huberLoss2(r, ModelInfo)   % uses ModelInfo.delta (+c and delta_min)
%
% Effective threshold: delta_eff = max(c * delta, delta_min)

    % Determine delta, c, and floor
    if isnumeric(delta_or_info)
        delta = delta_or_info;
        c = 1.0;            % no inflation if only a scalar provided
        delta_min = 0.0;    % no floor if only a scalar provided
    else
        MI = delta_or_info;
        if ~isfield(MI,'delta'),     error('ModelInfo.delta is required'); end
        if ~isfield(MI,'c'),         MI.c = 2.0;       end
        if ~isfield(MI,'delta_min'), MI.delta_min = 1e-3; end
        delta = MI.delta;
        c = MI.c;
        delta_min = MI.delta_min;
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
    % Determine delta_eff as in huberLoss2
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
