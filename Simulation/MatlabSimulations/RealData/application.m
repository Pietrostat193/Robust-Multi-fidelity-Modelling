%% === Single run on real PM2.5 data ===
clear; clc;

%% 1) Paths
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\ClassicMFGP_Utilities");
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\3D-Example");

%% 2) Load data
tmp      = load("out_from_R.mat");   % contains 'out' struct
HFwrap   = load("HF_clean_noNa.mat");% contains 'HF_clean_noNa'
out      = tmp.out;
HF_data  = HFwrap.HF_clean_noNa;     % [HF_site, time, x, y, value]

%% 3) Base hyper-parameters / model settings
clear global ModelInfo
global ModelInfo
ModelInfo = struct();

ModelInfo.cov_type     = "RBF";         % baseline compatibility
ModelInfo.kernel       = "RBF";
ModelInfo.MeanFunction = "zero";
ModelInfo.RhoFunction  = "constant";
ModelInfo.combination  = "multiplicative";
ModelInfo.jitter       = 1e-6;
ModelInfo.nn_size      = 100;

%% 4) Spatial subsample: keep LF rows from 15 nearest LF sites around each HF site
% Layout reminder:
% out.HF, out.LF: [site_id, time, x, y, value]
[HF_sites, iaHF] = unique(out.HF(:,1), 'stable');
HF_coords        = out.HF(iaHF, 3:4);    % x,y
nH               = numel(HF_sites);

[LF_sites, iaLF] = unique(out.LF(:,1), 'stable');
LF_coords        = out.LF(iaLF, 3:4);    % x,y
nL               = numel(LF_sites);

% Distances HF x LF
D = zeros(nH, nL);
for i = 1:nH
    diffs   = LF_coords - HF_coords(i,:);
    D(i,:)  = sqrt(sum(diffs.^2, 2));
end

% For each HF: take the 15 nearest LF sites (or all if <15)
k      = min(15, nL);
idx15  = zeros(nH, k);
for i = 1:nH
    [~, order] = sort(D(i,:), 'ascend');
    idx15(i,:) = order(1:k);
end

% Union of selected LF sites across all HF
LF_keep_sites = unique(LF_sites(idx15(:)));

% Subsample LF: keep all rows (all times) for the selected LF sites
LF_keep_mask = ismember(out.LF(:,1), LF_keep_sites);
LF_sub       = out.LF(LF_keep_mask, :);  % [LF_site, time, x, y, value]

%% 5) Time subsample: first 40 common timestamps between HF and LF_sub
tH = HF_data(:,2);
tL = LF_sub(:,2);

if isdatetime(tH) && isdatetime(tL)
    common_time = intersect(unique(tH), unique(tL));
else
    % numeric times: exact intersection (simple & minimal)
    common_time = intersect(unique(tH), unique(tL));
end

% qui cami puoi cambiare il numero di time points
common_time = sort(common_time);
kT          = min(130, numel(common_time));
keep_times  = common_time(1:kT);

HF_mask     = ismember(HF_data(:,2), keep_times);
LF_mask     = ismember(LF_sub(:,2),  keep_times);

HF_subtime  = HF_data(HF_mask, :);
LF_subtime  = LF_sub(LF_mask,  :);

%% 6) Build ModelInfo (training data)
% HF: use [site_id, time, x] as inputs (as in your earlier code), y = value
ModelInfo.X_H = HF_subtime(:, 1:3);   % [HF_site, time, x]
ModelInfo.y_H = HF_subtime(:, 5);     % value

% LF: use [site_id, time, x, y] as inputs, y = value
ModelInfo.X_L = LF_subtime(:, 1:4);   % [LF_site, time, x, y]
ModelInfo.y_L = LF_subtime(:, 5);     % value

%% 7) Test set
% Assuming out.HF_test has layout [site_id, time, x, y, value]
X_test = out.HF_test(keep_times, 1:3);   % keep consistent with X_H choice above
y_test = out.HF_test(keep_times, 5);

% Store for result bundle
result = struct();
result.y_test = y_test;
result.out    = out;

%% 8) Optimisation setup
hyp_init = rand(11,1); % initial guess (dimension must match your likelihood)
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','iter', ...
    'TolFun',1e-12, ...
    'TolX',1e-12, ...
    'MaxFunctionEvaluations',2000);

%% 9) Fit: Classic
try
    [hyp_classic, fval_c, ef_c] = fminunc(@likelihood2Dsp, hyp_init, options);
    ModelInfo.hyp   = hyp_classic;
    yhat_classic    = predict2Dsp(X_test);
    result.hyp_classic     = hyp_classic;
    result.yhat_classic    = yhat_classic;
    result.fval_classic    = fval_c;
    result.exitflag_classic= ef_c;
catch ME
    warning('Classic fit error: %s', ME.message);
    result.hyp_classic      = NaN(size(hyp_init));
    result.yhat_classic     = NaN(size(y_test));
    result.fval_classic     = NaN;
    result.exitflag_classic = -999;
end

%% 10) Fit: Robust
try
    [hyp_robust, fval_r, ef_r] = fminunc(@robustObjective2Dsp, hyp_init, options);
    ModelInfo.hyp    = hyp_robust;
    yhat_robust      = predict2Dsp(X_test);
    result.hyp_robust      = hyp_robust;
    result.yhat_robust     = yhat_robust;
    result.fval_robust     = fval_r;
    result.exitflag_robust = ef_r;
catch ME
    warning('Robust fit error: %s', ME.message);
    result.hyp_robust      = NaN(size(hyp_init));
    result.yhat_robust     = NaN(size(y_test));
    result.fval_robust     = NaN;
    result.exitflag_robust = -999;
end

%% 11) Metrics
if isfield(result, 'yhat_classic')
    mae_classic  = mean(abs(result.yhat_classic - result.y_test), 'omitnan');
    rmse_classic = sqrt(mean((result.yhat_classic - result.y_test).^2, 'omitnan'));
else
    mae_classic = NaN; rmse_classic = NaN;
end

if isfield(result, 'yhat_robust')
    mae_robust  = mean(abs(result.yhat_robust - result.y_test), 'omitnan');
    rmse_robust = sqrt(mean((result.yhat_robust - result.y_test).^2, 'omitnan'));
else
    mae_robust = NaN; rmse_robust = NaN;
end

result.metrics = table(mae_classic, rmse_classic, mae_robust, rmse_robust, ...
    'VariableNames', {'mae_classic','rmse_classic','mae_robust','rmse_robust'});

%% 12) Save & report
%save('results_real_run.mat', 'result', '-v7.3');
fprintf('Single run finished. Metrics:\n');
disp(result.metrics);
