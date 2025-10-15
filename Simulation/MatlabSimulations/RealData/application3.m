%% === Batched runs on real PM2.5 data (30-consecutive-time windows) ===
clear; clc;

%% 1) Paths
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\ClassicMFGP_Utilities");
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\3D-Example");
addpath 'C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\RobustMF_Utilities'

%% 2) Load data
tmp      = load("out_from_R.mat");    % contains 'out' struct
HFwrap   = load("HF_clean_noNa.mat"); % contains 'HF_clean_noNa'
out      = tmp.out;
HF_data  = HFwrap.HF_clean_noNa;      % [HF_site, time, x, y, value]

%% 3) Base hyper-parameters / model settings
clear global ModelInfo
global ModelInfo
ModelInfo = struct();
ModelInfo.cov_type     = "RBF";
ModelInfo.kernel       = "RBF";
ModelInfo.MeanFunction = "zero";
ModelInfo.RhoFunction  = "constant";
ModelInfo.combination  = "multiplicative";
ModelInfo.jitter       = 1e-6;

%% 4) Spatial subsample: keep LF rows from 15 nearest LF sites around each HF site
% out.HF, out.LF layout: [site_id, time, x, y, value]
[HF_sites, iaHF] = unique(out.HF(:,1), 'stable');
HF_coords        = out.HF(iaHF, 3:4);    % x,y
nH               = numel(HF_sites);

[LF_sites, iaLF] = unique(out.LF(:,1), 'stable');
LF_coords        = out.LF(iaLF, 3:4);    % x,y
nL               = numel(LF_sites);

% Distances HF x LF
D = zeros(nH, nL);
for i = 1:nH
    diffs  = LF_coords - HF_coords(i,:);
    D(i,:) = sqrt(sum(diffs.^2, 2));
end

% For each HF: 15 nearest LF sites (or all if <15)
k      = min(15, nL);
idx15  = zeros(nH, k);
for i = 1:nH
    [~, order] = sort(D(i,:), 'ascend');
    idx15(i,:) = order(1:k);
end

% Union of selected LF sites across all HF
LF_keep_sites = unique(LF_sites(idx15(:)));

% Subsample LF: keep all rows (all times) for selected LF sites
LF_keep_mask = ismember(out.LF(:,1), LF_keep_sites);
LF_sub       = out.LF(LF_keep_mask, :);  % [LF_site, time, x, y, value]

%% 5) Common times (we’ll process in disjoint windows of 30)
tH = HF_data(:,2);
tL = LF_sub(:,2);

common_time = sort(intersect(unique(tH), unique(tL)));

% Define non-overlapping windows of 30
winSize = 30;
nFull   = floor(numel(common_time) / winSize);
if nFull == 0
    error('Not enough common time points to form one %d-length window.', winSize);
end

%% 6) Optimisation setup (shared; we can warm-start across windows)
hyp_init = rand(11,1); % make sure this matches your likelihood dimensionality
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','iter', ...
    'TolFun',1e-12, ...
    'TolX',1e-12, ...
    'MaxFunctionEvaluations',2000);

%% 7) Performance logging setup
numIters = 4;                           % inner iterations per window
rowCount = nFull * numIters;            % total (outer * inner)

% Numeric log: [outer_idx, inner_idx, mae_classic, mae_robust1, mae_robust2]
MAE_log = nan(rowCount, 5);

% Optional detailed struct log (one entry per outer, inner)
Perf(rowCount,1) = struct( ...
    'outer', [], 'inner', [], ...
    'mae_classic', NaN, 'mae_robust1', NaN, 'mae_robust2', NaN, ...
    'fval_classic', NaN, 'fval_robust1', NaN, 'fval_robust2', NaN, ...
    'exitflag_classic', NaN, 'exitflag_robust1', NaN, 'exitflag_robust2', NaN, ...
    'time_classic', NaN, 'time_robust1', NaN, 'time_robust2', NaN ...
);

mae = @(yhat,y) mean(abs(yhat - y), 'omitnan');

row = 0; % running row index for logs

%% 8) Loop over time windows
for w = 1:nFull
    fprintf('\n=== Window %d/%d ===\n', w, nFull);
    idx_start  = (w-1)*winSize + 1;
    idx_end    = w*winSize;
    keep_times = common_time(idx_start:idx_end);

    % Subset HF/LF for this window
    HF_mask    = ismember(HF_data(:,2), keep_times);
    LF_mask    = ismember(LF_sub(:,2),  keep_times);

    HF_subtime = HF_data(HF_mask, :);   % [site, time, x, y, value]
    LF_subtime = LF_sub(LF_mask,  :);   % [site, time, x, y, value]

    % --- Train/Test splits for this window ---
    HF_selected = HF_subtime(HF_subtime(:,1) ~= 1, :);       % HF sites except site==1  (train H)
    Test        = HF_subtime(HF_subtime(:,1) == 1, [2,3,4]); % test inputs from HF site==1
    y_test      = HF_subtime(HF_subtime(:,1) == 1, 5);       % test targets

    % Build ModelInfo data (adjust indices to your needs)
    ModelInfo.X_H = HF_selected(:, [2,3,4]);   % [time, x, y] high-fidelity training
    ModelInfo.y_H = HF_selected(:, 5);
    ModelInfo.X_L = LF_subtime(:, [2,3,4]);    % [time, x, y] low-fidelity inputs
    ModelInfo.y_L = LF_subtime(:, 5);

    for i = 1:numIters
        row = row + 1;

        % --- Fit: Classic ---
        yhat_classic_test = nan(size(y_test));
        fval_c = NaN; ef_c = -999; tClassic = NaN;
        try
            t0 = tic;
            [hyp_classic, fval_c, ef_c] = fminunc(@likelihood2Dsp, hyp_init, options);
            tClassic = toc(t0);

            ModelInfo.hyp = hyp_classic;
            % Predict on test set (HF site==1 rows)
            yhat_classic_test = predict2Dsp(Test);

            % Warm start next inner iteration
            hyp_init = hyp_classic;
        catch ME
            warning('Classic fit error (outer %d, inner %d): %s', w, i, ME.message);
        end

        % --- Fit: Robust #1 ---
        yhat_robust1_test = nan(size(y_test));
        fval_r1 = NaN; ef_r1 = -999; tR1 = NaN;
        try
            t1 = tic;
            [hyp_robust1, fval_r1, ef_r1] = fminunc(@robustObjective2Dsp, hyp_classic, options);
            tR1 = toc(t1);

            ModelInfo.hyp = hyp_robust1;
            yhat_robust1_test = predict2Dsp(Test);
        catch ME
            warning('Robust1 fit error (outer %d, inner %d): %s', w, i, ME.message);
        end

        % --- Fit: Robust #2 ---
        yhat_robust2_test = nan(size(y_test));
        fval_r2 = NaN; ef_r2 = -999; tR2 = NaN;
        try
            t2 = tic;
            [hyp_robust2, fval_r2, ef_r2] = fminunc(@robustObjective2Dsp2, hyp_classic, options);
            tR2 = toc(t2);

            ModelInfo.hyp = hyp_robust2;
            yhat_robust2_test = predict2Dsp(Test);
        catch ME
            warning('Robust2 fit error (outer %d, inner %d): %s', w, i, ME.message);
        end

        % --- Metrics on the held-out HF "Test" set ---
        mae_c  = mae(yhat_classic_test,  y_test);
        mae_r1 = mae(yhat_robust1_test, y_test);
        mae_r2 = mae(yhat_robust2_test, y_test);

        % --- Numeric log (requested final array) ---
        MAE_log(row, :) = [w, i, mae_c, mae_r1, mae_r2];

        % --- Optional detailed log ---
        Perf(row).outer             = w;
        Perf(row).inner             = i;
        Perf(row).mae_classic       = mae_c;
        Perf(row).mae_robust1       = mae_r1;
        Perf(row).mae_robust2       = mae_r2;
        Perf(row).fval_classic      = fval_c;
        Perf(row).fval_robust1      = fval_r1;
        Perf(row).fval_robust2      = fval_r2;
        Perf(row).exitflag_classic  = ef_c;
        Perf(row).exitflag_robust1  = ef_r1;
        Perf(row).exitflag_robust2  = ef_r2;
        Perf(row).time_classic      = tClassic;
        Perf(row).time_robust1      = tR1;
        Perf(row).time_robust2      = tR2;

        % --- Optional quick plot (test predictions vs actual) ---
        %{
        figure('Name', sprintf('Test predictions | outer %d, inner %d', w, i));
        plot(yhat_classic_test, 'DisplayName','Classic'); hold on;
        plot(yhat_robust1_test, 'DisplayName','Robust 1');
        plot(yhat_robust2_test, 'DisplayName','Robust 2');
        plot(y_test,            'DisplayName','Actual');
        grid on; box on; axis tight;
        xlabel('Sample'); ylabel('Value');
        legend('Location','best');
        title(sprintf('HF site==1 test | outer %d, inner %d', w, i));
        %}
    end
end

%% 9) Show/Save outputs
disp('MAE_log columns: [outer, inner, mae_classic, mae_robust1, mae_robust2]');
disp(MAE_log);

PerfTable = struct2table(Perf);
disp(PerfTable(1:min(10,height(PerfTable)), :)); % preview

% Optionally save:
% save('MAE_log.mat','MAE_log');
% save('Perf_detailed.mat','Perf','PerfTable');
