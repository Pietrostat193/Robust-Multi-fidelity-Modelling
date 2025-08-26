%% === Experiment with try/catch/continue (skip failed runs) ===
clear; clc;

addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\Simulation_utilities")
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\7_Classic_MFGP_utility")
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\RobustFolder\RobustMF_Utilities")

m_list = [2, 5, 10];
a_list = [0.1, 0.3, 0.5];
n_runs_per_scenario = 100;

% storage (append on success only)
scenario_vec = [];
m_vec = []; a_vec = []; seed_vec = [];
mae_classic_vec = []; rmse_classic_vec = [];
mae_robust_vec  = []; rmse_robust_vec  = [];
yhat_classic_cell = {}; yhat_robust_cell = {}; y_test_cell = {}; out_cell = {};

options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'TolFun',1e-12, ...
    'TolX',1e-12, ...
    'MaxFunctionEvaluations',2000);

scenario_id = 0;
for im = 1:numel(m_list)
  for ia = 1:numel(a_list)
    scenario_id = scenario_id + 1;
    m = m_list(im);
    a = a_list(ia);

    for r = 1:n_runs_per_scenario
      seed = 1000*scenario_id + r;

      % 1) simulate + distort
    
        out = simulate_data(seed, 0.8);  % 80% stations train
        LF_dist = out.LF;
        distorted_subset = distort_sample(out.LF(out.test_row_idx,:), a, m, seed);
        LF_dist(out.test_row_idx,:) = distorted_subset;
        out.distorted = LF_dist;
    
      

      % 2) Build ModelInfo
     
        clear global ModelInfo
        global ModelInfo
        ModelInfo = struct();
        ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
        ModelInfo.y_H = out.HF_train.fH;
        ModelInfo.X_L = [out.distorted.t, out.distorted.s1, out.distorted.s2];
        ModelInfo.y_L = out.distorted.fL;
        ModelInfo.cov_type   = "RBF";
        ModelInfo.combination = "multiplicative";
        ModelInfo.jitter     = 1e-6;
        X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
        y_test = out.HF_test.fH;
        hyp_init = rand(11,1);
      

      % 3) Classic fit
      try
        [hyp_classic, ~, ef_c] = fminunc(@likelihood2Dsp, hyp_init, options);
        if ef_c <= 0, warning('Classic exitflag=%g -> skip', ef_c); continue; end
        ModelInfo.hyp = hyp_classic;
        yhat_classic  = predict2Dsp(X_test);
      catch ME
        warning('Classic fit failed (scen %d, m=%g, a=%g, run=%d): %s', ...
                 scenario_id, m, a, r, ME.message);
        continue
      end

      % 4) Robust fit
      try
        [hyp_robust, ~, ef_r] = fminunc(@robustObjective2Dsp, hyp_init, options);
        if ef_r <= 0, warning('Robust exitflag=%g -> skip', ef_r); continue; end
        ModelInfo.hyp = hyp_robust;
        yhat_robust   = predict2Dsp(X_test);
      catch ME
        warning('Robust fit failed (scen %d, m=%g, a=%g, run=%d): %s', ...
                 scenario_id, m, a, r, ME.message);
        continue
      end

      % 5) Metrics (success path only)
      mae_classic = mean(abs(yhat_classic - y_test));
      rmse_classic = sqrt(mean((yhat_classic - y_test).^2));
      mae_robust = mean(abs(yhat_robust - y_test));
      rmse_robust = sqrt(mean((yhat_robust - y_test).^2));

      % 6) Append results
      scenario_vec(end+1,1) = scenario_id; %#ok<*AGROW>
      m_vec(end+1,1) = m;
      a_vec(end+1,1) = a;
      seed_vec(end+1,1) = seed;

      mae_classic_vec(end+1,1) = mae_classic;
      rmse_classic_vec(end+1,1) = rmse_classic;
      mae_robust_vec(end+1,1) = mae_robust;
      rmse_robust_vec(end+1,1) = rmse_robust;

      yhat_classic_cell{end+1,1} = yhat_classic;
      yhat_robust_cell{end+1,1}  = yhat_robust;
      y_test_cell{end+1,1}       = y_test;
      out_cell{end+1,1}          = out;

      if mod(numel(scenario_vec),25)==0
        fprintf('Saved %3d runs | scen %d (m=%g,a=%g) | MAE_c=%.3f RMSE_c=%.3f | MAE_r=%.3f RMSE_r=%.3f\n', ...
                numel(scenario_vec), scenario_id, m, a, mae_classic, rmse_classic, mae_robust, rmse_robust);
      end
    end
  end
end

% Wrap to tables and save
metrics_tbl = table(scenario_vec, m_vec, a_vec, seed_vec, ...
    mae_classic_vec, rmse_classic_vec, mae_robust_vec, rmse_robust_vec, ...
    'VariableNames', {'scenario','m','a','seed','mae_classic','rmse_classic','mae_robust','rmse_robust'});

preds_tbl = table(scenario_vec, m_vec, a_vec, seed_vec, ...
    yhat_classic_cell, yhat_robust_cell, y_test_cell, out_cell, ...
    'VariableNames', {'scenario','m','a','seed','yhat_classic','yhat_robust','y_test','out'});

save('results_robust_experiment.mat', 'metrics_tbl', 'preds_tbl', '-v7.3');

fprintf('Finished. Successful runs saved: %d\n', height(metrics_tbl));
