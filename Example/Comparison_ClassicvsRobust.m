%addpath("\RobustMF_utilities")
%addpath("\ClassicMF_utilities")
%Add the right path to the script pointing to the above folders



load("ModelInfo.mat")
%ModelInfo is an object equipped with test and training multifidelity data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment one - here data are not "extreme", so not outliers
% Still the robust version appear to be a better interpolant
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Run MF classic
options = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...
        'Display', 'iter', ...
        'TolFun', 1e-12, ...
        'TolX', 1e-12, ...
        'MaxFunctionEvaluations', 2000);    
    [hyp_classic, opt_classic] = fminunc(@likelihood2Dsp, hyp_init, options);
    ModelInfo.hyp = hyp_classic;
    likelihood2Dsp(hyp_classic);
    y_classic = predict2Dsp(ModelInfo.X_L);


  %Run MF Robust

  [hyp_robust, fval, exitflag, output] = fminunc(@robustObjective2Dsp, hyp_init, options);
  ModelInfo.hyp = hyp_robust;
  y_robust = predict2Dsp(ModelInfo.X_L);

%200-300 is the test set area
        figure;
p = 3;
idx = (p-1)*100 + (1:100);

% Plot the three series
plot(idx, y_classic(idx), '-o', 'DisplayName', 'Classic'); hold on;
plot(idx, y_robust(idx), '--x', 'DisplayName', 'Robust');
plot(idx, ModelInfo.data(idx,1), 'DisplayName', 'Real Observations HF');

% Compute mean absolute errors
mae_classic = mean(abs(y_classic(idx) - ModelInfo.data(idx,1)));
mae_robust  = mean(abs(y_robust(idx)  - ModelInfo.data(idx,1)));

% Title and labels
title(sprintf('Observations station %d', p));
xlabel('Index'); ylabel('Prediction');
legend('Location','best');
grid on;

% Add text box with errors
str = sprintf('MAE Classic = %.3f\nMAE Robust = %.3f', mae_classic, mae_robust);
xpos = idx(1) + 5;        % adjust as needed
ypos = max(ModelInfo.data(idx,1)) * 0.9;  % adjust placement
text(xpos, ypos, str, 'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%###################################
% Experiment 2 - here LF data are extreme, 
% Still the robust version appear to be a much better interpolant
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####################################
global ModelInfo
%I create anomalous observation for the LF data in the test set
                                      % here I multup    
ModelInfo.y_L = [ModelInfo.y_L(1:200);
                ModelInfo.y_L(201:299)*10; % here I multiply by 10
                ModelInfo.y_L(300:end)];


[hyp_classic, opt_classic] = fminunc(@likelihood2Dsp, hyp_init, options);
    ModelInfo.hyp = hyp_classic;
    likelihood2Dsp(hyp_classic);
    y_classic = predict2Dsp(ModelInfo.X_L);


  %Run MF Robust

  [hyp_robust, fval, exitflag, output] = fminunc(@robustObjective2Dsp, hyp_init, options);
  ModelInfo.hyp = hyp_robust;
  y_robust = predict2Dsp(ModelInfo.X_L);

%200-300 is the test set area
        figure;
p = 3;
idx = (p-1)*100 + (1:100);

% Plot the three series
plot(idx, y_classic(idx), '-o', 'DisplayName', 'Classic'); hold on;
plot(idx, y_robust(idx), '--x', 'DisplayName', 'Robust');
plot(idx, ModelInfo.data(idx,1), 'DisplayName', 'Real Observations HF');

% Compute mean absolute errors
mae_classic = mean(abs(y_classic(idx) - ModelInfo.data(idx,1)));
mae_robust  = mean(abs(y_robust(idx)  - ModelInfo.data(idx,1)));

% Title and labels
title(sprintf('Observations station %d', p));
xlabel('Index'); ylabel('Prediction');
legend('Location','best');
grid on;

% Add text box with errors
str = sprintf('MAE Classic = %.3f\nMAE Robust = %.3f', mae_classic, mae_robust);
xpos = idx(1) + 5;        % adjust as needed
ypos = max(ModelInfo.data(idx,1)) * 0.9;  % adjust placement
text(xpos, ypos, str, 'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k');
