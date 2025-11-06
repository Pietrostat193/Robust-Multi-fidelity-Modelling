function [mean_pred, var_star] = predict2Dsp2(x_star)

global ModelInfo

X_L = ModelInfo.X_L;
X_H = ModelInfo.X_H;
y_L = ModelInfo.y_L;
y_H = ModelInfo.y_H;
hyp = ModelInfo.hyp;

rho       = hyp(5);
cov_type  = ModelInfo.cov_type;
combination = ModelInfo.combination;
L = ModelInfo.L;                    % Cholesky of training K
y = [y_L; y_H];

% === Cross-covariances as in your original code ===
switch cov_type
    case 'RBF'
        psi1_t = rho * k1(x_star(:,1),   X_L(:,1),    exp(hyp(1:2)));
        psi1_s = rho * k1(x_star(:,2:3), X_L(:,2:3),  exp(hyp(8:9)));

        k_t1 = k1(x_star(:,1),   X_H(:,1),    exp(hyp(1:2)));
        k_s1 = k1(x_star(:,2:3), X_H(:,2:3),  exp(hyp(8:9)));
        k_t2 = k1(x_star(:,1),   X_H(:,1),    exp(hyp(3:4)));
        k_s2 = k1(x_star(:,2:3), X_H(:,2:3),  exp(hyp(10:11)));
        psi2 = rho^4 * (k_t1 .* k_s1) + (k_t2 .* k_s2);

    case 'RBF_separate_rho'
        rho_s  = hyp(12);
        psi1_t = rho   * k1(x_star(:,1),   X_L(:,1),    exp(hyp(1:2)));
        psi1_s = rho_s * k1(x_star(:,2:3), X_L(:,2:3),  exp(hyp(8:9)));

        psi2_t = rho^2   * k1(x_star(:,1),   X_H(:,1),   exp(hyp(1:2))) + k1(x_star(:,1),   X_H(:,1),   exp(hyp(3:4)));
        psi2_s = rho_s^2 * k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(8:9))) + k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(10:11)));

    case 'Matern'
        psi1_t = rho * k_matern(x_star(:,1),   X_L(:,1),    exp(hyp(1:2)));
        psi1_s = rho * k_matern(x_star(:,2:3), X_L(:,2:3),  exp(hyp(8:9)));

        psi2_t = rho^2 * k_matern(x_star(:,1),   X_H(:,1),   exp(hyp(1:2))) + k_matern(x_star(:,1),   X_H(:,1),   exp(hyp(3:4)));
        psi2_s = rho^2 * k_matern(x_star(:,2:3), X_H(:,2:3), exp(hyp(8:9))) + k_matern(x_star(:,2:3), X_H(:,2:3), exp(hyp(10:11)));

    case 'Mix'
        psi1_t = rho * k1(x_star(:,1),   X_L(:,1),    exp(hyp(1:2)));
        psi1_s = rho * k_matern(x_star(:,2:3), X_L(:,2:3),  exp(hyp(8:9)));

        psi2_t = rho^2 * k1(x_star(:,1),   X_H(:,1),   exp(hyp(1:2))) + k1(x_star(:,1),   X_H(:,1),   exp(hyp(3:4)));
        psi2_s = rho^2 * k_matern(x_star(:,2:3), X_H(:,2:3), exp(hyp(8:9))) + k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(10:11)));

    otherwise
        error('Invalid covariance type.');
end

switch combination
    case 'additive'
        psi1 = psi1_t + psi1_s;
        psi2 = psi2_t + psi2_s;
    case 'multiplicative'
        psi1 = psi1_t .* psi1_s;
        if ~exist('psi2','var') && exist('psi2_t','var') && exist('psi2_s','var')
            psi2 = psi2_t .* psi2_s;
        end
    otherwise
        error('invalid combination')
end

q = [psi1 psi2];

% === Posterior mean ===
m = 0;
mean_pred = q*(L'\(L\y)) + m;

% === Posterior variance (diag) consistent with cov_type/combination ===
% Build prior self-cov at x_star for the HF output (Kxx diagonal)
switch cov_type
    case 'RBF'
        kL_t = k1(x_star(:,1),   x_star(:,1),   exp(hyp(1:2)));
        kL_s = k1(x_star(:,2:3), x_star(:,2:3), exp(hyp(8:9)));
        kH_t = k1(x_star(:,1),   x_star(:,1),   exp(hyp(3:4)));
        kH_s = k1(x_star(:,2:3), x_star(:,2:3), exp(hyp(10:11)));
        switch combination
            case 'multiplicative'
                Kxx_diag = diag( rho^4 * (kL_t .* kL_s) + (kH_t .* kH_s) );
            case 'additive'
                Kxx_diag = diag( (rho^2*kL_t + kH_t) + (rho^2*kL_s + kH_s) );
        end

    case 'RBF_separate_rho'
        rho_s = hyp(12);
        kL_t = k1(x_star(:,1),   x_star(:,1),   exp(hyp(1:2)));
        kL_s = k1(x_star(:,2:3), x_star(:,2:3), exp(hyp(8:9)));
        kH_t = k1(x_star(:,1),   x_star(:,1),   exp(hyp(3:4)));
        kH_s = k1(x_star(:,2:3), x_star(:,2:3), exp(hyp(10:11)));
        switch combination
            case 'multiplicative'
                Kxx_diag = diag( (rho^2*kL_t) .* (rho_s^2*kL_s) + (kH_t .* kH_s) );
            case 'additive'
                Kxx_diag = diag( (rho^2*kL_t + kH_t) + (rho_s^2*kL_s + kH_s) );
        end

    case 'Matern'
        kL_t = k_matern(x_star(:,1),   x_star(:,1),   exp(hyp(1:2)));
        kL_s = k_matern(x_star(:,2:3), x_star(:,2:3), exp(hyp(8:9)));
        kH_t = k_matern(x_star(:,1),   x_star(:,1),   exp(hyp(3:4)));
        kH_s = k_matern(x_star(:,2:3), x_star(:,2:3), exp(hyp(10:11)));
        switch combination
            case 'multiplicative'
                Kxx_diag = diag( rho^2*(kL_t .* kL_s) + (kH_t .* kH_s) );
            case 'additive'
                Kxx_diag = diag( (rho^2*kL_t + kH_t) + (rho^2*kL_s + kH_s) );
        end

    case 'Mix'
        kL_t = k1(x_star(:,1),   x_star(:,1),   exp(hyp(1:2)));
        kL_s = k_matern(x_star(:,2:3), x_star(:,2:3), exp(hyp(8:9)));
        kH_t = k1(x_star(:,1),   x_star(:,1),   exp(hyp(3:4)));
        kH_s = k_matern(x_star(:,2:3), x_star(:,2:3), exp(hyp(10:11)));
        switch combination
            case 'multiplicative'
                Kxx_diag = diag( rho^2*(kL_t .* kL_s) + (kH_t .* kH_s) );
            case 'additive'
                Kxx_diag = diag( (rho^2*kL_t + kH_t) + (rho^2*kL_s + kH_s) );
        end
end

% Reduction term: diag(q * K^{-1} * q') via Cholesky
v         = L \ q';
quad_term = sum(v.^2, 1)';

% Jitter floor and non-negativity clamp
vjitter = 1e-12;
if isfield(ModelInfo,'jitter') && ~isempty(ModelInfo.jitter)
    vjitter = max(1e-12, ModelInfo.jitter);
end
var_star = max(Kxx_diag - quad_term, 0) + vjitter;

% ---- (optional) quick summaries; comment out if not needed ----
% fprintf('mean_pred  -> mean=%.3f, std=%.3f, min=%.3f, max=%.3f\n', ...
%     mean(mean_pred,'omitnan'), std(mean_pred,'omitnan'), ...
%     min(mean_pred,[],'omitnan'), max(mean_pred,[],'omitnan'));
% fprintf('var_star   -> mean=%.3f, std=%.3f, min=%.3f, max=%.3f\n', ...
%     mean(var_star,'omitnan'), std(var_star,'omitnan'), ...
%     min(var_star,[],'omitnan'), max(var_star,[],'omitnan'));
end
