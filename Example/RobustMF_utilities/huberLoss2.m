function L = huberLoss2(r, delta)
    % Minimal tweak: inflate & floor delta to avoid over-robustness
    c = 2.0;                 % try 1.5–3.0 if still too flat
    delta_min = 1e-3;        % small floor (in residual units)
    delta_eff = max(c*delta, delta_min);

    a = abs(r);
    L = zeros(size(r));
    idx = (a <= delta_eff);

    % Quadratic region
    L(idx)  = 0.5 * r(idx).^2;

    % Linear region
    L(~idx) = delta_eff .* (a(~idx) - 0.5*delta_eff);
end
