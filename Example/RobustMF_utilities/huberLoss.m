function L = huberLoss(r, delta)
    % HUBERLOSS Compute Huber loss element‑wise
    %
    %   L = huberLoss(r, delta)
    %
    % Inputs:
    %   r     – residuals (predicted minus actual), scalar/vector/matrix
    %   delta – threshold where loss transitions from quadratic to linear
    %
    % Output:
    %   L     – same size as r, containing Huber loss values

    L = zeros(size(r));
    idx = abs(r) <= delta;

    % Quadratic region
    L(idx) = 0.5 * r(idx).^2;

    % Linear region
    L(~idx) = delta * (abs(r(~idx)) - 0.5 * delta);
end
