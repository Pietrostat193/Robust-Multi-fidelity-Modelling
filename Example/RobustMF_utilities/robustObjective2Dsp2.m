function HL = robustObjective2Dsp(hyp)
    global ModelInfo

    ModelInfo.hyp = hyp;
    y_hat_H = predict2Dsp(ModelInfo.X_H);
    resid_H = ModelInfo.y_H - y_hat_H;

    % Precision for held-out points
    idx0 = numel(ModelInfo.y_L) + 1;
    prec = diag(ModelInfo.W_inv(idx0:end, idx0:end));

    r = resid_H .* sqrt(prec);
    % Compute δ via MAD
    %MAD=median(∣Z∣)=Φ −1=(0.75)≈0.6745
    scale = median(abs(r)) / 0.6745;
    delta = 1.345 * scale;

    loss_vals = huberLoss(r, delta);
    HL = sum(loss_vals);
end
