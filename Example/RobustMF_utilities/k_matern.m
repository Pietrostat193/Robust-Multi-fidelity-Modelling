function K  =k_matern(s1, s2, params)  
% Spatial kernel (Matérn 5/2)
    mode = 'iso';        % Use 'iso' for isotropic distance
    par = [];            % Set par to empty, as required by 'iso'
    d = 3;               % Matern parameter as a scalar
    % Specify the hyperparameter for the covariance function
    s_ell=params(2);
    hyp_s = s_ell;    % Spatial length scale (as required by your covariance function)
    % Compute the spatial covariance matrix with Matérn 5/2 kernel
    [K, ~] = covMatern(mode, par, d, hyp_s, s1,s2);
    s_sig=params(1);
    K= s_sig*K;
end