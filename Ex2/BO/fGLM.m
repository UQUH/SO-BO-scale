function f = fGLM(beta, a, lnb, sigma2, d0)
    % fGLM - Computes the mean squared error (MSE) of the SROM, estimated from a generalized linear model.
    %
    % Parameters:
    % beta  - Hyper-parameter, the effective sample size
    % a     - Slope, the coefficient of log(beta)
    % lnb   - Intercept (log of the parameter b)
    % sigma2 - Noise variance
    % d0    - Reference L2 distance (e.g., from experimental observation to reference ROM prediction)
    %
    % Returns:
    % f - Mean squared error (MSE) estimate

    % Convert intercept from log scale
    b = exp(lnb);
    
    % Calculate constants C2 and C1
    C2 = b^2 * (exp(2 * sigma2) - exp(sigma2));
    C1 = b * exp(sigma2 / 2);
    
    % Compute the transformed beta
    xa = beta.^a;
    
    % Calculate the mean squared error
    f = C2 * xa.^2 + (C1 * xa - d0).^2;
end

