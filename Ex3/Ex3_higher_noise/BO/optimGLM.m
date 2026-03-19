function Mopt = optimGLM(a, lnb, sigma2, doExp)
    % optimGLM - Optimizes the GLM for given parameters.
    %
    % Parameters:
    % a      - Vector of slopes (coefficients of log(beta))
    % lnb    - Vector of intercepts (in log scale)
    % sigma2 - Vector of noise variances
    % doExp  - Reference L2 distance (e.g., from experimental data)
    %
    % Returns:
    % Mopt   - Matrix containing betaOpt and fOpt for each model

    % Number of models
    nModel = length(a);
    
    % Ensure input vectors have the same length
    assert(length(lnb) == nModel && length(sigma2) == nModel, ...
           'Vectors a, lnb, and sigma2 must have the same length');
    
    % Compute the optimal beta
    betaOpt = (doExp ./ exp(1.5 * sigma2 + lnb)).^(1 ./ a);
    
    % Ensure betaOpt is a column vector
    if isrow(betaOpt)
        betaOpt = betaOpt';
    end
    
    % Compute fOpt for each model using an arrayfun loop for vectorized operation
    fOpt = arrayfun(@(i) fGLM(betaOpt(i), a(i), lnb(i), sigma2(i), doExp), 1:nModel)';
    
    % Combine results into a matrix (both as column vectors)
    Mopt = [betaOpt, fOpt]; % betaOpt and fOpt should now have consistent dimensions
end
