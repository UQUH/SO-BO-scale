function Mparam = rParamBayesLM(nBatch, df, thetahat, Vtheta, s2)
    % nBatch: Number of samples to draw
    % df: Degrees of freedom
    % thetahat: Mean of the posterior distribution (regression coefficients)
    % Vbeta: Variance-covariance matrix of the posterior distribution
    % s2: Residual variance
    
    Mparam = rNIG(nBatch, s2 * df, df, thetahat, Vtheta);
end
