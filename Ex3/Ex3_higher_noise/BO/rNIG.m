function Mparam = rNIG(n, a, d, m, V)
    % Generate noise variance from inverse chi-square distribution
    sigma2 = a ./ chi2rnd(d, n, 1);
    
    % Cholesky factor of covariance matrix V
    R = chol(V, 'lower');
    
    % Number of coefficients
    k = length(m);
    
    % Generate random normal variables
    Z = randn(k, n);
    
    % Diagonal matrix of sqrt(sigma2)
    Sigma = diag(sqrt(sigma2));
    
    % Calculate coefficients (beta)
    theta = m + R * Z * Sigma;
    
    % Combine beta and sigma2 into one matrix
    Mparam = [theta; sigma2']';
end
