function Mparam = rNIG(n, a, df, m, V)
    % Generate noise variance from inverse chi-square distribution
    sigma2 = a ./ chi2rnd(df, n, 1);
    
    % Spectral sampling from the conditional distribution of coefficients
    R = chol(V, 'lower');% Cholesky factor of covariance matrix V
    
    k = length(m);% Number of coefficients
    
    Z = randn(k, n); %random kxn matrix
    
    Sigma = diag(sqrt(sigma2));% Diagonal matrix of sqrt(sigma2)
    
    theta = m + R * Z * Sigma;% Calculate coefficients (theta)
    
    % Combine beta and sigma2 into one matrix
    Mparam = [theta; sigma2']';
end
