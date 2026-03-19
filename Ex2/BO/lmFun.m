function result = lmFun(X, y)
    % lmFun - Performs OLS regression and returns relevant statistics.
    %
    % Parameters:
    % X - Input matrix (n x k), where n is the number of observations and k is the number of variables.
    % y - Response vector (n x 1).
    %
    % Returns:
    % result - A struct containing the degrees of freedom (df), coefficient estimates (betahat),
    %          variance-covariance matrix (Vbeta), and residual variance (s2).

    % Get the dimensions
    [n, k] = size(X);
    df = n - k;  % Degrees of freedom

    % Compute cross-product matrices
    XX = X' * X;       % Equivalent to crossprod(X) in R
    Xy = X' * y;       % Equivalent to crossprod(X, y) in R

    % Solve for betahat (coefficient estimates)
    thetahat = XX \ Xy; % Solves for betahat using the inverse or a more efficient solver

    % Compute fitted values and residuals
    fitted = X * thetahat; % Fitted values
    resid = y - fitted;   % Residuals

    % Compute residual variance (s2)
    s2 = sum(resid .^ 2) / df; % Residual variance

    % Compute the variance-covariance matrix for betahat (Vbeta)
    Vtheta = inv(XX); % Variance-covariance matrix

    % Return results in a struct
    result.df = df;          % Degrees of freedom
    result.thetahat = thetahat; % Coefficient estimates
    result.Vtheta = Vtheta;    % Variance-covariance matrix of betahat
    result.s2 = s2;          % Residual variance
end
