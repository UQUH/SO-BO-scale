function result = lmFun(X, y)
    % lmFun - Performs regression and returns relevant statistics.
    %
    % Parameters:
    % X - Input matrix consists of the cofeecients of a and ln(b).
    % y - Output log of the distance between ROM and SROM.
    %
    % Returns:
    % result - A struct containing the degrees of freedom (df), coefficient estimates (betahat),
    %          variance-covariance matrix (Vbeta), and residual variance (s2).

    [n, k] = size(X);     % Get the dimensions
    df = n - k;  % Degrees of freedom

    % Compute cross-product matrices
    XX = X' * X;       
    Xy = X' * y;       

    % Solve for coefficients a and lnb
    theta = XX \ Xy; 

    % Compute the L2 distance using the surrogate
    fitted = X * theta; % Fitted values
    resid = y - fitted;   % Residuals

    % Compute residual variance (s2)
    s2 = sum(resid .^ 2) / df; % Residual variance

    % Compute the variance-covariance matrix for betahat (Vbeta)
    Vtheta = inv(XX); % Variance-covariance matrix

    % Return results in a struct
    result.df = df;          % Degrees of freedom
    result.thetahat = theta; % Coefficient estimates
    result.Vtheta = Vtheta;    % Variance-covariance matrix of betahat
    result.s2 = s2;          % Residual variance
end
