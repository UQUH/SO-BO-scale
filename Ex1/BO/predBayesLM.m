function Mpred = predBayesLM(Xnew, df, thetahat, Vtheta, s2, withNoise)
    % Predictive distribution for Bayesian linear regression
    if nargin < 6
        withNoise = false;
    end
    
    % Predicted values
    ypred = Xnew * thetahat;
    
    % Variance-covariance matrix
    Vcov = s2 * Vtheta;
    
    % Predictive variance
    yvar = sum((Xnew * Vcov) .* Xnew, 2);
    
    if withNoise
        yvar = yvar + s2;
    end
    
    % Predictive standard deviations
    ysd = sqrt(yvar);
    
    % Return mean and standard deviation
    Mpred = [ypred, ysd];
end
