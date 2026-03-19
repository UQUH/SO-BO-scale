function f = fMC(beta,DTstat,store)
% Persistent variable to store evaluated beta values across function calls
persistent betaEval;
if isempty(betaEval)
    betaEval = [];
end
% Set default value for store if not provided
if nargin < 5
    store = true;
end
% If store is true, append the new beta value to betaEval
if store
    betaEval = [betaEval; beta];
end
assignin('base','store_beta_values',betaEval)

% Find the integer part (floor) of beta
beta0 = floor(beta);
w = beta - beta0;  % Calculate the weight based on fractional part

% Find the rows in DTstat for beta0 and beta0 + 1
f0 = DTstat.mse(DTstat.beta == beta0);        % MSE for beta0
f1 = DTstat.mse(DTstat.beta == beta0 + 1);    % MSE for beta0 + 1

% Perform linear interpolation
f = f0 * (1 - w) + f1 * w;
end