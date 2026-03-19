function [DTbest,betaOptLocal,betaOptRange] = local_regression(DTstat)

%% Local regression (smoothing)
% Check which column name exists for average log values
if ismember('avgLogdosrom', DTstat.Properties.VariableNames)
    avgLogCol = DTstat.avgLogdosrom;
elseif ismember('meanLogS', DTstat.Properties.VariableNames)
    avgLogCol = DTstat.meanLogS;
elseif ismember('logdosromLocal', DTstat.Properties.VariableNames)
    avgLogCol = DTstat.logdosromLocal;
else
    error('DTstat must contain avgLogdosrom, meanLogS, or logdosromLocal column');
end

fitLocal = smooth(DTstat.logbeta, avgLogCol, 0.1, 'loess');

% Create a new table for best beta values
DTbest = table(DTstat.logbeta, DTstat.beta, 'VariableNames', {'logbeta', 'beta'});

% Predict logdosrom values using the fitted model
DTbest.logdosromLocal = fitLocal;

% Perform local polynomial regression for MSE vs. beta
fitLocalMse = smooth(DTstat.logbeta, DTstat.mse, 0.2, 'loess');

% Predict smoothed MSE values
mseLocal = fitLocalMse;

% Store smoothed MSE values in DTbest
DTbest.mseLocal = mseLocal;

% Find the minimum value of the smoothed MSE
mseLocalMin = min(DTbest.mseLocal);

% Identify the optimal beta corresponding to the minimum MSE
[~, idxMin] = min(DTbest.mseLocal);
betaOptLocal = DTbest.beta(idxMin); % Optimal beta value

% Set relative tolerance for optimal MSE
tolF = 0.1; % 10% tolerance

% Find range of beta values where MSE is within the tolerance
betaOptRange = DTbest.beta(DTbest.mseLocal / mseLocalMin < (1 + tolF));

% Compute min and max beta values within the tolerance range
betaOptRangeMin = min(betaOptRange);
betaOptRangeMax = max(betaOptRange);

% Display results
disp(['Optimal mseLocal: ', num2str(mseLocalMin)]);
disp(['Optimal beta (local smoothing): ', num2str(betaOptLocal)]);
disp(['Beta range within tolerance: [', num2str(betaOptRangeMin), ', ', num2str(betaOptRangeMax), ']']);