function results = BOpolicy(DT0,DTstat, iter, doExp, nPost, model)
    % BOpolicy - Performs Bayesian GLM estimation and optimization.
    %
    % Parameters:
    % DT0   - Initial data table (assumed to be a MATLAB table).
    % iter  - Iteration number.
    % doExp - Reference L2 distance (experimental).
    % nPost - Number of posterior samples.
    % model - Boolean indicating if predictive distributions should be calculated.
    %
    % Returns:
    % results - A struct containing DToptLm, DTopt, and optionally DTmodel.

    if nargin < 6
        model = true; % Default value if not provided
    end

    k = iter;

    % Bayesian GLM estimation
    X0 = [ones(height(DT0), 1), DT0.logbeta]; % Design matrix with intercept and logbeta
    y0 = DT0.logdosrom; % Response variable
    ret = lmFun(X0, y0); % Call to lmFun for regression analysis

    % Unpack returned struct
    df = ret.df;
    thetahat = ret.thetahat;
    Vtheta = ret.Vtheta;
    s2 = ret.s2;

    a0 = thetahat(2);  % Slope
    lnb0 = thetahat(1); % Intercept

    % Optimum of the objective function from the classical estimate
    optLm = optimGLM(a0, lnb0, s2, doExp);
    DToptLm = table(repmat(k, size(optLm, 1), 1), optLm(:, 1), optLm(:, 2), ...
                    'VariableNames', {'iter', 'betaOpt', 'fOpt'});

    % Posterior samples
    Mparam = rParamBayesLM(nPost, df, thetahat, Vtheta, s2);

    % Ensure Mparam has nPost rows
    if size(Mparam, 1) ~= nPost
        error('The number of rows in Mparam must match nPost.');
    end

    a = Mparam(:, 2);      % Slope (logbeta)
    lnb = Mparam(:, 1);    % Intercept
    sigma2 = Mparam(:, 3); % Noise variance

    % Posterior sample of optimal hyperparameter and optimal objective value
    Mopt = optimGLM(a, lnb, sigma2, doExp);

    % Ensure Mopt has nPost rows
    if size(Mopt, 1) ~= nPost
        error('The number of rows in Mopt must match nPost.');
    end

    % Create DTopt table with consistent row counts
    DTopt = table(repmat(k, nPost, 1), Mparam(:, 1), Mparam(:, 2), Mparam(:, 3), ...
                  Mopt(:, 1), Mopt(:, 2), ...
                  'VariableNames', {'iter', 'intercept', 'logbeta', 'sigma2', 'betaOpt', 'fOpt'});

    % Compute fLm at the optimal beta
    fLmBetaOpt = arrayfun(@(i) fGLM(Mopt(i, 1), a0, lnb0, s2, doExp), 1:nPost)';
    DTopt.fLm = fLmBetaOpt;
    DTopt.relfLm = fLmBetaOpt ./ DToptLm.fOpt(1); % Ensure matching size

    % Calculate predictive distributions if requested
    if model
        DTmodel = table(DTstat.logbeta, DTstat.beta, 'VariableNames', {'logbeta', 'beta'});
        DTmodel.iter = repmat(k, height(DTmodel), 1);

        Xnew = [ones(height(DTmodel), 1), DTmodel.logbeta];
        Mpred = predBayesLM(Xnew, df, thetahat, Vtheta, s2);
        DTmodel.ypred = Mpred(:, 1);
        DTmodel.ysd = Mpred(:, 2);

        % Objective function values from the classical estimate
        DTmodel.fpred = arrayfun(@(i) fGLM(DTmodel.beta(i), a0, lnb0, s2, doExp), 1:height(DTmodel))';
        results = struct('DToptLm', DToptLm, 'DTopt', DTopt, 'DTmodel', DTmodel);
    else
        results = struct('DToptLm', DToptLm, 'DTopt', DTopt);
    end
end