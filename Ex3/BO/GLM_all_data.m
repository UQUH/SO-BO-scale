function [optLm1,DTm1,a1] = GLM_all_data(DT,beta_range,num_Beta,doExp)
% Classical estimate using all data
X = [ones(height(DT), 1), DT.logbeta]; % Design matrix with intercept
y = DT.logdosrom; % Response variable

% Perform linear regression
lm1 = lmFun(X, y);

% Extract values from the result
df1 = lm1.df;
thetahat1 = lm1.thetahat;
Vtheta1 = lm1.Vtheta;
s21 = lm1.s2;

% Step 2: Prepare DTm1 table for predictions
DTm1 = table('Size',[num_Beta,2],'VariableTypes',...
    {'double','double'},'VariableNames',{'logbeta','beta'});
DTm1.beta = beta_range(:);
DTm1.logbeta = log(beta_range(:));
% Create new design matrix for prediction
Xnew = [ones(height(DTm1), 1), DTm1.logbeta];
% Generate predictive distributions using Bayesian linear model
Mpred = predBayesLM(Xnew, df1, thetahat1, Vtheta1, s21);
% Add predictive values to DTm1
DTm1.ypred = Mpred(:, 1); % Predicted mean values
DTm1.ysd = Mpred(:, 2);   % Predicted standard deviations
% Step 3: Objective function and its optimum from classical estimation
a1 = thetahat1(2); % Slope (corresponding to 'logbeta')
lnb1 = thetahat1(1); % Intercept
% Compute predicted values of the objective function
DTm1.fpred = arrayfun(@(b) fGLM(b, a1, lnb1, s21, doExp), DTm1.beta);
% Compute the optimum using classical estimates
optLm1 = optimGLM(a1, lnb1, s21, doExp);
