%% Generate synthetic data for Ex3: Nonlinear log-log scale with homoscedastic noise
% This script generates the data and saves it to .mat files similar to Ex1
clear; clc; close all;

%% -------------------- User settings --------------------
betaMin = 2;
betaMax = 100;
Nbeta   = 99;
beta    = (betaMin:betaMax)';  % Integer values from 1 to 100

J = 1000; % replicates per beta (Nrep)
Nrep = J;

% Nonlinear mean on log-log scale
% log m(beta) = -log(log(beta))

% Homoscedastic noise (constant standard deviation on log-log scale)
sigma_log = 0.1;  % constant noise level
rng(12);

%% -------------------- Generate data (log space) --------------------
x = log(beta);
logm = -log(x);
logS = logm' + sigma_log * randn(J, Nbeta);  % J x Nbeta

% Convert to original scale
S = exp(logS);  % J x Nbeta

% dist matrix: same format as Ex1 (Nrep x Nbeta)
dist = S;  % J x Nbeta matrix of "distances"

% Define target (reference value)
sExp = 0.3;   % target value (chosen to be in the middle range)
doExp = sExp; % using doExp naming to match Ex1

fprintf('Data generated:\n');
fprintf('  Beta range: [%.2f, %.2f]\n', betaMin, betaMax);
fprintf('  Number of beta values: %d\n', Nbeta);
fprintf('  Number of MC replicates: %d\n', Nrep);
fprintf('  Target value (doExp): %.4f\n', doExp);

%% -------------------- Save dist.mat --------------------
save('dist.mat', 'dist', 'beta', 'Nrep', 'Nbeta', 'doExp', 'sigma_log', 'logm');
fprintf('\nSaved: dist.mat\n');

%% -------------------- Compute statistics per beta (DTstat) --------------------
beta_range = beta';
num_Beta = Nbeta;

DTstat = table(beta, x, logm, 'VariableNames', {'beta', 'logbeta', 'logmTrue'});
DTstat.meanS = mean(S, 1)';
DTstat.stdS = std(S, 0, 1)';
DTstat.meanLogS = mean(logS, 1)';
DTstat.stdLogS = std(logS, 0, 1)';

% MSE per beta: mean of (s - sExp)^2 across all reps
DTstat.mse = mean((S - doExp).^2, 1)';

% Local regression (LOWESS) for logs vs logbeta
frac_loess = 0.1;
DTstat.logdosromLocal = smooth(DTstat.logbeta, DTstat.meanLogS, frac_loess, 'lowess');

% Local regression for MSE
frac_loess_mse = 0.2;
DTstat.mseLocal = smooth(DTstat.logbeta, DTstat.mse, frac_loess_mse, 'lowess');

%% -------------------- Save DTstat_table.mat --------------------
save('DTstat_table.mat', 'DTstat', 'beta_range', 'num_Beta', 'doExp');
fprintf('Saved: DTstat_table.mat\n');

%% -------------------- Summary statistics --------------------
[mseLocalMin, idxOpt] = min(DTstat.mseLocal);
betaOptLocal = DTstat.beta(idxOpt);

% Find beta range within tolerance (10%)
tol_f = 0.1;
betaOptRangeMask = DTstat.mseLocal / mseLocalMin < (1 + tol_f);
betaOptRange = DTstat.beta(betaOptRangeMask);

fprintf('\nLocal regression results:\n');
fprintf('  Optimal MSE (local): %.6e\n', mseLocalMin);
fprintf('  Optimal beta (local): %.2f\n', betaOptLocal);
fprintf('  Beta range within %.0f%% tolerance: [%.2f, %.2f]\n', ...
    tol_f*100, betaOptRange(1), betaOptRange(end));

%% -------------------- Diagnostic plot --------------------
figure('Name', 'Generated Data');

subplot(1,2,1);
sub_sampled = randperm(J, 10);
logbeta_col = repmat(x', numel(sub_sampled), 1);
logbeta_col = logbeta_col(:);
logS_col = logS(sub_sampled, :);
logS_col = logS_col(:);

scatter(logbeta_col, logS_col, 8, 'k'); hold on;
plot(x, logm, 'm', 'LineWidth', 2);
plot(x, DTstat.logdosromLocal, 'r', 'LineWidth', 2);
xlabel('ln(\beta)');
ylabel('ln(s)');
legend('Samples', 'True mean', 'Local regression', 'Location', 'best');
title('Log-scale data');
grid on;

subplot(1,2,2);
plot(DTstat.beta, DTstat.mse, 'k.', 'MarkerSize', 10); hold on;
plot(DTstat.beta, DTstat.mseLocal, 'r', 'LineWidth', 2);
xline(betaOptLocal, 'r--', 'LineWidth', 1);
xlabel('\beta');
ylabel('MSE');
legend('MSE per \beta', 'Local regression', '\beta^* local', 'Location', 'best');
title('Objective function');
grid on;

fprintf('\nData generation complete.\n');
