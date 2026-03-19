%% Bayesian optimization under uncertainty - Ex3
% Nonlinear log-log scale with homoscedastic noise
rng(56); % Set random seed for reproducibility

%% Add Model folder into path
parentDir = fileparts(pwd); % Get parent directory
targetFolder = fullfile(parentDir, 'Model'); % Define target folder

if isfolder(targetFolder) % Check if folder exists
    addpath(targetFolder); % Add to path
    disp(['Successfully added path: ', targetFolder]);
else
    warning(['The directory "', targetFolder, '" does not exist.']);
end

%% Load up the data
load("dist.mat"); % dist matrix (J x Nbeta), beta, Nrep, Nbeta, doExp, sigma_log, logm
load("DTstat_table.mat"); % DTstat, beta_range, num_Beta, doExp

J = Nrep;
x = log(beta);
logS = log(dist);  % J x Nbeta
S = dist;
sExp = doExp;

%% Generate DT table
% Create matrices for beta and repetitions
beta_mat = repmat(beta', J, 1);  % J x Nbeta
rep_mat = repmat((1:J)', 1, Nbeta);  % J x Nbeta

% Flatten matrices into column vectors
beta_col = beta_mat(:);
rep_col = rep_mat(:);
dosrom_col = S(:);
logbeta_col = log(beta_col);
logdosrom_col = log(dosrom_col);

% Create a table with all computed values
DT = table(beta_col, rep_col, dosrom_col, logbeta_col, logdosrom_col, ...
    'VariableNames', {'beta', 'rep', 'dosrom', 'logbeta', 'logdosrom'});

%% Get Local regression results
[DTbest, betaOptLocal, betaOptRange] = local_regression(DTstat);

%% Get GLM results
[optLm1, DTm1, a1] = GLM_all_data(DT, beta_range, num_Beta, doExp);

%% Bayesian GLM
% Bayesian Generalized Linear Model (GLM) with Thompson sampling,
% analytical optimization, and synchronous batch Bayesian Optimization.

% BO policy parameters
nInitial = 10; % Initial sample size for SROM
nBatch = 10;   % Batch size for new observations in each iteration
nPost = 100;   % Posterior sample size for Bayesian optimization

% Generate an initial sample on an equi-distant log scale
logbetaGrid = linspace(min(DTstat.logbeta), max(DTstat.logbeta), nInitial);
betaValues = round(exp(logbetaGrid))';

% Snap to nearest available beta in grid
betaValues = interp1(DTstat.beta, DTstat.beta, betaValues, 'nearest');

% Create DT0id table
DT0id = table('Size',[nInitial,2], 'VariableNames', {'beta', 'rep'},'VariableTypes',{'double','double'});
DT0id.beta = betaValues;
% Create the 'rep' column with sequential indices grouped by 'beta'
[groups, ~, groupIndices] = unique(DT0id.beta);
for g = 1:length(groups)
    DT0id.rep(groupIndices == g) = 1:sum(groupIndices == g);
end

% Merge DT0id with the main dataset DT based on 'beta' and 'rep' columns
DT0 = innerjoin(DT, DT0id, 'Keys', {'beta', 'rep'});

% First Bayesian Optimization (BO) policy computation
ret = BOpolicy(DT0, DTstat, 0, doExp, nPost, true);
DToptLm = ret.DToptLm;
DTopt = ret.DTopt;
DTmodel = ret.DTmodel;

% Bayesian Optimization Iterations
maxIter = 50;
boundBeta = [min(beta), max(beta)];

for BO_iter = 1:maxIter
    % Select the next set of beta values to evaluate
    betaNew = DTopt.betaOpt(DTopt.iter == BO_iter-1);
    betaNew = betaNew(1:nBatch);
    
    % Enforce bounds on beta values
    betaNew(betaNew < boundBeta(1)) = boundBeta(1);
    betaNew(betaNew > boundBeta(2)) = boundBeta(2);
    
    % Snap to nearest available beta in grid
    betaNew = interp1(DTstat.beta, DTstat.beta, betaNew, 'nearest');
    
    % Create a table for new beta values with placeholder repetitions
    DTidnew = table(sort(betaNew), Inf(size(betaNew)), true(size(betaNew)), ...
        'VariableNames', {'beta', 'rep', 'isNew'});
    
    % Create a table for existing beta values
    DT0id = table(DT0.beta, DT0.rep, false(size(DT0.beta)), 'VariableNames', {'beta', 'rep', 'isNew'});
    
    % Merge old and new beta datasets
    DTid = [DT0id; DTidnew];
    
    % Assign unique "rep" values within each beta group
    [groups, ~, groupIndices] = unique(DTid.beta);
    for g = 1:length(groups)
        DTid.rep(groupIndices == g) = 1:sum(groupIndices == g);
    end
    
    % Extract only new beta values
    DTidnew = DTid(DTid.isNew, :);
    
    % Acquire new observations from dataset
    DTnew = innerjoin(DT, DTidnew, 'Keys', {'beta', 'rep'});
    
    % Append new observations to the main dataset
    DT0 = [DT0; DTnew(:,1:5)];
    
    % Update BO policy with the new dataset
    ret = BOpolicy(DT0, DTstat, BO_iter, doExp, nPost, true);
    DToptLm = [DToptLm; ret.DToptLm];
    DTopt = [DTopt; ret.DTopt];
    DTmodel = [DTmodel; ret.DTmodel];
    
    % Stopping criteria: Convergence check using relative function improvement
    DTscore = grpstats(DTopt, 'iter', @(x) quantile(x, 0.98), 'DataVars', 'relfLm');
    if sum(DTscore{:, 3} < 1.03) >= 10
        fprintf('Converged at iteration %d\n', BO_iter);
        break;
    end
end

% Final BO summary
betaOptBO = round(DToptLm.betaOpt(end));
nDataBO = height(DT0);
nEvalMC = 13;
nEvalEfficiency = nEvalMC * Nrep / nDataBO;

fprintf('\nBO Results:\n');
fprintf('  Optimal beta (BO): %.2f\n', betaOptBO);
fprintf('  Optimal beta (local): %.2f\n', betaOptLocal);
fprintf('  Total samples used: %d\n', nDataBO);

%% Plot styling and output directory
width_plot = 8; % inches
height_plot = 6; % inches

set(groot,'DefaultTextFontSize',20);
set(groot,'DefaultAxesFontSize',20);
set(groot,'DefaultAxesFontName','Helvetica');
set(groot,'DefaultTextFontName','Helvetica');
set(groot,'DefaultAxesTickLabelInterpreter','latex');
set(groot,'DefaultLegendInterpreter','latex');
set(groot,'DefaultTextInterpreter','latex');
set(groot,'DefaultLegendFontSize',20);
set(groot,'DefaultLegendFontName','Helvetica');

th = 1.6; % line thickness

% Color palette
blue = [30, 120, 179] / 256;
blueshade = [166, 204, 227] / 256;
green = [48, 158, 43] / 256;
greenshade = [176, 222, 138] / 256;
greenshade2 = [48, 158, 43] / 256;

% Ensure output folder exists
if ~exist(fullfile(pwd,'Results'),'dir')
    mkdir(fullfile(pwd,'Results'));
end

%% plot 0: Raw data, GLM and Local regression
iterplot = BO_iter;

DTmodel_iter = DTmodel(DTmodel.iter == iterplot, :);
logbeta = DTmodel_iter.logbeta;
ypred = DTmodel_iter.ypred;
ysd = DTmodel_iter.ysd;

sub_sampled = randperm(J, 10);
logbeta_col = repmat(x', numel(sub_sampled), 1);
logbeta_col = logbeta_col(:);
logdosrom_col = reshape(logS(sub_sampled, :), [], 1);

q25Logdosrom_col = quantile(logS, 0.25, 1)';
q75Logdosrom_col = quantile(logS, 0.75, 1)';

% Create figure
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);

% Plot raw data
p1 = scatter(logbeta_col, logdosrom_col, 'k');
hold on;

% Local regression function (red curve)
p2 = plot(logbeta, DTbest.logdosromLocal, 'r', 'LineWidth', 2);

% Pointwise quantiles: 25% and 75% (cyan lines)
p3 = plot(logbeta, q25Logdosrom_col, 'c', 'LineWidth', 2);
plot(logbeta, q75Logdosrom_col, 'c', 'LineWidth', 2);

% GLM prediction (blue curve)
p5 = plot(DTm1.logbeta, DTm1.ypred, 'b', 'LineWidth', 2);

xlabel('ln$\beta$');
ylabel('ln(s($\omega$))', 'Rotation', 90);
xlim([min(x)-0.1, max(x)+0.1]);

legend([p1, p2, p3, p5], {sprintf('Data points (N = %d)', num_Beta*J), ...
    'Local regression function', ...
    'Pointwise quantiles: 25\%, 75\%', ...
    'Generalized linear regression function'}, ...
    'Location', 'southwest', 'Box', 'off');

hold off;

filename = 'glm_Ex3.pdf';
output_path = fullfile(pwd, 'Results', filename);
exportgraphics(gcf, output_path, 'ContentType', 'vector');

%% plot 1: Bayesian GLM, local regression, and GLM regression function
sz = 10;
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);

scatter(DT0.logbeta, DT0.logdosrom, sz, 'k');
hold on;

xlabel('ln$\beta$');
ylabel('ln(s($\omega$))');

qt975 = 1.96;

% Bayesian GLM regression function (blue curve)
plot(logbeta, ypred, 'color', blue, 'LineWidth', 2);

% 95% predictive intervals
P2 = fill([logbeta; flipud(logbeta)], ...
    [ypred + qt975 * ysd; flipud(ypred - qt975 * ysd)], ...
    blueshade, 'EdgeColor', 'none', 'FaceAlpha', 0.7);

% Local regression function (red curve)
plot(DTbest.logbeta, DTbest.logdosromLocal, 'r', 'LineWidth', 2);

% GLM using all data (blue curve)
plot(DTm1.logbeta, DTm1.ypred, 'b', 'LineWidth', 2);

legend(['BO data points: N = ', num2str(nDataBO)], ...
    'GLM regression function', ...
    '95\% predictive intervals', ...
    'Local regression function', ...
    'GLM regression function (all data)', ...
    'Location', 'southwest', 'Box', 'off');

hold off;

filename = 'glm_BO_Ex3.pdf';
output_path = fullfile(pwd, 'Results', filename);
exportgraphics(gcf, output_path, 'ContentType', 'vector');

%% plot 2: Posterior distribution of (beta*, f*)
mseScale = min(DTbest.mseLocal);

f = figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on;
ylimMR = [0, 8];
xlim([min(beta), max(beta)]);
ylim(ylimMR);
xlabel('$\beta$');
ylabel('$f / f^*$');

% Plot the optimal region
xVertices = [betaOptRange(1), betaOptRange(end), betaOptRange(end), betaOptRange(1)];
yVertices = [ylimMR(1), ylimMR(1), ylimMR(2), ylimMR(2)];
fill(xVertices, yVertices, [0, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

% Posterior sample of objective function (MSE)
DTparam = DTopt(DTopt.iter == iterplot, :);
Mfpred = zeros(num_Beta, nPost);
for i = 1:nPost
    Mfpred(:, i) = fGLM(beta_range, DTparam.logbeta(i), DTparam.intercept(i), DTparam.sigma2(i), doExp);
end
plt = plot(beta_range, Mfpred / mseScale, 'Color', blueshade, 'LineStyle', '-');
h1 = plt(1);

% Posterior sample optima
p5 = scatter(DTopt.betaOpt(DTopt.iter == iterplot), DTopt.fOpt(DTopt.iter == iterplot) / mseScale, ...
    'MarkerFaceColor', greenshade2, 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.5);

% Objective function from GLM mean model
currentModel = DTmodel(DTmodel.iter == iterplot, :);
p1 = plot(currentModel.beta, currentModel.fpred / mseScale, 'color', blue, 'LineWidth', 2);

% Bayesian GLM optima
p6 = plot(DToptLm.betaOpt(DToptLm.iter == iterplot), DToptLm.fOpt(DToptLm.iter == iterplot) / mseScale, ...
    'x', 'Color', blue, 'MarkerSize', sz, 'LineWidth', 2);

% Local regression result
p4 = plot(DTbest.beta, DTbest.mseLocal / mseScale, 'Color', 'r', 'LineWidth', 2);
[~, idxMin] = min(DTbest.mseLocal);
p8 = plot(DTbest.beta(idxMin), DTbest.mseLocal(idxMin) / mseScale, 'x', 'Color', 'r', 'MarkerSize', 10, 'LineWidth', 2);

% Vertical line at local regression betaOpt
xline(betaOptLocal, '--', 'LineWidth', 1);

% GLM all data objective function and optimum
p3 = plot(DTm1.beta, DTm1.fpred / mseScale, 'Color', 'b', 'LineWidth', 2);
p7 = plot(optLm1(1), optLm1(2) / mseScale, 'x', 'Color', 'b', 'MarkerSize', 10, 'LineWidth', 2);

legend([p1, h1, p3, p4, p5, p6, p7, p8], {'f() via GLM: N = ' + string(height(DT0)), ...
    'f() via Bayesian GLM: posterior sample', ...
    'f() via GLM: all data', ...
    'f() via local regression', ...
    '($\beta^*$, f$^*$) of posterior sample', ...
    '($\beta^*$, f$^*$) via GLM', ...
    '($\beta^*$, f$^*$) via GLM: all data', ...
    '($\beta^*$, f$^*$) via local regression'}, ...
    'Location', 'northeast', 'Box', 'off');
hold off;

filename = 'f_glm_TS_Ex3.pdf';
output_path = fullfile(pwd, 'Results', filename);
exportgraphics(gcf, output_path, 'ContentType', 'vector');

%% plot 3: Histogram
betaOptTS = DTopt.betaOpt(DTopt.iter == iterplot);
beta_opt_varTS = var(betaOptTS);
fprintf('Posterior beta* variance: %.6g\n', beta_opt_varTS);

figure('Color', [1 1 1], 'Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on

h1 = histogram(betaOptTS, 'BinWidth', 1);
h1.FaceColor = greenshade2;
h1.FaceAlpha = 0.5;

yl = ylim;
pOpt = fill([betaOptRange(1), betaOptRange(end), betaOptRange(end), betaOptRange(1)], ...
    [0, 0, yl(2), yl(2)], [0, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

p2 = xline(DToptLm.betaOpt(DToptLm.iter == iterplot), 'LineWidth', th);
p2.Color = blue;

p3 = xline(optLm1(1), 'b', 'LineWidth', th);
p4 = xline(DTbest.beta(idxMin), 'r', 'LineWidth', th);

ylabel('Frequency');
xlabel('$\beta^*$');

lgd = legend([h1, p2, p3, p4], {'$\beta^*$ of posterior sample', ...
    '$\beta^*$ via GLM', ...
    '$\beta^*$ via GLM: all data', ...
    '$\beta^*$ via local regression'}, ...
    'Location', 'northeast', 'Box', 'off');

hold off;

filename = 'hist_betaOpt_Ex3.pdf';
output_path = fullfile(pwd, 'Results', filename);
exportgraphics(gcf, output_path, 'ContentType', 'vector');

%% plot 4: series sample
nSamples = height(DT0);
iter = zeros(nSamples, 1);
iter(1:nInitial) = 0;
for i = 1:BO_iter
    idx_start = nInitial + (i-1)*nBatch + 1;
    idx_end = min(nInitial + i*nBatch, nSamples);
    if idx_start <= nSamples
        iter(idx_start:idx_end) = i;
    end
end

DT_plot3 = table('Size', [nSamples, 2], 'VariableTypes', {'double', 'double'}, 'VariableNames', {'Iter', 'Beta'});
DT_plot3.Iter = iter;
DT_plot3.Beta = DT0.beta;

figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on;

xlim([-1, BO_iter+1]);
ylim([min(beta)*0.9, max(beta)*1.1]);

scatter(DT_plot3.Iter, DT_plot3.Beta, 'k');

xFill = [-1, BO_iter+0.5, BO_iter+0.5, -1];
yFill = [betaOptRange(1), betaOptRange(1), betaOptRange(end), betaOptRange(end)];
fill(xFill, yFill, [0, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

legend('$\beta$ sample', 'Optimal region', 'Location', 'southeast', 'Box', 'off');

xticks(0:5:BO_iter);
xlabel('BO iteration', 'VerticalAlignment', 'top');
ylabel('$\beta$');

filename = 'series_sample_Ex3.pdf';
output_path = fullfile(pwd, 'Results', filename);
exportgraphics(gcf, output_path, 'ContentType', 'vector');

%% plot 5: series beta opt
data_plot5 = DTopt.betaOpt;
data_plot5 = reshape(data_plot5, nPost, BO_iter+1);
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);

numGroups = size(data_plot5, 2);
groupData = repelem(0:(numGroups-1), size(data_plot5, 1))';

dataVector = data_plot5(:);
h = boxchart(groupData, dataVector, 'MarkerColor', greenshade2', 'BoxFaceColor', greenshade2, 'BoxFaceAlpha', 0.5);
hold on;

p1 = plot(DToptLm.iter, DToptLm.betaOpt, '-o', 'color', blue, 'MarkerSize', 4, 'LineWidth', th);

p2 = yline(optLm1(1), 'b', 'LineWidth', 2);
p3 = yline(DTbest.beta(idxMin), 'r', 'LineWidth', 2);

xFill = [-0.5, BO_iter+1, BO_iter+1, -0.5];
yFill = [betaOptRange(1), betaOptRange(1), betaOptRange(end), betaOptRange(end)];
fill(xFill, yFill, [0, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

ax = gca;
ax.XTick = 0:5:BO_iter+1;
ax.XTickLabel = 0:5:BO_iter+1;
xlabel('BO iteration');
ylabel('$\beta^*$');

xlim([-0.5, BO_iter+1]);
ylim([min(beta)*0.9, max(beta)*1.1]);

legend([h, p1, p2, p3], {'$\beta^*$ posterior sample', ...
    '$\beta^*$ via GLM', ...
    '$\beta^*$ via GLM: all data', ...
    '$\beta^*$ via local regression'}, ...
    'Location', 'southeast', 'NumColumns', 2, 'Box', 'off');

filename = 'series_betaOpt_Ex3.pdf';
output_path = fullfile(pwd, 'Results', filename);
exportgraphics(gcf, output_path, 'ContentType', 'vector');

%% Save summary stats
bo_stats.nDataBO       = nDataBO;
bo_stats.betaOpt       = betaOptBO;
bo_stats.betaOptLocal  = betaOptLocal;
bo_stats.sigma_betaOpt = std(betaOptTS);
save(fullfile(pwd, 'Results', 'bo_stats_Ex3.mat'), '-struct', 'bo_stats');
