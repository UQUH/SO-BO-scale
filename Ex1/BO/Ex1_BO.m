%% Bayesian optimization under uncertainty Linear static problem
rng(56); % Set random seed for reproducibility

%% Add Model folder into path
parentDir = fileparts(pwd); % Get parent directory
targetFolder = fullfile(parentDir, 'Model'); % Define target folder

if isfolder(targetFolder) % Check if folder exists
    addpath(targetFolder); % Add to path
    disp(['Successfully added path: ', targetFolder]); % Success message
else
    warning(['The directory "', targetFolder, '" does not exist.']); % Warning message
end

%% Load up the data
load("dist.mat"); % data Matrix of L2 distances (1000 repetitions x 143 beta values)

k = 8; % ROM (Reduced Order Model) dimension
doExp = 3.456255510150231e-04; % reference distance

%% Generate DT table
[Nrep,num_Beta] = size(dist);
%Nrep = Number ofMonte carlo interations
%num_Beta = Total number of beta values

boundBeta = [k, num_Beta+k-1];  % Define the search space for beta values
beta_range = boundBeta(1):1:boundBeta(2); % Range of beta values
rep_range = 1:1:Nrep; % Range of repetitions

% % Create matrices for beta and repetitions
beta_mat = repmat(beta_range, Nrep, 1);
rep_mat = repmat(rep_range, num_Beta, 1);

% % Flatten matrices into column vectors
beta_col = reshape(beta_mat, Nrep * num_Beta, 1);
rep_col = reshape(rep_mat', Nrep * num_Beta, 1);
dosrom_col = dist(:); % Flatten dist into a column vector

% Compute log-transformed values
logbeta_col = log(beta_col);
logdosrom_col = log(dosrom_col);

% Create a table with all computed values
DT = table(beta_col, rep_col, dosrom_col, logbeta_col, logdosrom_col, ...
           'VariableNames', {'beta', 'rep', 'dosrom', 'logbeta', 'logdosrom'});

%% Get Local regression results
load('DTstat_table.mat',"DTstat"); %load DTstat table
[DTbest, betaOptLocal, betaOptRange] = local_regression(DTstat); % Run local regression

%% Get GLM results
[optLm1, DTm1,a1] = GLM_all_data(DT,beta_range, num_Beta, doExp); % Run GLM with all data

%% Bayesian GLM
% Bayesian Generalized Linear Model (GLM) with Thompson sampling,
% analytical optimization, and synchronous batch Bayesian Optimization.

% BO policy parameters
nInitial = 40; % Initial sample size for SROM
nBatch = 10;   % Batch size for new observations in each iteration
nPost = 100;   % Posterior sample size for Bayesian optimization

% Generate an initial sample on an equi-distant log scale
logbetaGrid = linspace(min(DTstat.logbeta), max(DTstat.logbeta), nInitial);
betaValues = round(exp(logbetaGrid))'; % Convert log scale back to beta values

% Create DT0id table
DT0id = table('Size',[nInitial,2], 'VariableNames', {'beta', 'rep'},'VariableTypes',{'double','double'});
DT0id.beta = betaValues;
% Create the 'rep' column with sequential indices grouped by 'beta'
[groups, ~, groupIndices] = unique(DT0id.beta); % Group 'beta' values
for g = 1:length(groups)
    % Assign sequential indices within each group
    DT0id.rep(groupIndices == g) = 1:sum(groupIndices == g);
end
% Create DT0 table
DT0 = table('Size',[nInitial,5], 'VariableNames', {'beta', 'rep','dosrom', 'logbeta', 'logdosrom'},'VariableTypes',{'double','double','double','double','double'});
DT0.beta = DT0id.beta;
DT0.rep = DT0id.rep;
DT0.logbeta = log(DT0.beta);

% Merge DT0id with the main dataset DT based on 'beta' and 'rep' columns
DT0 = innerjoin(DT, DT0id, 'Keys', {'beta', 'rep'});

% First Bayesian Optimization (BO) policy computation
ret = BOpolicy(DT0, DTstat, 0, doExp, nPost, true); 
DToptLm = ret.DToptLm; % Store optimal beta values from local model
DTopt = ret.DTopt; % Store overall optimal values
DTmodel = ret.DTmodel; % Store BO model information

% Bayesian Optimization Iterations
maxIter = 50; % Maximum number of BO iterations

for BO_iter = 1:maxIter
    % Select the next set of beta values to evaluate
    betaNew = DTopt.betaOpt(DTopt.iter == BO_iter-1);
    betaNew = betaNew(1:nBatch); % Take only nBatch new points

    % Enforce bounds on beta values
    betaNew(betaNew < boundBeta(1)) = boundBeta(1);
    betaNew(betaNew > boundBeta(2)) = boundBeta(2);

    % Create a table for new beta values with placeholder repetitions
    DTidnew = table(sort(round(betaNew)), Inf(size(betaNew)), true(size(betaNew)), ...
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
    DToptLm = [DToptLm; ret.DToptLm]; % Update local optima tracking
    DTopt = [DTopt; ret.DTopt]; % Update overall optimal values
    DTmodel = [DTmodel; ret.DTmodel]; % Update BO model state

    % Stopping criteria: Convergence check using relative function improvement
    DTscore = grpstats(DTopt, 'iter', @(x) quantile(x, 0.98), 'DataVars', 'relfLm');
    if sum(DTscore{:, 3} < 1.03) >= 3
        break; % Stop if the 98th percentile improvement is below threshold for 3 iterations
    end
end

% Final BO summary
betaOptBO = round(DToptLm.betaOpt(end)); % final optimal beta
nDataBO = height(DT0); % total BO samples
nEvalMC = 13; % MC iterations 
nRep = Nrep; % MC samples per iterations
nEvalEfficiency = nEvalMC * nRep / nDataBO; % baseline/BO efficiency ratio

%% Plot styling and output directory
width_plot = 8; % inches
height_plot = 6; % inches

% Global default styling (affects session). Consider scoping via set on axes/fig instead.
set(groot,'DefaultTextFontSize',20);
set(groot,'DefaultAxesFontSize',20);
set(groot,'DefaultAxesFontName','Helvetica');
set(groot,'DefaultTextFontName','Helvetica');
set(groot,'DefaultAxesTickLabelInterpreter','latex');
set(groot,'DefaultLegendInterpreter','latex');
set(groot,'DefaultTextInterpreter','latex');
set(groot,'DefaultLegendFontSize',20);
set(groot,'DefaultLegendFontName','Helvetica');

th = 1.6; % line thickness for emphasis

% Color palette
blue = [30, 120, 179] / 256; % GLM curves
blueshade = [166, 204, 227] / 256; % predictive band
green = [48, 158, 43] / 256; % unused base green
greenshade = [176, 222, 138] / 256; % not used below
greenshade2 = [48, 158, 43] / 256; % same hue as `green`; transparency is controlled via FaceAlpha

% Ensure output folder exists
if ~exist(fullfile(pwd,'Results'),'dir')
mkdir(fullfile(pwd,'Results'));
end

%% plot 0: Raw data, GLM and Local regression
load('dist.mat', 'dist'); % Load distance data between experimental and ROM from file
Nrep = 1000; % Number of repetitions for sampling
beta_mat = repmat(beta_range, Nrep, 1); % Replicate beta_range for sampling
iterplot = BO_iter; % Set current iteration for plotting

% Extract data for the current iteration
DTmodel_iter = DTmodel(DTmodel.iter == iterplot, :); % Filter data for the iteration
logbeta = DTmodel_iter.logbeta; % Logarithmic beta values
ypred = DTmodel_iter.ypred; % Predicted values from model
ysd = DTmodel_iter.ysd; % Standard deviation of predictions

sub_sampled = randperm(1000, 10); % Randomly sample 10 indices from 1000
logbeta_col = reshape(log(beta_mat(sub_sampled, :)), 10*num_Beta, 1); % Reshape log beta values
logdosrom_col = reshape(log(dist(sub_sampled, :)), 10*num_Beta, 1); % Reshape log distance values

q25Logdosrom_col = quantile(log(dist), 0.25); % 25th quantile
q75Logdosrom_col = quantile(log(dist), 0.75); % 75th quantile

% Create figure with specific size
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);

% Plot raw data
p1 = scatter(logbeta_col, logdosrom_col, 'k'); % Scatter plot for data points
hold on;

% Add local regression function (red curve)
p2 = plot(logbeta, DTbest.logdosromLocal, 'r', 'LineWidth', 2); 

% Add pointwise quantiles: 25% and 75% (cyan lines)
p3 = plot(logbeta, q25Logdosrom_col, 'c', 'LineWidth', 2); 
plot(logbeta, q75Logdosrom_col, 'c', 'LineWidth', 2); 

% Add GLM prediction (blue curve)
p5 = plot(DTm1.logbeta, DTm1.ypred, 'b', 'LineWidth', 2); 

xlabel('ln$\beta$'); % x-axis label
ylabel('ln(d$_o$(u$_L$))', 'Rotation', 90); % y-axis label (rotated)
xlim([2 5.1]); % x-axis limits
ylim([-9.2 -6.2]); % y-axis limits

legend([p1, p2, p3, p5], {sprintf('Data points (N = %d)', num_Beta*1000), ...
    'Local regression function', ...
    'Pointwise quantiles: 25\%, 75\%', ...
    'Generalized linear regression function'}, ...
    'Location', 'southwest', 'Box', 'off'); 

hold off; % Release hold on the current figure

filename = 'glm_Ex1.pdf';
output_path = fullfile(pwd, 'Results', filename);% Combine into a full relative path
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure

%% plot 1: Bayesian GLM, local regression, and GLM regression function
sz = 10; % Marker size for scatter plot
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]); % Create figure with specified size

% Scatter plot of BO data points
scatter(DT0.logbeta, DT0.logdosrom, sz, 'k'); 
hold on;

% Set axis labels and limits
set(gca, 'XTick', (0:1:5)); % x-axis ticks
xlim([2 5.1]); % Set x-axis limits
ylim([-9.2 -6.2]); % Set y-axis limits
xlabel('ln$\beta$'); % x-axis label
ylabel('ln(d$_{o}$(u$_{L}$))'); % y-axis label

qt975 = 1.96; % 95% quantile for normal distribution (for predictive intervals)

% Plot the Bayesian GLM regression function (blue curve)
plot(logbeta, ypred, 'color', blue, 'LineWidth', 2);

% Plot 95% predictive intervals (shaded area around Bayesian GLM regression)
P2 = fill([logbeta; flipud(logbeta)], ...
    [ypred + qt975 * ysd; flipud(ypred - qt975 * ysd)], ...
    blueshade, 'EdgeColor', 'none', 'FaceAlpha', 0.7);

% Plot the local regression function (red curve)
plot(DTbest.logbeta, DTbest.logdosromLocal, 'r', 'LineWidth', 2);

% Plot the generalized linear model using all data (blue curve)
plot(DTm1.logbeta, DTm1.ypred, 'b', 'LineWidth', 2);

% Add a legend with descriptions
legend(['BO data points: N = ', num2str(nDataBO)], ...
    'GLM regression function', ...
    '95\% predictive intervals', ...
    'Local regression function', ...
    'GLM regression function (all data)', ...
    'Location', 'southwest', 'Box', 'off');

hold off; % Release the hold on the current plot

filename = 'glm_BO_Ex1.pdf';
output_path = fullfile(pwd, 'Results', filename);% Combine into a full relative path
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure

%% plot 2: Posterior distribution of $(\beta^*, f^*)$
f = figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]); % Create figure
hold on;
ylimMR = [0, 8]; % y-axis limits for the plot
xlim([22, boundBeta(2)]); % Set x-axis limits
ylim(ylimMR); % Set y-axis limits
xlabel('$\beta$'); % x-axis label
ylabel('$f / f^*$'); % y-axis label

mseScale = min(DTbest.mseLocal); % Scale factor

% Plot the optimal region (rectangle)
xVertices = [betaOptRange(1), betaOptRange(end), betaOptRange(end), betaOptRange(1)]; % Define x vertices
yVertices = [ylimMR(1), ylimMR(1), ylimMR(2), ylimMR(2)]; % Define y vertices
fill(xVertices, yVertices, [0.5, 0.5, 0.5], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Plot transparent rectangle

% Posterior sample of objective function (MSE)
DTparam = DTopt(DTopt.iter == iterplot, :); % Get current iteration data
Mfpred = zeros(num_Beta, nPost); % Initialize matrix for posterior predictions
for i = 1:nPost
    Mfpred(:, i) = fGLM(beta_range, DTparam.logbeta(i), DTparam.intercept(i), DTparam.sigma2(i), doExp); % Compute MSE predictions
end
plt = plot(beta_range, Mfpred / mseScale, 'Color', blueshade, 'LineStyle', '-'); % Plot MSE predictions
h1 = plt(1); % Pick one sample curve for the legend

% Plot posterior sample optima
p5 = scatter(DTopt.betaOpt(DTopt.iter == iterplot), DTopt.fOpt(DTopt.iter == iterplot) / mseScale, ...
    'MarkerFaceColor', greenshade2, 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.5); % Scatter plot of optima

% Plot objective function and optimum from GLM mean model
currentModel = DTmodel(DTmodel.iter == iterplot, :); % Get current model
p1 = plot(currentModel.beta, currentModel.fpred / mseScale, 'color', blue, 'LineWidth', 2); % GLM prediction

% Plot Bayesian GLM optima
p6 = plot(DToptLm.betaOpt(DToptLm.iter == iterplot), DToptLm.fOpt(DToptLm.iter == iterplot) / mseScale, ...
    'x', 'Color', blue, 'MarkerSize', sz, 'LineWidth', 2); 

% Plot local regression result
p4 = plot(DTbest.beta, DTbest.mseLocal / mseScale, 'Color', 'r', 'LineWidth', 2); % Local regression estimate
[~, idxMin] = min(DTbest.mseLocal); % Get index of minimum MSE
p8 = plot(DTbest.beta(idxMin), DTbest.mseLocal(idxMin) / mseScale, 'x', 'Color', 'r', 'MarkerSize', 10, 'LineWidth', 2); % Best MSE point

% Plot vertical line at local regression betaOpt
xline(betaOptLocal, '--', 'LineWidth', 1);

% Plot GLM all data objective function and optimum
p3 = plot(DTm1.beta, DTm1.fpred / mseScale, 'Color', 'b', 'LineWidth', 2); % GLM with all data
p7 = plot(optLm1(1), optLm1(2) / mseScale, 'x', 'Color', 'b', 'MarkerSize', 10, 'LineWidth', 2); % Optimum from GLM with all data

% Add legend
legend([p1, h1, p3, p4, p5, p6, p7, p8], {'f() via GLM: N = ' + string(height(DT0)), ...
    'f() via Bayesian GLM: posterior sample', ...
    'f() via GLM: all data', ...
    'f() via local regression', ...
    '($\beta^*$, f$^*$) of posterior sample', ...
    '($\beta^*$, f$^*$) via GLM', ...
    '($\beta^*$, f$^*$) via GLM: all data', ...
    '($\beta^*$, f$^*$) via local regression'}, ...
    'Location', 'northeast', 'Box', 'off'); % Legend positioning
hold off; % Release hold on plot


filename = 'f_glm_TS_Ex1.pdf';
output_path = fullfile(pwd, 'Results', filename);% Combine into a full relative path
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure

%% plot 3: Histogram
betaOptTS = DTopt.betaOpt(DTopt.iter == iterplot); % Extract posterior samples for beta*
<<<<<<< HEAD
beta_opt_varTS = var(betaOptTS);
fprintf('Posterior beta* variance: %.6g\n', beta_opt_varTS);
=======

>>>>>>> b3229bacb6f237d710e56b960cef1e890be6f428
% Create figure
figure('Color', [1 1 1], 'Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on

% Plot histogram of posterior beta* samples
h1 = histogram(betaOptTS, 'BinWidth', 1); % Set bin width to 1
h1.FaceColor = greenshade2; % Set face color
h1.FaceAlpha = 0.5; % Set transparency

% Plot vertical lines for various beta* values
p2 = xline(DToptLm.betaOpt(DToptLm.iter == iterplot), 'LineWidth', th); % Beta from Bayesian GLM
p2.Color = blue; % Color of line

p3 = xline(optLm1(1), 'b', 'LineWidth', th); % Beta from GLM using all data
p4 = xline(DTbest.beta(idxMin), 'r', 'LineWidth', th); % Beta from local regression

<<<<<<< HEAD

=======
ylim([0 18]); % Set y-axis limits
>>>>>>> b3229bacb6f237d710e56b960cef1e890be6f428

% Set axis labels
ylabel('Frequency'); 
xlabel('$\beta^*$'); 

% Add legend
lgd = legend([h1, p2, p3, p4], {'$\beta^*$ of posterior sample', ...
    '$\beta^*$ via GLM', ...
    '$\beta^*$ via GLM: all data', ...
    '$\beta^*$ via local regression'}, ...
    'Location', 'northeast', 'Box', 'off'); % Position the legend

hold off; % Release hold on plot

filename = 'hist_betaOpt_Ex1.pdf';
output_path = fullfile(pwd, 'Results', filename);% Combine into a full relative path
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure

%% plot 4: series sample
iter = [zeros(nInitial, 1); repelem((1:BO_iter)', nBatch)]; % Create iterations array

% Create a table for plotting
DT_plot3 = table('Size', [length(iter), 2], 'VariableTypes', {'double', 'double'}, 'VariableNames', {'Iter', 'Beta'});
DT_plot3.Iter = iter; % Assign iteration values
DT_plot3.Beta = DTid.beta; % Assign beta values

% Create the plot
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]); 
hold on;

% Set up axes limits and labels
xlim([-1, BO_iter+1]); % Adjust x-axis for visibility
ylim([60, boundBeta(2)]); % Set y-axis limits for better visibility

% Scatter plot for beta samples
scatter(DT_plot3.Iter, DT_plot3.Beta, 'k'); 

% Add rectangle for optimal region
xFill = [-1, BO_iter+0.5, BO_iter+0.5, -1]; % X-coordinates of rectangle
yFill = [betaOptRange(1), betaOptRange(1), betaOptRange(end), betaOptRange(end)]; % Y-coordinates of rectangle
fill(xFill, yFill, [0, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Draw filled rectangle

% Add legend
legend('$\beta$ sample', 'Optimal region', 'Location', 'southeast', 'Box', 'off'); 

% Annotate axes with labels and ticks
xticks(0:5:BO_iter); % Set x-tick marks
xlabel('BO iteration', 'VerticalAlignment', 'top'); % x-axis label
ylabel('$\beta$'); % y-axis label

filename = 'series_sample_Ex1.pdf';
output_path = fullfile(pwd, 'Results', filename);% Combine into a full relative path
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure

%% plot 5: series beta opt
data_plot5 = DTopt.betaOpt; % Get the betaOpt values
data_plot5 = reshape(data_plot5, nPost, BO_iter+1); % Reshape for plotting
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]); % Create figure

% Create group data starting at 0
numGroups = size(data_plot5, 2); % Number of groups
groupData = repelem(0:(numGroups-1), size(data_plot5, 1))'; % Groups: 0, 1, 2

% Reshape data into a vector for boxchart
dataVector = data_plot5(:); % Convert matrix to a single column vector
h = boxchart(groupData, dataVector, 'MarkerColor', greenshade2', 'BoxFaceColor', greenshade2, 'BoxFaceAlpha', 0.5); % Plot boxchart
hold on;

% Plot iteration-wise beta optimum
p1 = plot(DToptLm.iter, DToptLm.betaOpt, '-o', 'color', blue, 'MarkerSize', 4, 'LineWidth', th); 

% Add horizontal lines for GLM and local regression
p2 = yline(optLm1(1), 'b', 'LineWidth', 2); % β* GLM all data
p3 = yline(DTbest.beta(idxMin), 'r', 'LineWidth', 2); % β* via local regression

% Add rectangle for optimal region using the fill function
xFill = [-0.5, BO_iter+1, BO_iter+1, -0.5]; % X-coordinates of rectangle
yFill = [betaOptRange(1), betaOptRange(1), betaOptRange(end), betaOptRange(end)]; % Y-coordinates of rectangle
fill(xFill, yFill, [0, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Draw filled rectangle

% Customize axes
ax = gca; % Get current axes
ax.XTick = 0:5:BO_iter+1; % Set x-tick positions
ax.XTickLabel = 0:5:BO_iter+1; % Set x-tick labels
xlabel('BO iteration'); % x-axis label
ylabel('$\beta^*$'); % y-axis label

xlim([-0.5, BO_iter+1]); % Set x-axis limits
ylim([60, boundBeta(2)]); % Set y-axis limits

% Add legend
legend([h, p1, p2, p3], {'$\beta^*$ posterior sample', ...
    '$\beta^*$ via GLM', ...
    '$\beta^*$ via GLM: all data', ...
    '$\beta^*$ via local regression'}, ...
    'Location', 'southeast', 'NumColumns', 2, 'Box', 'off'); % Position legend

filename = 'series_betaOpt_Ex1.pdf';
output_path = fullfile(pwd, 'Results', filename);% Combine into a full relative path
<<<<<<< HEAD
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure

%% Save summary stats
bo_stats.nDataBO       = nDataBO;
bo_stats.betaOpt       = betaOptBO;
bo_stats.betaOptLocal  = betaOptLocal;
bo_stats.sigma_betaOpt = std(betaOptTS);
save(fullfile(pwd, 'Results', 'bo_stats_Ex1.mat'), '-struct', 'bo_stats');
=======
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure
>>>>>>> b3229bacb6f237d710e56b960cef1e890be6f428
