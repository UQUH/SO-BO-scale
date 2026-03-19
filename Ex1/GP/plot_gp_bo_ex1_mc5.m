%% Plot GP-based Bayesian Optimization Results for Ex1
clear; clc; close all;

%% Plot styling (consistent with BO folder)
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
sz = 10;  % marker size

% Color palette
blue = [30, 120, 179] / 256;
blueshade = [166, 204, 227] / 256;
green = [48, 158, 43] / 256;
greenshade = [176, 222, 138] / 256;
greenshade2 = [48, 158, 43] / 256;

%% Setup paths
script_dir = fileparts(mfilename('fullpath'));

%% Load results
results_dir = fullfile(script_dir, 'Results', 'mc5');
data = load(fullfile(results_dir, 'bo_ex1_mc5.mat'));

%% Extract data
% BO samples
iter = double(data.iter(:));
beta_samples = double(data.beta_samples(:));
mse_samples = double(data.mse_samples(:));
s_samples = double(data.s_samples(:));

% GP posterior
gp_grid = double(data.gp_grid(:));
gp_mean = double(data.gp_mean(:));
gp_std = double(data.gp_std(:));
if isfield(data, 'use_log_transform')
    is_log_gp = logical(data.use_log_transform);
else
    is_log_gp = any(gp_mean < 0);
end
has_post = isfield(data, 'gp_posterior_samples');
if has_post
    gp_posterior_samples = double(data.gp_posterior_samples);
    gp_posterior_opt_beta = double(data.gp_posterior_opt_beta(:));
    gp_posterior_opt_val = double(data.gp_posterior_opt_val(:));
end

% Detect log-space GP (MSE should never be negative)
is_log_gp = any(gp_mean < 0);
if is_log_gp
    gp_mean_plot = exp(gp_mean);
    gp_lower = exp(gp_mean - 2 * gp_std);
    gp_upper = exp(gp_mean + 2 * gp_std);
else
    gp_mean_plot = gp_mean;
    gp_lower = gp_mean - 2 * gp_std;
    gp_upper = gp_mean + 2 * gp_std;
end

% Best results from GP
beta_opt_gp = data.beta_opt_gp;  % Rounded integer for display
beta_opt_gp_exact = data.beta_opt_gp_exact;  % Exact grid value for plotting
mse_opt_gp = data.mse_opt_gp;

% Parameters
beta_min = double(data.beta_min);
beta_max = double(data.beta_max);
doExp = double(data.doExp);
nDataBO = length(beta_samples);
if isfield(data, 'n_eval_mc')
    nDataMC = double(data.n_eval_mc);
elseif isfield(data, 'n_mc_var')
    nMC = double(data.n_mc_var);
    nDataMC = nDataBO * nMC;
else
    nDataMC = nDataBO;
end

%% Compute local regression using existing function
model_dir = fullfile(script_dir, '..', 'Model');
DTstat_data = load(fullfile(model_dir, 'DTstat_table.mat'));
DTstat = DTstat_data.DTstat;

% Call local_regression function
[DTbest, beta_opt_local, betaOptRange] = local_regression(DTstat);

% Extract local regression results
beta_range = DTbest.beta;
mse_local = DTbest.mseLocal;
mse_local_min = min(mse_local);
beta_opt_range_min = min(betaOptRange);
beta_opt_range_max = max(betaOptRange);
tol_f = 0.1;  % 10% tolerance (same as in local_regression.m)

%% Scale factor for normalized plot
mse_scale = mse_local_min;

%% Figure 1: Normalized Objective Function (f/f*)
f = figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on;

ylimMR = [0, 8];
xlim([22, beta_max]);
ylim(ylimMR);
xlabel('$\beta$');
ylabel('$f / f^*$');

% Plot the optimal region
xVertices = [beta_opt_range_min, beta_opt_range_max, beta_opt_range_max, beta_opt_range_min];
yVertices = [ylimMR(1), ylimMR(1), ylimMR(2), ylimMR(2)];
fill(xVertices, yVertices, [0.5, 0.5, 0.5], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

if has_post
    % Plot posterior sample curves and their optima
    nPost = size(gp_posterior_samples, 1);
    for i = 1:nPost
        post_curve = gp_posterior_samples(i, :).';
        if is_log_gp
            post_curve = exp(post_curve);
        end
       plt = plot(gp_grid, post_curve / mse_scale, 'Color', blueshade, 'LineWidth', 1);
    h1 = plt(1);
    end
    if is_log_gp
        opt_vals = exp(gp_posterior_opt_val);
    else
        opt_vals = gp_posterior_opt_val;
    end
    p2 = scatter(gp_posterior_opt_beta, opt_vals / mse_scale, 20, ...
        greenshade2, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');
else
    % Plot GP confidence interval (normalized)
    gp_lower = max(0, gp_lower / mse_scale);  % Clip at 0
    gp_upper = gp_upper / mse_scale;
    P2 = fill([gp_grid; flipud(gp_grid)], [gp_lower; flipud(gp_upper)], ...
        blueshade, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
end

% Plot GP mean (normalized)
p1 = plot(gp_grid, gp_mean_plot / mse_scale, 'color', blue, 'LineWidth', 2);

% Plot local regression (normalized)
p4 = plot(beta_range, mse_local / mse_scale, 'r', 'LineWidth', 2);

% GP mean optimum (use exact grid value so it sits on the curve)
p6 = plot(beta_opt_gp_exact, mse_opt_gp / mse_scale, 'x', 'Color', blue, 'MarkerSize', sz, 'LineWidth', 2);

% Local regression optimum
p8 = plot(beta_opt_local, 1, 'x', 'Color', 'r', 'MarkerSize', sz, 'LineWidth', 2);

% Vertical line at local regression betaOpt
xline(beta_opt_local, '--', 'LineWidth', 1);
legend([p1, h1, p4, p2, p6, p8], ...
        {['f() via GP mean: N = ', num2str(nDataMC)], ...
        'f() via GP: posterior sample', ...
        'f() via local regression', ...
        '($\beta^*$, f$^*$) of posterior sample', ...
        '($\beta^*$, f$^*$) via GP mean', ...
        '($\beta^*$, f$^*$) via local regression'}, ...
        'Location', 'northeast', 'Box', 'off');
hold off;

% Save figure
filename = 'f_gp_mc5_Ex1.pdf';
output_path = fullfile(results_dir, filename);
exportgraphics(gcf, output_path, 'ContentType', 'vector');
fprintf('Saved: %s\n', filename);

%% Figure 2: Histogram of beta samples
figure('Color', [1 1 1], 'Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on

if isfield(data, 'gp_posterior_opt_beta')
    hist_data = data.gp_posterior_opt_beta(:);
else
    hist_data = beta_samples;
end
beta_opt_mean = mean(hist_data);
beta_opt_var = var(hist_data);
h1 = histogram(hist_data, 'BinWidth',1);
h1.FaceColor = greenshade2;
h1.FaceAlpha = 0.5;

yl = ylim;
pOpt = fill([beta_opt_range_min, beta_opt_range_max, beta_opt_range_max, beta_opt_range_min], ...
    [0, 0, yl(2), yl(2)], [0.5, 0.5, 0.5], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

p2 = xline(beta_opt_gp_exact, 'LineWidth', th);
p2.Color = blue;

p4 = xline(beta_opt_local, 'r', 'LineWidth', th);

ylabel('Frequency');
xlabel('$\beta^*$');

lgd = legend([h1, p2, p4], ...
    {'$\beta^*$ of posterior samples', ...
    '$\beta^*$ via GP mean', ...
    '$\beta^*$ via local regression'}, ...
    'Location', 'northeast', 'Box', 'off');

hold off;

% Save figure
filename = 'hist_beta_gp_mc5_Ex1.pdf';
output_path = fullfile(results_dir, filename);
exportgraphics(gcf, output_path, 'ContentType', 'vector');
fprintf('Saved: %s\n', filename);

%% Print summary
fprintf('\n=== GP-based BO Results ===\n');
fprintf('Best beta (GP): %d\n', beta_opt_gp_exact);
fprintf('Best MSE (GP): %.6e\n', mse_opt_gp);
fprintf('Best beta (local): %d\n', beta_opt_local);
fprintf('Best MSE (local): %.6e\n', mse_local_min);
fprintf('Total samples: %d\n', nDataBO);
fprintf('Total MC samples: %d\n', nDataMC);
fprintf('Reference doExp: %.6f\n', doExp);
fprintf('Posterior beta* variance: %.6g\n', beta_opt_var);
