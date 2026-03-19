function plot_sota_bo_ex3(n_mc_eval, acquisition_function)
%% Plot SOTA noisy-BO baseline results for Ex3
if nargin < 1 || isempty(n_mc_eval)
    n_mc_eval = 1;
end
if nargin < 2 || isempty(acquisition_function)
    acquisition_function = 'qlognei';
end

switch lower(acquisition_function)
    case 'qnei'
        acq_label = 'qNEI';
    case 'qlognei'
        acq_label = 'qLogNEI';
    otherwise
        acq_label = acquisition_function;
end

clc; close all;

%% Plot styling
width_plot = 8;
height_plot = 6;

set(groot,'DefaultTextFontSize',20);
set(groot,'DefaultAxesFontSize',20);
set(groot,'DefaultAxesFontName','Helvetica');
set(groot,'DefaultTextFontName','Helvetica');
set(groot,'DefaultAxesTickLabelInterpreter','latex');
set(groot,'DefaultLegendInterpreter','latex');
set(groot,'DefaultTextInterpreter','latex');
set(groot,'DefaultLegendFontSize',20);
set(groot,'DefaultLegendFontName','Helvetica');

th = 1.6;
sz = 10;

blue = [30, 120, 179] / 256;
blueshade = [166, 204, 227] / 256;
greenshade2 = [48, 158, 43] / 256;

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', 'GP'));

results_tag = sprintf('%s_mc%d', acquisition_function, n_mc_eval);
results_dir = fullfile(script_dir, 'Results', results_tag);
data = load(fullfile(results_dir, sprintf('bo_ex3_%s.mat', results_tag)));

beta_samples = double(data.beta_samples(:));
gp_grid = double(data.gp_grid(:));
gp_mean = double(data.gp_mean(:));
gp_std = double(data.gp_std(:));
beta_opt_gp_exact = double(data.beta_opt_gp_exact);
mse_opt_gp = double(data.mse_opt_gp);
beta_min = double(data.beta_min);
beta_max = double(data.beta_max);
doExp = double(data.doExp);
nDataBO = length(beta_samples);
if isfield(data, 'n_eval_mc')
    nDataMC = double(data.n_eval_mc);
else
    nDataMC = nDataBO;
end

model_dir = fullfile(script_dir, '..', 'Model');
DTstat_data = load(fullfile(model_dir, 'DTstat_table.mat'));
DTstat = DTstat_data.DTstat;
[DTbest, beta_opt_local, betaOptRange] = local_regression(DTstat);

beta_range = DTbest.beta;
mse_local = DTbest.mseLocal;
mse_local_min = min(mse_local);
beta_opt_range_min = min(betaOptRange);
beta_opt_range_max = max(betaOptRange);
mse_scale = mse_local_min;

if isfield(data, 'gp_posterior_samples')
    post_samp = double(data.gp_posterior_samples);
    gp_lower = max(0, quantile(post_samp, 0.025, 1)');
    gp_upper = quantile(post_samp, 0.975, 1)';
else
    gp_lower = max(0, gp_mean - 2 * gp_std);
    gp_upper = gp_mean + 2 * gp_std;
end

figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on;

ylimMR = [0, 8];
xlim([beta_min, beta_max]);
ylim(ylimMR);
xlabel('$\beta$');
ylabel('$f / f^*$');

xVertices = [beta_opt_range_min, beta_opt_range_max, beta_opt_range_max, beta_opt_range_min];
yVertices = [ylimMR(1), ylimMR(1), ylimMR(2), ylimMR(2)];
fill(xVertices, yVertices, [0.5, 0.5, 0.5], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

if isfield(data, 'gp_posterior_samples')
    for i = 1:size(post_samp, 1)
        post_curve = post_samp(i, :).';
        plt = plot(gp_grid, post_curve / mse_scale, 'Color', blueshade, 'LineWidth', 1);
    end
    h1 = plt(1);
    opt_vals = double(data.gp_posterior_opt_val(:));
    p2 = scatter(double(data.gp_posterior_opt_beta), opt_vals / mse_scale, 20, ...
        greenshade2, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');
else
    p2 = fill([gp_grid; flipud(gp_grid)], [gp_lower / mse_scale; flipud(gp_upper / mse_scale)], ...
        blueshade, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    h1 = p2;
end
p1 = plot(gp_grid, gp_mean / mse_scale, 'Color', blue, 'LineWidth', 2);
p4 = plot(beta_range, mse_local / mse_scale, 'r', 'LineWidth', 2);
p6 = plot(beta_opt_gp_exact, mse_opt_gp / mse_scale, 'x', 'Color', blue, ...
    'MarkerSize', sz, 'LineWidth', 2);
p8 = plot(beta_opt_local, 1, 'x', 'Color', 'r', 'MarkerSize', sz, 'LineWidth', 2);
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

filename = sprintf('f_sota_%s_Ex3.pdf', results_tag);
exportgraphics(gcf, fullfile(results_dir, filename), 'ContentType', 'vector');
fprintf('Saved: %s\n', filename);

figure('Color', [1 1 1], 'Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on;

if isfield(data, 'gp_posterior_opt_beta')
    hist_data = double(data.gp_posterior_opt_beta(:));
else
    hist_data = beta_samples;
end
beta_sample_var = var(hist_data);
h1 = histogram(hist_data, 'BinWidth', 1);
h1.FaceColor = greenshade2;
h1.FaceAlpha = 0.5;


p2 = xline(beta_opt_gp_exact, 'LineWidth', th);
p2.Color = blue;
p4 = xline(beta_opt_local, 'r', 'LineWidth', th);

ylabel('Frequency');
xlabel('$\beta^*$');
legend([h1, p2, p4], ...
    {'$\beta^*$ of posterior samples', '$\beta^*$ via GP mean', '$\beta^*$ via local regression'}, ...
    'Location', 'northeast', 'Box', 'off');
hold off;

filename = sprintf('hist_beta_sota_%s_Ex3.pdf', results_tag);
exportgraphics(gcf, fullfile(results_dir, filename), 'ContentType', 'vector');
fprintf('Saved: %s\n', filename);

fprintf('\n=== SOTA %s Results (Ex3) ===\n', acq_label);
fprintf('Best beta (%s-GP): %.6f\n', acq_label, beta_opt_gp_exact);
fprintf('Best MSE (%s-GP): %.6e\n', acq_label, mse_opt_gp);
fprintf('Best beta (local): %d\n', beta_opt_local);
fprintf('Best MSE (local): %.6e\n', mse_local_min);
fprintf('Total samples: %d\n', nDataBO);
fprintf('Total MC samples: %d\n', nDataMC);
fprintf('Reference doExp: %.6f\n', doExp);
fprintf('Posterior beta* variance: %.6g\n', beta_sample_var);
