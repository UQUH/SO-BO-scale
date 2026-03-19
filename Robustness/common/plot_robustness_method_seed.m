function plot_robustness_method_seed(example_name, method_name, seed)
%PLOT_ROBUSTNESS_METHOD_SEED Plot method-specific robustness diagnostics for one seed.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
example_root = fullfile(repo_root, example_name);
robust_example_root = fullfile(repo_root, 'Robustness', example_name);
seed_tag = sprintf('seed_%03d', seed);

addpath(fullfile(example_root, 'BO'));
addpath(fullfile(example_root, 'GP'));

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

dtstat_data = load(fullfile(example_root, 'Model', 'DTstat_table.mat'));
DTstat = dtstat_data.DTstat;
[DTbest, beta_opt_local, betaOptRange] = local_regression(DTstat);
beta_range = DTbest.beta;
mse_local = DTbest.mseLocal;
mse_local_min = min(mse_local);
beta_opt_range_min = min(betaOptRange);
beta_opt_range_max = max(betaOptRange);
mse_scale = mse_local_min;

[data, results_dir, method_slug, method_mean_label, method_band_label, ...
    method_opt_label, method_hist_label, hist_data, beta_samples, beta_plot, curve_mean, curve_lower, ...
    curve_upper, beta_opt_method, mse_opt_method, beta_min, beta_max, ...
    doExp, nDataBO, nDataMC] = load_method_payload(example_name, method_name, ...
    robust_example_root, seed_tag);

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

p2 = fill([beta_plot; flipud(beta_plot)], ...
    [curve_lower / mse_scale; flipud(curve_upper / mse_scale)], ...
    blueshade, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
p1 = plot(beta_plot, curve_mean / mse_scale, 'Color', blue, 'LineWidth', 2);
p4 = plot(beta_range, mse_local / mse_scale, 'r', 'LineWidth', 2);
p6 = plot(beta_opt_method, mse_opt_method / mse_scale, 'x', 'Color', blue, ...
    'MarkerSize', sz, 'LineWidth', 2);
p8 = plot(beta_opt_local, 1, 'x', 'Color', 'r', 'MarkerSize', sz, 'LineWidth', 2);
xline(beta_opt_local, '--', 'LineWidth', 1);

legend([p1, p2, p4, p6, p8], ...
    {sprintf('%s: N = %d', method_mean_label, nDataMC), ...
    method_band_label, ...
    'f() via local regression', ...
    method_opt_label, ...
    '($\beta^*$, f$^*$) via local regression'}, ...
    'Location', 'northeast', 'Box', 'off');
hold off;

filename = sprintf('f_%s_%s_%s.pdf', method_slug, lower(example_name), seed_tag);
exportgraphics(gcf, fullfile(results_dir, filename), 'ContentType', 'vector');
fprintf('Saved: %s\n', filename);

figure('Color', [1 1 1], 'Units', 'inches', 'Position', [0, 0, width_plot, height_plot]);
hold on;

beta_sample_var = var(hist_data);
h1 = histogram(hist_data, 'BinWidth', 1);
h1.FaceColor = greenshade2;
h1.FaceAlpha = 0.5;

p2 = xline(beta_opt_method, 'LineWidth', th);
p2.Color = blue;
p4 = xline(beta_opt_local, 'r', 'LineWidth', th);

ylabel('Frequency');
xlabel('$\beta$');
legend([h1, p2, p4], ...
    {'$\beta^*$ of posterior samples', ...
    method_hist_label, ...
    '$\beta^*$ via local regression'}, ...
    'Location', 'northeast', 'Box', 'off');
hold off;

filename = sprintf('hist_beta_%s_%s_%s.pdf', method_slug, lower(example_name), seed_tag);
exportgraphics(gcf, fullfile(results_dir, filename), 'ContentType', 'vector');
fprintf('Saved: %s\n', filename);

fprintf('\n=== Robustness %s %s seed %d ===\n', example_name, method_name, seed);
fprintf('Best beta (method): %.6f\n', beta_opt_method);
fprintf('Best MSE (method): %.6e\n', mse_opt_method);
fprintf('Best beta (local): %d\n', beta_opt_local);
fprintf('Best MSE (local): %.6e\n', mse_local_min);
fprintf('Total objective evaluations: %d\n', nDataBO);
fprintf('Total MC samples: %d\n', nDataMC);
fprintf('Reference doExp: %.6f\n', doExp);
fprintf('Posterior beta* variance: %.6g\n', beta_sample_var);
end

function [data, results_dir, method_slug, method_mean_label, method_band_label, ...
    method_opt_label, method_hist_label, hist_data, beta_samples, beta_plot, curve_mean, curve_lower, ...
    curve_upper, beta_opt_method, mse_opt_method, beta_min, beta_max, ...
    doExp, nDataBO, nDataMC] = load_method_payload(example_name, method_name, ...
    robust_example_root, seed_tag)

% Use script key so Ex3_sigma03 still maps to 'ex3' in filenames
if strcmp(example_name, 'Ex3_sigma03')
    script_key = 'ex3';
else
    script_key = lower(example_name);
end

switch upper(method_name)
    case 'GP'
        results_dir = fullfile(robust_example_root, 'GP', 'Results', 'raw', seed_tag);
        data = load(fullfile(results_dir, sprintf('bo_%s.mat', script_key)));
        beta_samples = double(data.beta_samples(:));
        beta_plot = double(data.gp_grid(:));
        gp_mean = double(data.gp_mean(:));
        gp_std = double(data.gp_std(:));
        [curve_mean, curve_lower, curve_upper] = convert_saved_gp_curve( ...
            gp_mean, gp_std, double(data.mse_opt_gp), logical(data.use_log_transform));
        hist_data = double(data.gp_posterior_opt_beta(:));
        beta_opt_method = double(data.beta_opt_gp_exact);
        mse_opt_method = double(data.mse_opt_gp);
        beta_min = double(data.beta_min);
        beta_max = double(data.beta_max);
        doExp = double(data.doExp);
        nDataBO = length(beta_samples);
        if isfield(data, 'n_eval_mc')
            nDataMC = double(data.n_eval_mc);
        else
            nDataMC = nDataBO;
        end
        method_slug = 'gp';
        method_mean_label = 'f() via GP mean';
        method_band_label = 'GP predictive interval';
        method_opt_label = '($\beta^*$, f$^*$) via GP mean';
        method_hist_label = '$\beta^*$ via GP mean';

    case 'SOTA_BO'
        results_dir = fullfile(robust_example_root, 'SOTA_BO', 'Results', 'raw', seed_tag);
        data = load(fullfile(results_dir, sprintf('bo_%s_qlognei_mc10.mat', script_key)));
        beta_samples = double(data.beta_samples(:));
        beta_plot = double(data.gp_grid(:));
        gp_mean = double(data.gp_mean(:));
        gp_std = double(data.gp_std(:));
        [curve_mean, curve_lower, curve_upper] = convert_saved_gp_curve( ...
            gp_mean, gp_std, double(data.mse_opt_gp), logical(data.use_log_transform));
        hist_data = double(data.gp_posterior_opt_beta(:));
        beta_opt_method = double(data.beta_opt_gp_exact);
        mse_opt_method = double(data.mse_opt_gp);
        beta_min = double(data.beta_min);
        beta_max = double(data.beta_max);
        doExp = double(data.doExp);
        nDataBO = length(beta_samples);
        if isfield(data, 'n_eval_mc')
            nDataMC = double(data.n_eval_mc);
        else
            nDataMC = nDataBO;
        end
        method_slug = 'sota_qlognei';
        method_mean_label = 'f() via qLogNEI-GP mean';
        method_band_label = 'qLogNEI-GP predictive interval';
        method_opt_label = '($\beta^*$, f$^*$) via qLogNEI-GP mean';
        method_hist_label = '$\beta^*$ via qLogNEI-GP mean';

    case 'SOTA_BO_QNEI'
        results_dir = fullfile(robust_example_root, 'SOTA_BO_QNEI', 'Results', 'raw', seed_tag);
        data = load(fullfile(results_dir, sprintf('bo_%s_qnei_mc10.mat', script_key)));
        beta_samples = double(data.beta_samples(:));
        beta_plot = double(data.gp_grid(:));
        gp_mean = double(data.gp_mean(:));
        gp_std = double(data.gp_std(:));
        [curve_mean, curve_lower, curve_upper] = convert_saved_gp_curve( ...
            gp_mean, gp_std, double(data.mse_opt_gp), logical(data.use_log_transform));
        hist_data = double(data.gp_posterior_opt_beta(:));
        beta_opt_method = double(data.beta_opt_gp_exact);
        mse_opt_method = double(data.mse_opt_gp);
        beta_min = double(data.beta_min);
        beta_max = double(data.beta_max);
        doExp = double(data.doExp);
        nDataBO = length(beta_samples);
        if isfield(data, 'n_eval_mc')
            nDataMC = double(data.n_eval_mc);
        else
            nDataMC = nDataBO;
        end
        method_slug = 'sota_qnei';
        method_mean_label = 'f() via qNEI-GP mean';
        method_band_label = 'qNEI-GP predictive interval';
        method_opt_label = '($\beta^*$, f$^*$) via qNEI-GP mean';
        method_hist_label = '$\beta^*$ via qNEI-GP mean';

    case 'BO'
        results_dir = fullfile(robust_example_root, 'BO', 'Results', 'raw', seed_tag);
        data = load(fullfile(results_dir, sprintf('bo_%s.mat', script_key)));
        beta_samples = double(data.beta_samples(:));
        iterplot = double(data.iterplot);
        DTmodel = data.DTmodel;
        DTopt = data.DTopt;
        DToptLm = data.DToptLm;
        currentModel = DTmodel(DTmodel.iter == iterplot, :);
        DTparam = DTopt(DTopt.iter == iterplot, :);
        beta_plot = double(currentModel.beta);
        curve_mean = double(currentModel.fpred);

        Mfpred = zeros(numel(beta_plot), height(DTparam));
        for i = 1:height(DTparam)
            Mfpred(:, i) = fGLM(beta_plot, DTparam.logbeta(i), DTparam.intercept(i), DTparam.sigma2(i), data.doExp);
        end
        curve_lower = max(0, quantile(Mfpred, 0.025, 2));
        curve_upper = quantile(Mfpred, 0.975, 2);

        hist_data = double(DTopt.betaOpt(DTopt.iter == iterplot));
        beta_opt_method = double(DToptLm.betaOpt(DToptLm.iter == iterplot));
        mse_opt_method = double(DToptLm.fOpt(DToptLm.iter == iterplot));
        beta_min = double(data.beta_min);
        beta_max = double(data.beta_max);
        doExp = double(data.doExp);
        nDataBO = length(beta_samples);
        nDataMC = double(data.n_eval_mc);
        method_slug = 'bo';
        method_mean_label = 'f() via Bayesian GLM mean';
        method_band_label = 'Bayesian GLM posterior interval';
        method_opt_label = '($\beta^*$, f$^*$) via Bayesian GLM mean';
        method_hist_label = '$\beta^*$ via Bayesian GLM mean';

    otherwise
        error('Unsupported method: %s', method_name);
end
end

function [curve_mean, curve_lower, curve_upper] = convert_saved_gp_curve(gp_mean, gp_std, mse_opt_gp, use_log_transform)
if ~use_log_transform
    curve_mean = gp_mean;
    curve_lower = max(0, gp_mean - 2 * gp_std);
    curve_upper = gp_mean + 2 * gp_std;
    return;
end

raw_min = min(gp_mean);
err_raw = abs(raw_min - mse_opt_gp);
err_exp = abs(exp(raw_min) - mse_opt_gp);

if err_exp < err_raw
    curve_mean = exp(gp_mean);
    curve_lower = exp(gp_mean - 2 * gp_std);
    curve_upper = exp(gp_mean + 2 * gp_std);
else
    curve_mean = gp_mean;
    curve_lower = max(0, gp_mean - 2 * gp_std);
    curve_upper = gp_mean + 2 * gp_std;
end
end
