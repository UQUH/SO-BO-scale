function run_bo_seeded(example_name, seed, output_path)
%RUN_BO_SEEDED Run one seeded BO robustness job and save standardized output.

fprintf('run_bo_seeded start: %s seed=%d\n', example_name, seed);

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
example_root = fullfile(repo_root, example_name);
bo_root = fullfile(example_root, 'BO');
model_root = fullfile(example_root, 'Model');
truth_path = fullfile(repo_root, 'Robustness', example_name, 'truth_data.mat');

addpath(bo_root);
addpath(model_root);

cfg = get_config(example_name);
truth = load(truth_path);
fprintf('Loaded config and truth: %s\n', truth_path);

out_dir = fileparts(output_path);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

rng(seed);
dist_data = load(fullfile(model_root, 'dist.mat'));
dtstat_data = load(fullfile(model_root, 'DTstat_table.mat'), 'DTstat');
dist = dist_data.dist;
DTstat = dtstat_data.DTstat;
fprintf('Loaded model data from %s\n', model_root);

if ismember(example_name, {'Ex3', 'Ex3_sigma03'})
    beta_grid = dist_data.beta(:);
    num_beta = numel(beta_grid);
    do_exp = dist_data.doExp;
else
    [~, num_beta] = size(dist);
    beta_grid = (cfg.k:(cfg.k + num_beta - 1))';
    do_exp = cfg.doExp;
end

dt = build_dt(example_name, dist, beta_grid);
evaluated_beta_seq = [];
fprintf('Built DT with %d rows\n', height(dt));

% Initial design
logbeta_grid = linspace(min(DTstat.logbeta), max(DTstat.logbeta), cfg.nInitial);
beta_values = round(exp(logbeta_grid))';
if ismember(example_name, {'Ex3', 'Ex3_sigma03'})
    beta_values = interp1(DTstat.beta, DTstat.beta, beta_values, 'nearest');
end
evaluated_beta_seq = [evaluated_beta_seq; beta_values(:)];

dt0id = table('Size', [cfg.nInitial, 2], ...
    'VariableNames', {'beta', 'rep'}, ...
    'VariableTypes', {'double', 'double'});
dt0id.beta = beta_values;

[groups, ~, group_indices] = unique(dt0id.beta);
for g = 1:length(groups)
    dt0id.rep(group_indices == g) = 1:sum(group_indices == g);
end

dt0 = innerjoin(dt, dt0id, 'Keys', {'beta', 'rep'});

ret = BOpolicy(dt0, DTstat, 0, do_exp, cfg.nPost, true);
dtOptLm = ret.DToptLm;
dtOpt = ret.DTopt;
DTmodel = ret.DTmodel;
fprintf('Completed initial BOpolicy\n');

bo_iter = 0;
bound_beta = [min(beta_grid), max(beta_grid)];
for iter_idx = 1:cfg.maxIter
    beta_new = dtOpt.betaOpt(dtOpt.iter == iter_idx - 1);
    beta_new = beta_new(1:cfg.nBatch);
    beta_new(beta_new < bound_beta(1)) = bound_beta(1);
    beta_new(beta_new > bound_beta(2)) = bound_beta(2);
    beta_new = round(beta_new);

    if ismember(example_name, {'Ex3', 'Ex3_sigma03'})
        beta_new = interp1(DTstat.beta, DTstat.beta, beta_new, 'nearest');
    end

    evaluated_beta_seq = [evaluated_beta_seq; beta_new(:)];

    dtid_new = table(sort(beta_new), Inf(size(beta_new)), true(size(beta_new)), ...
        'VariableNames', {'beta', 'rep', 'isNew'});
    dt0id = table(dt0.beta, dt0.rep, false(size(dt0.beta)), ...
        'VariableNames', {'beta', 'rep', 'isNew'});
    dtid = [dt0id; dtid_new];

    [groups, ~, group_indices] = unique(dtid.beta);
    for g = 1:length(groups)
        dtid.rep(group_indices == g) = 1:sum(group_indices == g);
    end

    dtid_new = dtid(dtid.isNew, :);
    dt_new = innerjoin(dt, dtid_new, 'Keys', {'beta', 'rep'});
    dt0 = [dt0; dt_new(:, 1:5)];

    ret = BOpolicy(dt0, DTstat, iter_idx, do_exp, cfg.nPost, true);
    dtOptLm = [dtOptLm; ret.DToptLm];
    dtOpt = [dtOpt; ret.DTopt];
    DTmodel = [DTmodel; ret.DTmodel];

    bo_iter = iter_idx;
    dtscore = grpstats(dtOpt, 'iter', @(x) quantile(x, 0.98), 'DataVars', 'relfLm');
    if sum(dtscore{:, 3} < 1.03) >= 3
        break;
    end
end

eval_true_mse_seq = interp1(truth.beta_grid, truth.true_mse, evaluated_beta_seq, 'linear', 'extrap');
best_true_mse_so_far_seq = cummin(eval_true_mse_seq);
eval_count_seq = (1:numel(evaluated_beta_seq))';

final_recommended_beta = round(dtOptLm.betaOpt(end));
final_recommended_true_mse = interp1(truth.beta_grid, truth.true_mse, final_recommended_beta, 'linear', 'extrap');
[final_best_true_mse, best_idx] = min(eval_true_mse_seq);
final_best_evaluated_beta = evaluated_beta_seq(best_idx);

results = struct();
results.seed = seed;
results.method = 'BO';
results.example = example_name;
results.evaluated_beta_seq = evaluated_beta_seq;
results.eval_true_mse_seq = eval_true_mse_seq;
results.best_true_mse_so_far_seq = best_true_mse_so_far_seq;
results.eval_count_seq = eval_count_seq;
results.final_recommended_beta = final_recommended_beta;
results.final_recommended_true_mse = final_recommended_true_mse;
results.final_best_evaluated_beta = final_best_evaluated_beta;
results.final_best_true_mse = final_best_true_mse;
results.stop_eval_count = numel(evaluated_beta_seq);
results.total_mc_evaluations = numel(evaluated_beta_seq);
results.n_mc_per_objective = 1;
results.is_deterministic = false;
results.bo_iter = bo_iter;
results.n_initial = cfg.nInitial;
results.n_batch = cfg.nBatch;
results.n_post = cfg.nPost;

raw_dir = fullfile(out_dir, 'raw', sprintf('seed_%03d', seed));
if ~exist(raw_dir, 'dir')
    mkdir(raw_dir);
end

iterplot = bo_iter;
beta_samples = evaluated_beta_seq;
beta_min = min(beta_grid);
beta_max = max(beta_grid);
n_eval = numel(evaluated_beta_seq);
n_eval_mc = n_eval;
doExp = do_exp;
DToptLm = dtOptLm;
DTopt = dtOpt;

% Use script key so Ex3_sigma03 raw file is named bo_ex3.mat (consistent with GP/SOTA)
if strcmp(example_name, 'Ex3_sigma03')
    raw_script_key = 'ex3';
else
    raw_script_key = lower(example_name);
end
raw_output_path = fullfile(raw_dir, sprintf('bo_%s.mat', raw_script_key));
save(raw_output_path, 'seed', 'example_name', 'doExp', 'beta_samples', ...
    'beta_grid', 'beta_min', 'beta_max', 'iterplot', 'n_eval', 'n_eval_mc', ...
    'DToptLm', 'DTopt', 'DTmodel');
results.raw_output_path = raw_output_path;

save(output_path, '-struct', 'results');
fprintf('Saved BO robustness result: %s\n', output_path);
end

function cfg = get_config(example_name)
switch example_name
    case 'Ex1'
        cfg = struct('k', 8, 'doExp', 3.456255510150231e-04, ...
            'nInitial', 40, 'nBatch', 10, 'nPost', 100, 'maxIter', 50);
    case 'Ex2'
        cfg = struct('k', 10, 'doExp', 16.1938, ...
            'nInitial', 40, 'nBatch', 10, 'nPost', 100, 'maxIter', 50);
    case {'Ex3', 'Ex3_sigma03'}
        cfg = struct('k', [], 'doExp', [], ...
            'nInitial', 10, 'nBatch', 10, 'nPost', 100, 'maxIter', 50);
    otherwise
        error('Unsupported example: %s', example_name);
end
end

function dt = build_dt(example_name, dist, beta_grid)
[nrep, num_beta] = size(dist);
beta_mat = repmat(beta_grid(:)', nrep, 1);
rep_mat = repmat((1:nrep)', 1, num_beta);

beta_col = beta_mat(:);
rep_col = rep_mat(:);
dosrom_col = dist(:);
logbeta_col = log(beta_col);
logdosrom_col = log(dosrom_col);

dt = table(beta_col, rep_col, dosrom_col, logbeta_col, logdosrom_col, ...
    'VariableNames', {'beta', 'rep', 'dosrom', 'logbeta', 'logdosrom'});

if ismember(example_name, {'Ex3', 'Ex3_sigma03'})
    dt.beta = round(dt.beta);
end
end
