%% Ablation study for Ex3 BO: initial design bounds (log-uniform sampling)
% Runs three settings for initial sampling bounds while keeping BO bounds fixed.

rng(56); % Base seed for reproducibility

%% Add Model folder into path
parentDir = fileparts(pwd);
targetFolder = fullfile(parentDir, 'Model');
if isfolder(targetFolder)
    addpath(targetFolder);
    disp(['Successfully added path: ', targetFolder]);
else
    warning(['The directory "', targetFolder, '" does not exist.']);
end

%% Load data
load("dist.mat"); % dist matrix (Nrep x Nbeta), beta, Nrep, Nbeta, doExp
load("DTstat_table.mat", "DTstat"); % summary table

k = min(beta);
doExp = doExp;

%% Build full DT table once
J = Nrep;
beta_vec = beta(:)';
rep_range = 1:J;

beta_mat = repmat(beta_vec, J, 1);
rep_mat = repmat(rep_range', 1, numel(beta_vec));

beta_col = beta_mat(:);
rep_col = rep_mat(:);
dosrom_col = dist(:);
logbeta_col = log(beta_col);
logdosrom_col = log(dosrom_col);

DT = table(beta_col, rep_col, dosrom_col, logbeta_col, logdosrom_col, ...
    'VariableNames', {'beta', 'rep', 'dosrom', 'logbeta', 'logdosrom'});

%% BO parameters (fixed across ablations)
nInitial = 40;
nBatch = 10;
nPost = 100;
maxIter = 50;

% BO bounds remain fixed for all runs
boundBeta = [min(beta_vec), max(beta_vec)];
beta_grid = beta_vec;

%% Ablation settings: initial bounds
init_bounds = [k, k + 5, k + 10];
init_ub = boundBeta(2);

results = table('Size', [numel(init_bounds), 5], ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'init_lb', 'init_ub', 'betaOptBO', 'nDataBO', 'iterStop'});

for i = 1:numel(init_bounds)
    % Reset seed for each run for comparability
    rng(56);

    init_lb = init_bounds(i);

    % Initial sample on log-uniform grid within [init_lb, init_ub]
    logbetaGrid = linspace(log(init_lb), log(init_ub), nInitial);
    betaValues = round(exp(logbetaGrid))';
    betaValues = min(max(betaValues, boundBeta(1)), boundBeta(2));
    betaValues = interp1(beta_grid, beta_grid, betaValues, 'nearest');

    DT0id = table('Size', [nInitial, 2], ...
        'VariableNames', {'beta', 'rep'}, 'VariableTypes', {'double', 'double'});
    DT0id.beta = betaValues;
    [groups, ~, groupIndices] = unique(DT0id.beta);
    for g = 1:length(groups)
        DT0id.rep(groupIndices == g) = 1:sum(groupIndices == g);
    end

    DT0 = innerjoin(DT, DT0id, 'Keys', {'beta', 'rep'});

    % First BO policy computation
    ret = BOpolicy(DT0, DTstat, 0, doExp, nPost, true);
    DToptLm = ret.DToptLm;
    DTopt = ret.DTopt;

    % BO iterations
    for BO_iter = 1:maxIter
        betaNew = DTopt.betaOpt(DTopt.iter == BO_iter-1);
        betaNew = betaNew(1:nBatch);

        betaNew(betaNew < boundBeta(1)) = boundBeta(1);
        betaNew(betaNew > boundBeta(2)) = boundBeta(2);
        betaNew = interp1(beta_grid, beta_grid, betaNew, 'nearest');

        DTidnew = table(sort(round(betaNew)), Inf(size(betaNew)), true(size(betaNew)), ...
            'VariableNames', {'beta', 'rep', 'isNew'});
        DT0id = table(DT0.beta, DT0.rep, false(size(DT0.beta)), ...
            'VariableNames', {'beta', 'rep', 'isNew'});
        DTid = [DT0id; DTidnew];

        [groups, ~, groupIndices] = unique(DTid.beta);
        for g = 1:length(groups)
            DTid.rep(groupIndices == g) = 1:sum(groupIndices == g);
        end

        DTidnew = DTid(DTid.isNew, :);
        DTnew = innerjoin(DT, DTidnew, 'Keys', {'beta', 'rep'});
        DT0 = [DT0; DTnew(:, 1:5)];

        ret = BOpolicy(DT0, DTstat, BO_iter, doExp, nPost, true);
        DToptLm = [DToptLm; ret.DToptLm];
        DTopt = [DTopt; ret.DTopt];

        DTscore = grpstats(DTopt, 'iter', @(x) quantile(x, 0.98), 'DataVars', 'relfLm');
        if sum(DTscore{:, 3} < 1.03) >= 10
            break;
        end
    end

    betaOptBO = round(DToptLm.betaOpt(end));
    nDataBO = height(DT0);
    iterStop = BO_iter;

    results{i, :} = [init_lb, init_ub, betaOptBO, nDataBO, iterStop];
    fprintf('Run %d: init=[%d,%d], betaOptBO=%d, nData=%d, iterStop=%d\n', ...
        i, init_lb, init_ub, betaOptBO, nDataBO, iterStop);
end

disp('Ablation summary:');
disp(results);
