function plot_robustness_seed_suite(example_name)
%PLOT_ROBUSTNESS_SEED_SUITE Generate BO/GP/SOTA robustness plots for all saved seeds.

if nargin < 1 || isempty(example_name)
    example_name = 'all';
end

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
seed_file = fullfile(repo_root, 'Robustness', 'common', 'seeds.txt');
seeds = readmatrix(seed_file);

if ischar(example_name) || isstring(example_name)
    example_name = char(example_name);
else
    error('example_name must be a string or char.');
end

if strcmpi(example_name, 'all')
    examples = {'Ex1', 'Ex2', 'Ex3'};
else
    examples = {example_name};
end

for e = 1:numel(examples)
    current_example = examples{e};
    fprintf('\n=== Plotting Robustness Seeds: %s ===\n', current_example);
    for s = 1:numel(seeds)
        seed = seeds(s);
        fprintf('[%s] seed %d\n', current_example, seed);
        plot_robustness_method_seed(current_example, 'BO', seed);
        plot_robustness_method_seed(current_example, 'GP', seed);
        plot_robustness_method_seed(current_example, 'SOTA_BO', seed);
    end
end
end
