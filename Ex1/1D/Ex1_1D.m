%% Hyper-parameter training via 1-D optimization algorithm
% in conjunction with Monte Carlo sampling
rng(48) % Set random seed for reproducibility

%% Add Model and BO folder into path
% Get the parent directory of the current working directory
parentDir = fileparts(pwd);

% Define the target folder (which exists in the parent directory)
targetFolder = fullfile(parentDir, 'Model');

% Check if the directory exists before adding it to the path
if isfolder(targetFolder)
    addpath(targetFolder);
    disp(['Successfully added path: ', targetFolder]);
else
    warning(['The directory "', targetFolder, '" does not exist.']);
end

% Define the target folder (which exists in the parent directory)
targetFolder = fullfile(parentDir, 'BO');

% Check if the directory exists before adding it to the path
if isfolder(targetFolder)
    addpath(targetFolder);
    disp(['Successfully added path: ', targetFolder]);
else
    warning(['The directory "', targetFolder, '" does not exist.']);
end

%% Load data
load('DTstat_table.mat',"DTstat"); %load DTstat table

%% Optimization for hyperparameter selection
k = 8; % ROM (Reduced Order Model) dimension
doExp = 3.456255510150231e-04; % reference distance

% Start optimization timer
tic;

% Define search interval for beta
boundBeta = [k, 150];
num_Beta = boundBeta(2) - boundBeta(1) + 1;

% Set tolerance for optimization
options = optimset('TolX', 0.1);

% Perform bounded scalar optimization using fminbnd
[beta_eval, fval] = fminbnd(@(beta) fMC(beta, DTstat), ...
    boundBeta(1), boundBeta(2), options);

beta_opt_1D = round(beta_eval);     % Optimal beta value
disp(['Optimal Beta: ', num2str(beta_opt_1D)]);

%% Custom plot specifications
width_plot = 8;   % Width in inches
height_plot = 6;  % Height in inches

% Global figure settings for font and LaTeX rendering
set(groot, 'DefaultTextFontSize', 20); 
set(groot, 'DefaultAxesFontSize', 20); 
set(groot, 'DefaultAxesFontName','Helvetica');
set(groot, 'DefaultTextFontName', 'Helvetica');
set(groot, 'DefaultAxesTickLabelInterpreter','latex')
set(groot, 'DefaultLegendInterpreter','latex')
set(groot, 'DefaultTextInterpreter', 'latex');
set(groot, 'DefaultLegendFontSize', 20);
set(groot, 'DefaultLegendFontName', 'Helvetica');

th = 1.6;  % Line thickness

% Define color palette
blue = [30, 120, 179] / 256;
blueshade = [166, 204, 227] / 256;
green = [48, 158, 43] / 256;
greenshade = [176, 222, 138] / 256;
greenshade2 = [48, 158, 43] / 256;

% Create the result folder if it doesn't exist
if ~exist(fullfile(pwd, 'Results'), 'dir')
    mkdir(fullfile(pwd, 'Results'));
end

%% Prepare data for scatter plot
[DTbest, ~, betaOptRange] = local_regression(DTstat); % Get local regression data
mseLocalMin = min(DTbest.mseLocal); % Minimum of local MSE

% Get floor and ceil of each element
beta_floor = floor(store_beta_values);
beta_ceil  = ceil(store_beta_values);

% Combine and take unique
uniqueBeta = unique([beta_floor; beta_ceil]);

% Number of unique betas
N = length(uniqueBeta);

% Generate labels, each repeated twice
x = repelem(1:N/2, 2);

%% Plot 1: Beta Evolution Across Iterations
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]); 
scatter(x, uniqueBeta, 'k', 'filled'); % Scatter plot of beta values
hold on

% Add rectangle for optimal region
xFill = [-0.5, N+1, N+1, -0.5]; % X-coordinates of rectangle
yFill = [betaOptRange(1), betaOptRange(1), betaOptRange(end), betaOptRange(end)]; % Y-coordinates of rectangle
fill(xFill, yFill, [0, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Draw filled rectangle with transparency

xlim([0, N]); % Set x-axis limits
xticks(1:1:N); % Set x-axis tick marks for each iteration
xlabel('Iteration number'); % x-axis label
ylabel('$\beta$'); % y-axis label

legend('$\beta$ sample', 'Optimal region', 'Location', 'southeast', 'Box', 'off'); 

filename = 'series_betaOpt_1D_Ex1.pdf';
output_path = fullfile(pwd, 'Results', filename);% Combine into a full relative path
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure

%% Plot 2: Log-scale Objective Function vs Beta
% Extract the MSE values that correspond to those betas
mse_values = DTstat.mse(ismember(DTstat.beta, uniqueBeta))/mseLocalMin;

load("dist.mat")
mse_mat = (dist(:,uniqueBeta) - doExp).^2;
mse_mat_vec = mse_mat(:)/mseLocalMin;
%%
figure('Units', 'inches', 'Position', [0, 0, width_plot, height_plot]); 
ylimMR = [0, 3]; % y-axis limits for the plot
hold on

groupData = repelem(uniqueBeta, 1000)'; 
h = boxchart(groupData,mse_mat_vec,'MarkerColor',greenshade2','BoxFaceColor',greenshade2, 'BoxFaceAlpha', 0.5);
p1 = plot(uniqueBeta,mse_values,'-o','color',blue, 'MarkerSize', 4,'LineWidth',th); %iteration wise beta opt

%1D optimim 
p3 = plot(beta_opt_1D,DTstat.mse(ismember(DTstat.beta, beta_opt_1D))/mseLocalMin,'x', 'Color',blue, 'MarkerSize', 10, 'LineWidth', 2);

% Plot the optimal region (rectangle)
xVertices = [betaOptRange(1), betaOptRange(end), betaOptRange(end), betaOptRange(1)]; % Define x vertices
yVertices = [ylimMR(1), ylimMR(1), ylimMR(2), ylimMR(2)]; % Define y vertices
fill(xVertices, yVertices, [0.5, 0.5, 0.5], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Plot transparent rectangle

% Plot local regression result
p2 = plot(DTbest.beta, DTbest.mseLocal / mseLocalMin, 'Color', 'r', 'LineWidth', 2); % Local regression estimate
[~, idxMin] = min(DTbest.mseLocal); % Get index of minimum MSE
p4 = plot(DTbest.beta(idxMin), DTbest.mseLocal(idxMin) / mseLocalMin, 'x', 'Color', 'r', 'MarkerSize', 10, 'LineWidth', 2); % Best MSE point

ylim(ylimMR); % Set y-axis limits
xlim([60 120])
xlabel('$\beta$');
ylabel('$f / f^*$'); 

% Add legend
legend([h, p1, p2, p3, p4], {'f() via MC samples', ...
    'f() via mean of MC samples', ...
    'f() via local regression', ...
    '($\beta^*$, f$^*$) via mean of MC samples', ...
    '($\beta^*$, f$^*$) via local regression'}, ...
    'Position', [0.38,0.65,0.15,0.15], 'Box', 'off'); % Legend positioning
hold off; % Release hold on plot

filename = 'Obj_fun_eval_1D_Ex1.pdf';
output_path = fullfile(pwd, 'Results', filename);% Combine into a full relative path
exportgraphics(gcf, output_path, 'ContentType', 'vector');% Export the figure