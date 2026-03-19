#!/usr/bin/env python3
"""
Bayesian Optimization for SS-PPCA Hyperparameter (beta) using BoTorch.

HETEROSCEDASTIC NOISE VERSION with log-transform.

This script optimizes the hyperparameter beta for the SS-PPCA method using
Gaussian Process based Bayesian Optimization with Thompson Sampling.

Uses pre-simulated data from dist.mat (1000 MC samples x 143 beta values).

Ex1: Linear Static Problem
- k = 8 (ROM dimension)
- doExp = 3.456255510150231e-04 (reference distance)
- Beta range: [8, 150]

Simply edit the parameters in the "USER PARAMETERS" section and run:
    python bo_ex1_hetero.py
"""

import os
import scipy.io as sio
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.constraints import Interval
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import gpytorch

# Suppress warnings (common with small-scale objectives)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="botorch")
warnings.filterwarnings("ignore", message=".*not p.d.*")
warnings.filterwarnings("ignore", message=".*not contained to the unit cube.*")
warnings.filterwarnings("ignore", message=".*Very small noise values.*")
warnings.filterwarnings("ignore", message=".*Negative variance values.*")

# =============================================================================
# USER PARAMETERS - Edit these as needed
# =============================================================================
# Note: beta_min, beta_max, and doExp are set from the loaded data

seed = int(os.environ.get("ROBUSTNESS_SEED", "12"))  # Random seed for reproducibility
n_initial = 10          # Number of initial samples
n_iterations = 50       # Number of BO iterations
batch_size = 10         # Number of candidates per iteration
grid_size = 200         # Grid size for Thompson Sampling
# Posterior beta* variance stopping criterion
posterior_var_tol = 3       # Stop when variance drops below this threshold
posterior_var_streak = 3    # Stop after this many consecutive hits
posterior_var_samples = 100  # Posterior samples to estimate beta* variance
# Use heteroscedastic noise - variance estimated from multiple MC samples
noise_model = "hetero"
n_mc_var = 5           # Number of MC samples to estimate variance at each beta
noise_floor = 0.01      # Minimum noise variance in LOG space (not MSE space)
debug_selection = False  # Extra prints to debug candidate selection
# Toggle per-iteration and final plots (disabled for MATLAB plotting)
enable_plots = False
plot_posterior_samples = 100  # Number of posterior samples to plot per iteration
ts_num_samples = 100         # Number of posterior samples for batch Thompson
final_posterior_samples = 100  # Number of posterior samples to export to MATLAB

# Use log-transform for MSE (helps with numerical stability for very small values)
use_log_transform = True

# =============================================================================
# GP HYPERPARAMETERS - Optimized for noisy optimization
# =============================================================================

# RBF kernel for log-noisy EI
kernel_type = "rbf"
kernel_list = ["rbf"]

# Length scale: controls smoothness of GP posterior
# NOTE: With Normalize(d=1) input transform, inputs are scaled to [0,1]
# So lengthscale should be relative to [0,1], not original [8, 150] range
lengthscale_init = 0.8            # Reasonable for normalized [0,1] input
lengthscale_bounds = (0.05, 1.0)  # Allow learning

# Output scale: let it be learned (Standardize handles scaling)
outputscale_init = None
outputscale_bounds = None

# Noise settings - relative to standardized scale (Standardize transform used)
# Since data is standardized, noise should be O(1), not O(1e-8)
noise_init = 0.5                   # Similar to Ex2
noise_bounds = (1e-4, 2.0)         # Reasonable bounds for standardized data

# =============================================================================
# ACQUISITION FUNCTION - Choose acquisition strategy
# =============================================================================

# Available options:
#   "thompson"  - Thompson Sampling (draw from posterior, select minimum)
#   "ei"        - Expected Improvement
#   "log_ei"    - Log Expected Improvement (more numerically stable)
#   "pi"        - Probability of Improvement
#   "ucb"       - Upper Confidence Bound (set ucb_beta below)
#   "qlognei"   - qLogNoisyExpectedImprovement (RECOMMENDED for noisy objectives)
acquisition_functions = ["thompson"]

# UCB-specific parameter (exploration-exploitation trade-off)
ucb_beta = 2.0  # Higher = more exploration

# =============================================================================
# SET RANDOM SEEDS
# =============================================================================

np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

# Counter for objective evaluations
eval_count = 0

# =============================================================================
# LOAD PRE-SIMULATED DATA
# =============================================================================

print("Loading pre-simulated data...")

# Get paths
script_dir = Path(__file__).resolve().parent
model_dir = script_dir.parent / "Model"

# Load pre-simulated distance matrix
# dist.mat contains: dist (Nrep x num_Beta) matrix of L2 distances
# Nrep = 1000 Monte Carlo repetitions
# num_Beta = 143 beta values (beta = k to k + num_Beta - 1, where k = 8)
dist_data = sio.loadmat(model_dir / "dist.mat")
dist_matrix = dist_data["dist"]  # Shape: (1000, 143)

Nrep, num_Beta = dist_matrix.shape
print(f"Loaded dist matrix: {Nrep} repetitions x {num_Beta} beta values")

# Define parameters based on loaded data (Ex1 specific)
k = 8  # ROM dimension (minimum beta value)
doExp = 3.456255510150231e-04  # Reference distance (from MATLAB code)

# Beta range: k to k + num_Beta - 1
beta_min = k
beta_max = k + num_Beta - 1

print(f"Beta range: [{beta_min}, {beta_max}]")
print(f"Reference distance (doExp): {doExp:.10e}")

# Beta range for reference
beta_range = np.arange(beta_min, beta_max + 1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dist_for_beta(beta, rep_idx):
    """
    Get the L2 distance for a given beta and repetition index from pre-simulated data.

    Parameters
    ----------
    beta : int
        Beta value (must be in range [k, k + num_Beta - 1])
    rep_idx : int
        Repetition index (0-indexed, must be in range [0, Nrep - 1])

    Returns
    -------
    float
        L2 distance (dosrom) for the given beta and repetition
    """
    # Convert beta to column index (beta = k corresponds to column 0)
    col_idx = beta - k

    # Ensure indices are valid
    col_idx = np.clip(col_idx, 0, num_Beta - 1)
    rep_idx = rep_idx % Nrep  # Wrap around if needed

    return float(dist_matrix[rep_idx, col_idx])


def evaluate_beta(beta, eval_seed=None):
    """
    Evaluate the SS-PPCA objective for a given beta using pre-simulated data.

    For heteroscedastic noise: uses multiple MC samples to estimate both
    mean MSE and its variance.

    Parameters
    ----------
    beta : float
        Beta value to evaluate
    eval_seed : int, optional
        Random seed for selecting which MC samples to use

    Returns
    -------
    mean_mse : float
        Mean MSE value across MC samples
    mean_dosrom : float
        Mean L2 distance value
    mse_var : float
        Variance of MSE (for heteroscedastic GP)
    """
    global eval_count
    eval_count += 1
    beta = int(round(beta))
    beta = np.clip(beta, k, k + num_Beta - 1)

    local_rng = np.random.default_rng(eval_seed)

    # Get multiple MC samples to estimate variance
    rep_indices = local_rng.choice(Nrep, size=n_mc_var, replace=False)

    mse_samples = []
    dosrom_samples = []
    for rep_idx in rep_indices:
        dosrom = get_dist_for_beta(beta, rep_idx)
        mse = (dosrom - doExp) ** 2
        mse_samples.append(mse)
        dosrom_samples.append(dosrom)

    mean_mse = np.mean(mse_samples)
    mean_dosrom = np.mean(dosrom_samples)
    mse_var = np.var(mse_samples, ddof=1)  # Sample variance

    # For log-transform: convert MSE variance to log-space variance using delta method
    # Var(log(X)) ≈ Var(X) / E[X]^2
    if use_log_transform and mean_mse > 0:
        log_var = mse_var / (mean_mse ** 2)
        # Apply noise floor in log space
        log_var = max(log_var, noise_floor)
        return float(mean_mse), float(mean_dosrom), float(log_var)
    else:
        return float(mean_mse), float(mean_dosrom), float(max(mse_var, noise_floor))


def plot_gp_iteration(model, train_x, train_y, bounds, grid, iteration,
                      next_beta, results_dir, acquisition_function, best_y):
    """
    Plot GP posterior with observations and save to file.
    """
    model.eval()

    with torch.no_grad():
        # Get posterior predictions
        posterior = model.posterior(grid)
        mean = posterior.mean.squeeze().numpy()
        torch_state = torch.random.get_rng_state()
        samples = posterior.rsample(torch.Size(
            [plot_posterior_samples])).squeeze().cpu().numpy()
        ts_sample = posterior.rsample().squeeze().cpu().numpy()
        torch.random.set_rng_state(torch_state)

    grid_np = grid.squeeze().numpy()
    train_x_np = train_x.squeeze().numpy()
    train_y_np = train_y.squeeze().numpy()

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left plot: GP posterior samples and optima ---
    ax1 = axes[0]
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
    for s in samples:
        ax1.plot(grid_np, s, color='blue', alpha=0.15, linewidth=1)
    min_indices = np.argmin(samples, axis=1)
    ax1.scatter(grid_np[min_indices], samples[np.arange(samples.shape[0]), min_indices],
                c='green', s=30, alpha=0.7, label='Posterior optima')
    ax1.plot(grid_np, mean, 'b-', linewidth=2, label='GP Mean')
    ax1.scatter(train_x_np, train_y_np, c='red', s=50, zorder=5,
                label='Observations', edgecolors='black')
    ax1.axvline(x=next_beta, color='green', linestyle='--', linewidth=2,
                label=f'Next: β={next_beta}')

    # Mark the best observed point
    best_idx = np.argmin(train_y_np)
    ax1.scatter(train_x_np[best_idx], train_y_np[best_idx],
                c='gold', s=150, marker='*', zorder=6,
                label=f'Best: β={int(train_x_np[best_idx])}', edgecolors='black')

    ax1.set_xlabel('β (beta)', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title(f'Iteration {iteration}: GP Posterior Samples', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_xlim(bounds[0, 0].item(), bounds[1, 0].item())
    ax1.grid(True, alpha=0.3)

    # --- Right plot: Acquisition / Thompson Sample ---
    ax2 = axes[1]
    if acquisition_function == "thompson":
        ax2.plot(grid_np, ts_sample, 'purple',
                 linewidth=2, label='Thompson Sample')
        ax2.scatter(train_x_np, train_y_np, c='red', s=50, zorder=5,
                    label='Observations', edgecolors='black')

        # Mark minimum of Thompson sample
        ts_min_idx = np.argmin(ts_sample)
        ax2.axvline(x=grid_np[ts_min_idx], color='green', linestyle='--', linewidth=2,
                    label=f'TS min: β≈{int(round(grid_np[ts_min_idx]))}')
        ax2.set_title(f'Iteration {iteration}: Thompson Sample', fontsize=14)
    else:
        if acquisition_function == "qlognei":
            sampler = SobolQMCNormalSampler(torch.Size([128]))
            objective = GenericMCObjective(lambda Y, X: -Y.squeeze(-1))
            acq = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_x,
                sampler=sampler,
                objective=objective,
                prune_baseline=True,
            )
        elif acquisition_function == "ei":
            acq = ExpectedImprovement(
                model=model, best_f=best_y, maximize=False)
        elif acquisition_function == "log_ei":
            acq = LogExpectedImprovement(
                model=model, best_f=best_y, maximize=False)
        elif acquisition_function == "pi":
            acq = ProbabilityOfImprovement(
                model=model, best_f=best_y, maximize=False)
        elif acquisition_function == "ucb":
            acq = UpperConfidenceBound(
                model=model, beta=ucb_beta, maximize=False)
        else:
            raise ValueError(
                f"Unknown acquisition_function: {acquisition_function}")

        with torch.no_grad():
            acq_grid = grid.unsqueeze(-2)
            acq_vals = acq(acq_grid).squeeze().numpy()

        ax2.plot(grid_np, acq_vals, 'purple',
                 linewidth=2, label=acquisition_function.upper())
        ax2.scatter(train_x_np, train_y_np, c='red', s=50, zorder=5,
                    label='Observations', edgecolors='black')
        ax2.axvline(x=next_beta, color='green', linestyle='--', linewidth=2,
                    label=f'Next: β={next_beta}')
        ax2.set_title(
            f'Iteration {iteration}: {acquisition_function.upper()}',
            fontsize=14,
        )

    ax2.set_xlabel('β (beta)', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.set_xlim(bounds[0, 0].item(), bounds[1, 0].item())
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save figure
    plot_path = results_dir / f"bo_iter_{iteration:02d}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def build_gp_model(train_x, train_y, train_yvar=None):
    """
    Build GP model with custom or default hyperparameters.

    Returns
    -------
    model : SingleTaskGP
        Configured GP model.
    """
    # Build custom covariance module if specified
    covar_module = None

    if kernel_type != "default":
        # Select base kernel
        if kernel_type == "rbf" or kernel_type == "se":
            base_kernel = RBFKernel()
        elif kernel_type == "matern12":
            base_kernel = MaternKernel(nu=0.5)
        elif kernel_type == "matern32":
            base_kernel = MaternKernel(nu=1.5)
        elif kernel_type == "matern52":
            base_kernel = MaternKernel(nu=2.5)
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        # Set length scale constraints
        if lengthscale_bounds is not None:
            base_kernel.lengthscale_constraint = Interval(
                lengthscale_bounds[0], lengthscale_bounds[1]
            )

        # Initialize length scale
        if lengthscale_init is not None:
            base_kernel.lengthscale = lengthscale_init

        # Wrap in ScaleKernel
        covar_module = ScaleKernel(base_kernel)

        # Set output scale constraints
        if outputscale_bounds is not None:
            covar_module.outputscale_constraint = Interval(
                outputscale_bounds[0], outputscale_bounds[1]
            )

        # Initialize output scale
        if outputscale_init is not None:
            covar_module.outputscale = outputscale_init

    likelihood = None
    if train_yvar is not None:
        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_yvar,
            learn_additional_noise=True,
        )

    # Build model
    if covar_module is not None:
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )
    else:
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            likelihood=likelihood,
            input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )

    # Set noise constraints if specified
    if train_yvar is None:
        if noise_bounds is not None:
            model.likelihood.noise_covar.noise_constraint = Interval(
                noise_bounds[0], noise_bounds[1]
            )

        # Initialize noise if specified
        if noise_init is not None:
            model.likelihood.noise = noise_init

    return model


def thompson_sampling(model, grid, batch_size=1, n_samples=100):
    """
    Select next candidates using Thompson Sampling.

    Draw a sample from the GP posterior and return the point(s)
    with minimum sampled value.

    Parameters
    ----------
    model : SingleTaskGP
        Fitted GP model.
    grid : torch.Tensor
        Grid of candidate points, shape (n_grid, 1).
    batch_size : int
        Number of candidates to select.

    Returns
    -------
    candidates : list
        List of selected beta values.
    """
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if debug_selection:
            print("  TS: computing posterior...", flush=True)
        posterior = model.posterior(grid)

        if debug_selection:
            print("  TS: sampling posterior...", flush=True)
        samples = posterior.rsample(torch.Size([n_samples])).squeeze()

    # Select candidates by taking the minimum of each posterior sample
    candidates = []
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
    min_indices = torch.argmin(samples, dim=1)
    for idx in min_indices:
        beta = int(round(grid[idx, 0].item()))
        if beta not in candidates:
            candidates.append(beta)
        if len(candidates) >= batch_size:
            break
    # If still short, fill with best points from posterior mean
    if len(candidates) < batch_size:
        mean = posterior.mean.squeeze()
        k_top = min(batch_size * 3, mean.numel())
        _, indices = torch.topk(mean, k_top, largest=False)
        for idx in indices:
            beta = int(round(grid[idx, 0].item()))
            if beta not in candidates:
                candidates.append(beta)
            if len(candidates) >= batch_size:
                break
    if debug_selection:
        print(f"  TS: candidates selected ({len(candidates)}).", flush=True)

    return candidates


def select_next_candidate(model, bounds, grid, best_y, train_x, batch_size=1):
    """
    Select next candidate(s) using the configured acquisition function.

    Parameters
    ----------
    model : SingleTaskGP
        Fitted GP model.
    bounds : torch.Tensor
        Input bounds, shape (2, d).
    grid : torch.Tensor
        Grid for Thompson sampling.
    best_y : float
        Best observed value so far (for EI/PI).
    batch_size : int
        Number of candidates to select.

    Returns
    -------
    candidates : list
        List of selected beta values.
    """
    if acquisition_function == "thompson":
        return thompson_sampling(model, grid, batch_size, n_samples=ts_num_samples)
    if acquisition_function == "qlognei":
        sampler = SobolQMCNormalSampler(torch.Size([128]))
        objective = GenericMCObjective(lambda Y, X: -Y.squeeze(-1))
        acq = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            sampler=sampler,
            objective=objective,
            prune_baseline=True,
        )
        candidates_list = []
        for _ in range(batch_size):
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )
            beta = int(round(candidate[0, 0].item()))
            candidates_list.append(beta)
        return candidates_list

    # For other acquisition functions, use optimize_acqf
    if acquisition_function == "ei":
        acq = ExpectedImprovement(model=model, best_f=best_y, maximize=False)
    elif acquisition_function == "log_ei":
        acq = LogExpectedImprovement(
            model=model, best_f=best_y, maximize=False)
    elif acquisition_function == "pi":
        acq = ProbabilityOfImprovement(
            model=model, best_f=best_y, maximize=False)
    elif acquisition_function == "ucb":
        acq = UpperConfidenceBound(model=model, beta=ucb_beta, maximize=False)
    else:
        raise ValueError(
            f"Unknown acquisition_function: {acquisition_function}")

    # Optimize acquisition function
    candidates_list = []
    for _ in range(batch_size):
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
        beta = int(round(candidate[0, 0].item()))
        candidates_list.append(beta)

    return candidates_list


# =============================================================================
# BAYESIAN OPTIMIZATION
# =============================================================================

custom_results_dir = os.environ.get("ROBUSTNESS_RESULTS_DIR")
base_results_dir = Path(custom_results_dir) if custom_results_dir else script_dir / "Results" / "mc5"
base_results_dir.mkdir(parents=True, exist_ok=True)

for kernel_type in kernel_list:
    for acquisition_function in acquisition_functions:
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        results_dir = base_results_dir  # Save directly in Results folder
        eval_count = 0

        print("\n" + "="*60)
        print("BAYESIAN OPTIMIZATION (Pre-simulated Data) - Ex1")
        print("="*60)
        print(f"Beta range: [{beta_min}, {beta_max}]")
        print(f"Available MC samples: {Nrep}")
        print(f"Initial samples: {n_initial}")
        print(f"BO iterations: {n_iterations}")
        print(f"Kernel: {kernel_type}")
        print(f"Noise model: {noise_model}")
        print(f"Acquisition function: {acquisition_function}")
        print(
            "Early stopping: posterior beta* variance "
            f"(tol={posterior_var_tol}, streak={posterior_var_streak})")

        # Define bounds in original beta space
        bounds = torch.tensor(
            [[float(beta_min)], [float(beta_max)]], dtype=torch.double)

        # Create grid for Thompson Sampling (in original space)
        grid = torch.linspace(beta_min, beta_max,
                              grid_size).unsqueeze(-1).double()
        grid_np = grid.squeeze().numpy()

        # Storage for results
        records = []  # (iter, beta, mse, dosrom, mse_var)

        # -------------------------------------------------------------------------
        # INITIAL SAMPLING
        # -------------------------------------------------------------------------
        print(f"\nInitial sampling...")

        init_betas = np.linspace(beta_min, beta_max, n_initial).astype(int)
        init_betas = np.unique(init_betas)  # Remove duplicates

        train_x_list = []
        train_y_list = []
        train_yvar_list = []

        for i, beta in enumerate(init_betas):
            mse, dosrom, mse_var = evaluate_beta(beta, eval_seed=seed + i)
            records.append((0, int(beta), mse, dosrom, mse_var))
            train_x_list.append([float(beta)])
            train_y_list.append([mse])
            # Use actual variance from MC samples
            train_yvar_list.append([mse_var])

        print(f"Initial best MSE: {min(r[2] for r in records):.6e}")

        # -------------------------------------------------------------------------
        # BO ITERATIONS
        # -------------------------------------------------------------------------
        print("\nBO iterations...")

        var_tol_streak = 0

        for it in range(1, n_iterations + 1):
            # Prepare training data
            train_x = torch.tensor(train_x_list, dtype=torch.double)
            train_y_raw = torch.tensor(train_y_list, dtype=torch.double)
            # Apply log transform if enabled
            if use_log_transform:
                train_y = torch.log(train_y_raw)
            else:
                train_y = train_y_raw
            if noise_model == "hetero":
                train_yvar = torch.tensor(train_yvar_list, dtype=torch.double)
            else:
                train_yvar = None

            # Build and fit GP model
            model = build_gp_model(train_x, train_y, train_yvar=train_yvar)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Posterior beta* variance for stopping criterion
            with torch.no_grad():
                posterior = model.posterior(grid)
                torch_state = torch.random.get_rng_state()
                var_samples = posterior.rsample(
                    torch.Size([posterior_var_samples])
                ).squeeze().cpu().numpy()
                torch.random.set_rng_state(torch_state)
            if var_samples.ndim == 1:
                var_samples = var_samples[np.newaxis, :]
            opt_idx = np.argmin(var_samples, axis=1)
            opt_beta_samples = grid_np[opt_idx]
            ddof = 1 if opt_beta_samples.size > 1 else 0
            beta_var = float(np.var(opt_beta_samples, ddof=ddof))

            # Select next candidate(s)
            best_y = train_y.min().item()
            try:
                candidates = select_next_candidate(
                    model, bounds, grid, best_y, train_x, batch_size
                )
            except Exception as exc:
                print(f"Iter {it:2d}: candidate selection failed: {exc}; "
                      "falling back to random.", flush=True)
                candidates = []
                while len(candidates) < batch_size:
                    beta = int(rng.integers(beta_min, beta_max + 1))
                    if beta not in candidates:
                        candidates.append(beta)

            # Ensure candidates are within bounds
            final_candidates = []
            for beta in candidates:
                beta = int(np.clip(beta, beta_min, beta_max))
                if beta not in final_candidates:
                    final_candidates.append(beta)
                if len(final_candidates) >= batch_size:
                    break

            # If we need more candidates, sample randomly
            while len(final_candidates) < batch_size:
                beta = int(rng.integers(beta_min, beta_max + 1))
                if beta not in final_candidates:
                    final_candidates.append(beta)

            # Save plot for this iteration
            if enable_plots:
                plot_path = plot_gp_iteration(
                    model, train_x, train_y, bounds, grid, it,
                    final_candidates[0], results_dir, acquisition_function, best_y
                )

            # Evaluate candidates
            for j, beta in enumerate(final_candidates):
                mse, dosrom, mse_var = evaluate_beta(
                    beta, eval_seed=seed + it * 100 + j)
                records.append((it, int(beta), mse, dosrom, mse_var))
                train_x_list.append([float(beta)])
                train_y_list.append([mse])
                # Use actual variance from MC samples
                train_yvar_list.append([mse_var])

            # Report progress
            current_best = min(r[2] for r in records)
            print(f"Iter {it:2d}: beta={final_candidates[0]:3d}, "
                  f"mse={mse:.4e}, best={current_best:.4e}, "
                  f"var_beta={beta_var:.3e}")

            # Early stopping check
            if beta_var <= posterior_var_tol:
                var_tol_streak += 1
            else:
                var_tol_streak = 0

            if var_tol_streak >= posterior_var_streak:
                print(
                    "Stopping early: posterior beta* variance met tol "
                    f"for {posterior_var_streak} consecutive iterations "
                    f"(iter={it}, tol={posterior_var_tol:.3e})."
                )
                break

        # =====================================================================
        # FINAL PLOT - Normalized Objective Function (f/f*)
        # =====================================================================
        train_x = torch.tensor(train_x_list, dtype=torch.double)
        train_y_raw = torch.tensor(train_y_list, dtype=torch.double)
        # Apply log transform if enabled
        if use_log_transform:
            train_y = torch.log(train_y_raw)
        else:
            train_y = train_y_raw
        if noise_model == "hetero":
            train_yvar = torch.tensor(train_yvar_list, dtype=torch.double)
        else:
            train_yvar = None

        model = build_gp_model(train_x, train_y, train_yvar=train_yvar)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        model.eval()
        with torch.no_grad():
            posterior = model.posterior(grid)
            mean_transformed = posterior.mean.squeeze().numpy()
            std_transformed = posterior.variance.sqrt().squeeze().numpy()
            torch_state = torch.random.get_rng_state()
            post_samples = posterior.rsample(torch.Size(
                [final_posterior_samples])).squeeze().cpu().numpy()
            torch.random.set_rng_state(torch_state)

        grid_np = grid.squeeze().numpy()
        train_x_np = train_x.squeeze().numpy()

        # Convert back to original scale if log transform was used
        if use_log_transform:
            # For log-normal: mean = exp(mu + sigma^2/2), but for plotting we use exp(mu)
            mean = np.exp(mean_transformed)
            # For uncertainty, use delta method: std_original ≈ exp(mu) * std_log
            std = mean * std_transformed
            post_samples = np.exp(post_samples)
            train_y_np = train_y_raw.squeeze().numpy()
        else:
            mean = mean_transformed
            std = std_transformed
            train_y_np = train_y.squeeze().numpy()

        # Plotting disabled - use MATLAB for publication plots
        # (local regression computed in MATLAB using local_regression.m)

        # =====================================================================
        # RESULTS
        # =====================================================================
        # Best from GP posterior mean (recommended)
        gp_min_idx = np.argmin(mean)
        # Exact grid value for plotting
        beta_opt_gp_exact = grid_np[gp_min_idx]
        beta_opt_gp = int(round(beta_opt_gp_exact))  # Rounded for reporting
        mse_opt_gp = mean[gp_min_idx]

        # Best observed sample (for comparison)
        best_obs = min(records, key=lambda r: r[2])
        beta_best_obs = best_obs[1]
        mse_best_obs = best_obs[2]

        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Best beta (GP mean): {beta_opt_gp}")
        print(f"Best MSE (GP mean): {mse_opt_gp:.6e}")
        print(f"Best beta (observed): {beta_best_obs}")
        print(f"Best MSE (observed): {mse_best_obs:.6e}")
        print(f"Reference doExp: {doExp:.10e}")
        print(
            f"Total objective evaluations: {eval_count} (x{n_mc_var} MC samples = {eval_count * n_mc_var} total)")
        print(f"\nPlots saved to: {results_dir}")

        # Save results to .mat file for MATLAB plotting
        out_path = results_dir / "bo_ex1_mc5.mat"

        # Prepare data arrays
        iter_arr = np.array([r[0] for r in records])
        beta_arr = np.array([r[1] for r in records])
        mse_arr = np.array([r[2] for r in records])
        s_arr = np.array([r[3] for r in records])
        mse_var_arr = np.array([r[4] for r in records])

        # GP posterior data for plotting
        gp_grid = grid_np
        gp_mean = mean
        gp_std = std
        if post_samples.ndim == 1:
            post_samples = post_samples[np.newaxis, :]
        gp_posterior_samples = post_samples
        gp_posterior_opt_idx = np.argmin(gp_posterior_samples, axis=1)
        gp_posterior_opt_beta = gp_grid[gp_posterior_opt_idx]
        gp_posterior_opt_val = gp_posterior_samples[np.arange(
            gp_posterior_samples.shape[0]), gp_posterior_opt_idx]

        # Save all data to .mat file
        sio.savemat(out_path, {
            # BO iteration data
            'iter': iter_arr,
            'beta_samples': beta_arr,
            'mse_samples': mse_arr,
            's_samples': s_arr,
            'mse_var_samples': mse_var_arr,
            # GP posterior
            'gp_grid': gp_grid,
            'gp_mean': gp_mean,
            'gp_std': gp_std,
            'gp_posterior_samples': gp_posterior_samples,
            'gp_posterior_opt_beta': gp_posterior_opt_beta,
            'gp_posterior_opt_val': gp_posterior_opt_val,
            # Best results from GP mean (recommended)
            'beta_opt_gp': beta_opt_gp,
            'beta_opt_gp_exact': beta_opt_gp_exact,
            'mse_opt_gp': mse_opt_gp,
            # Best observed sample (for comparison)
            'beta_best_obs': beta_best_obs,
            'mse_best_obs': mse_best_obs,
            # Parameters
            'beta_min': beta_min,
            'beta_max': beta_max,
            'doExp': doExp,
            'n_initial': n_initial,
            'n_iterations': n_iterations,
            'kernel_type': kernel_type,
            'acquisition_function': acquisition_function,
            'n_eval': eval_count,
            'n_mc_var': n_mc_var,
            'n_eval_mc': eval_count * n_mc_var,
            'use_log_transform': use_log_transform,
        })

        print(f"Results saved to: {out_path}")
