#!/usr/bin/env python3
"""
SOTA noisy-BO baseline for Ex3 using qNEI/qLogNEI without explicit replicates.

This script uses one noisy observation per beta evaluation and fits a GP with
learned observation noise. Candidate selection is done with qNEI on the
discrete beta grid.
"""

import os
from pathlib import Path
import time
import warnings

import gpytorch
import numpy as np
import scipy.io as sio
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore", category=RuntimeWarning, module="botorch")
warnings.filterwarnings("ignore", message=".*not p.d.*")
warnings.filterwarnings("ignore", message=".*not contained to the unit cube.*")
warnings.filterwarnings("ignore", message=".*Very small noise values.*")
warnings.filterwarnings("ignore", message=".*Negative variance values.*")
warnings.filterwarnings(
    "ignore",
    message=".*qNoisyExpectedImprovement has known numerical issues.*",
)

# =============================================================================
# USER PARAMETERS
# =============================================================================

seed = int(os.environ.get("ROBUSTNESS_SEED", "12"))
n_initial = 10
n_iterations = 50
batch_size = 10
use_log_transform = True
posterior_beta_var_tol = 0.5
posterior_beta_var_streak = 3

# Acquisition settings
# Available options: "qnei", "qlognei"
acquisition_function = os.environ.get("SOTA_ACQUISITION_FUNCTION", "qlognei")
acq_mc_samples = 128
n_mc_eval = int(os.environ.get("SOTA_N_MC_EVAL", "10"))
posterior_beta_var_samples = 100
final_posterior_samples = 100

# GP hyperparameters
lengthscale_init = 0.3
lengthscale_bounds = (0.05, 1.0)
noise_init = 0.2
noise_bounds = (1e-4, 5.0)

np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

eval_count = 0

print("Loading pre-simulated data...")
t_start = time.perf_counter()

script_dir = Path(__file__).resolve().parent
model_dir = script_dir.parent / "Model"

dist_data = sio.loadmat(model_dir / "dist.mat")
dist_matrix = dist_data["dist"]
beta_values = dist_data["beta"].flatten().astype(float)
do_exp = float(dist_data["doExp"].flatten()[0])

Nrep, num_beta = dist_matrix.shape
beta_min = int(beta_values[0])
beta_max = int(beta_values[-1])
beta_grid_np = beta_values.copy()

print(f"Loaded dist matrix: {Nrep} repetitions x {num_beta} beta values")
print(f"Beta range: [{beta_min}, {beta_max}]")
print(f"Reference value (doExp): {do_exp:.6f}")
print(f"Data loading time: {time.perf_counter() - t_start:.4f}s")


def get_dist_for_beta(beta, rep_idx):
    beta = int(np.clip(round(beta), beta_min, beta_max))
    col_idx = beta - beta_min
    rep_idx = rep_idx % Nrep
    return float(dist_matrix[rep_idx, col_idx])


def evaluate_beta(beta, eval_seed=None):
    global eval_count
    eval_count += 1

    beta = int(np.clip(round(beta), beta_min, beta_max))
    local_rng = np.random.default_rng(eval_seed)
    rep_indices = local_rng.choice(Nrep, size=n_mc_eval, replace=False)
    dosrom_samples = []
    mse_samples = []
    for rep_idx in rep_indices:
        dosrom = get_dist_for_beta(beta, rep_idx)
        dosrom_samples.append(dosrom)
        mse_samples.append((dosrom - do_exp) ** 2)
    mean_mse = float(np.mean(mse_samples))
    mean_dosrom = float(np.mean(dosrom_samples))
    return mean_mse, mean_dosrom


def build_gp_model(train_x, train_y):
    base_kernel = RBFKernel(
        lengthscale_constraint=Interval(
            lengthscale_bounds[0], lengthscale_bounds[1])
    )
    base_kernel.lengthscale = lengthscale_init
    covar_module = ScaleKernel(base_kernel)

    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=covar_module,
        input_transform=Normalize(d=1),
        outcome_transform=Standardize(m=1),
    )
    model.likelihood.noise_covar.noise_constraint = Interval(
        noise_bounds[0], noise_bounds[1]
    )
    model.likelihood.noise = noise_init
    return model


def transform_targets(train_y_raw):
    if use_log_transform:
        return torch.log(train_y_raw.clamp_min(1e-20))
    return train_y_raw


def build_acquisition(model, train_x, pending=None):
    sampler = SobolQMCNormalSampler(torch.Size([acq_mc_samples]))
    objective = GenericMCObjective(lambda Y, X: -Y.squeeze(-1))
    if acquisition_function == "qnei":
        return qNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            sampler=sampler,
            objective=objective,
            X_pending=pending,
            prune_baseline=True,
        )
    if acquisition_function == "qlognei":
        return qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            sampler=sampler,
            objective=objective,
            X_pending=pending,
            prune_baseline=True,
        )
    raise ValueError(
        f"Unsupported acquisition_function: {acquisition_function}")


def select_candidates(model, train_x, beta_grid, n_select):
    candidates = []
    pending = None

    for _ in range(n_select):
        acq = build_acquisition(model, train_x, pending=pending)
        with torch.no_grad():
            acq_vals = acq(beta_grid.unsqueeze(1)).view(-1)

        order = torch.argsort(acq_vals, descending=True)
        chosen_idx = None
        for idx in order.tolist():
            beta = int(round(beta_grid[idx, 0].item()))
            if beta not in candidates:
                chosen_idx = idx
                break

        if chosen_idx is None:
            break

        candidates.append(int(round(beta_grid[chosen_idx, 0].item())))
        next_x = beta_grid[chosen_idx:chosen_idx + 1]
        pending = next_x if pending is None else torch.cat(
            [pending, next_x], dim=0)

    return candidates


def estimate_posterior_beta_var(model, beta_grid, n_samples):
    with torch.no_grad():
        posterior = model.posterior(beta_grid)
        post_samples = posterior.rsample(
            torch.Size([n_samples])).squeeze(-1).cpu().numpy()
    if post_samples.ndim == 1:
        post_samples = post_samples[np.newaxis, :]
    if use_log_transform:
        post_samples = np.exp(post_samples)
    opt_idx = np.argmin(post_samples, axis=1)
    opt_beta = beta_grid.squeeze(-1).cpu().numpy()[opt_idx]
    return float(np.var(opt_beta)), opt_beta


results_tag = f"{acquisition_function}_mc{n_mc_eval}"
custom_results_dir = os.environ.get("ROBUSTNESS_RESULTS_DIR")
results_dir = Path(
    custom_results_dir) if custom_results_dir else script_dir / "Results" / results_tag
results_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("SOTA NOISY BO BASELINE (Ex3)")
print("=" * 60)
print(f"Acquisition function: {acquisition_function}")
print(f"Results tag: {results_tag}")
print(f"Initial samples: {n_initial}")
print(f"BO iterations: {n_iterations}")
if n_mc_eval == 1:
    print("Observation model: single noisy sample per beta (no explicit replicates)")
else:
    print(f"Observation model: mean of {n_mc_eval} noisy samples per beta")
print(
    "Early stopping: posterior beta* variance "
    f"(tol={posterior_beta_var_tol:.6f}, streak={posterior_beta_var_streak})"
)

beta_grid = torch.tensor(beta_grid_np, dtype=torch.double).unsqueeze(-1)
records = []
train_x_list = []
train_y_list = []

print("\nInitial sampling...")
init_betas = np.unique(np.linspace(
    beta_min, beta_max, n_initial).round().astype(int))

for i, beta in enumerate(init_betas):
    mse, dosrom = evaluate_beta(beta, eval_seed=seed + i)
    records.append((0, int(beta), mse, dosrom))
    train_x_list.append([float(beta)])
    train_y_list.append([mse])

print(f"Initial best MSE: {min(r[2] for r in records):.6e}")

print("\nBO iterations...")
posterior_beta_var_streak_count = 0
posterior_beta_var_tol_iter = -1

for it in range(1, n_iterations + 1):
    train_x = torch.tensor(train_x_list, dtype=torch.double)
    train_y_raw = torch.tensor(train_y_list, dtype=torch.double)
    train_y = transform_targets(train_y_raw)

    model = build_gp_model(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    try:
        final_candidates = select_candidates(
            model, train_x, beta_grid, batch_size)
    except Exception as exc:
        print(
            f"Iter {it:2d}: candidate selection failed: {exc}; falling back to random.")
        final_candidates = []

    while len(final_candidates) < batch_size:
        beta = int(rng.integers(beta_min, beta_max + 1))
        if beta not in final_candidates:
            final_candidates.append(beta)

    last_mse = None
    for j, beta in enumerate(final_candidates):
        mse, dosrom = evaluate_beta(beta, eval_seed=seed + it * 100 + j)
        records.append((it, int(beta), mse, dosrom))
        train_x_list.append([float(beta)])
        train_y_list.append([mse])
        last_mse = mse

    stop_train_x = torch.tensor(train_x_list, dtype=torch.double)
    stop_train_y_raw = torch.tensor(train_y_list, dtype=torch.double)
    stop_train_y = transform_targets(stop_train_y_raw)
    stop_model = build_gp_model(stop_train_x, stop_train_y)
    stop_mll = ExactMarginalLogLikelihood(stop_model.likelihood, stop_model)
    fit_gpytorch_mll(stop_mll)
    stop_model.eval()

    current_best = min(r[2] for r in records)
    posterior_beta_var, _ = estimate_posterior_beta_var(
        stop_model, beta_grid, posterior_beta_var_samples
    )
    print(
        f"Iter {it:2d}: beta={final_candidates[0]:3d}, mse={last_mse:.4e}, "
        f"best={current_best:.4e}, posterior_beta_var={posterior_beta_var:.4f}"
    )

    if posterior_beta_var <= posterior_beta_var_tol:
        if posterior_beta_var_tol_iter < 0:
            posterior_beta_var_tol_iter = it
        posterior_beta_var_streak_count += 1
    else:
        posterior_beta_var_streak_count = 0

    if posterior_beta_var_streak_count >= posterior_beta_var_streak:
        print(
            "Stopping early: posterior beta* variance met tol for "
            f"{posterior_beta_var_streak} consecutive iterations "
            f"(var={posterior_beta_var:.4f}, tol={posterior_beta_var_tol:.4f})."
        )
        break

train_x = torch.tensor(train_x_list, dtype=torch.double)
train_y_raw = torch.tensor(train_y_list, dtype=torch.double)
train_y = transform_targets(train_y_raw)

model = build_gp_model(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
model.eval()

with torch.no_grad():
    posterior = model.posterior(beta_grid)
    mean_transformed = posterior.mean.squeeze().cpu().numpy()
    std_transformed = posterior.variance.sqrt().squeeze().cpu().numpy()
    post_samples = posterior.rsample(torch.Size(
        [final_posterior_samples])).squeeze(-1).cpu().numpy()

if use_log_transform:
    gp_mean = np.exp(mean_transformed)
    gp_std = gp_mean * std_transformed
    gp_posterior_samples = np.exp(post_samples)
else:
    gp_mean = mean_transformed
    gp_std = std_transformed
    gp_posterior_samples = post_samples

gp_grid = beta_grid.squeeze(-1).cpu().numpy()
gp_min_idx = int(np.argmin(gp_mean))
beta_opt_gp_exact = float(gp_grid[gp_min_idx])
beta_opt_gp = beta_opt_gp_exact
mse_opt_gp = float(gp_mean[gp_min_idx])
if gp_posterior_samples.ndim == 1:
    gp_posterior_samples = gp_posterior_samples[np.newaxis, :]
gp_posterior_opt_idx = np.argmin(gp_posterior_samples, axis=1)
gp_posterior_opt_beta = gp_grid[gp_posterior_opt_idx]
gp_posterior_opt_val = gp_posterior_samples[
    np.arange(gp_posterior_samples.shape[0]), gp_posterior_opt_idx
]

best_obs = min(records, key=lambda r: r[2])
beta_best_obs = int(best_obs[1])
mse_best_obs = float(best_obs[2])

print("\n" + "=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)
print(f"Best beta (GP mean): {beta_opt_gp:.6f}")
print(f"Best MSE (GP mean): {mse_opt_gp:.6e}")
print(f"Best beta (observed): {beta_best_obs}")
print(f"Best MSE (observed): {mse_best_obs:.6e}")
print(f"Reference doExp: {do_exp:.6f}")
print(
    f"Total objective evaluations: {eval_count} "
    f"(x{n_mc_eval} MC samples = {eval_count * n_mc_eval} total)"
)

out_path = results_dir / f"bo_ex3_{results_tag}.mat"
sio.savemat(
    out_path,
    {
        "iter": np.array([r[0] for r in records]),
        "beta_samples": np.array([r[1] for r in records]),
        "mse_samples": np.array([r[2] for r in records]),
        "s_samples": np.array([r[3] for r in records]),
        "gp_grid": gp_grid,
        "gp_mean": gp_mean,
        "gp_std": gp_std,
        "gp_posterior_samples": gp_posterior_samples,
        "gp_posterior_opt_beta": gp_posterior_opt_beta,
        "gp_posterior_opt_val": gp_posterior_opt_val,
        "beta_opt_gp": beta_opt_gp,
        "beta_opt_gp_exact": beta_opt_gp_exact,
        "mse_opt_gp": mse_opt_gp,
        "beta_best_obs": beta_best_obs,
        "mse_best_obs": mse_best_obs,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "doExp": do_exp,
        "n_initial": n_initial,
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "use_log_transform": use_log_transform,
        "acquisition_function": acquisition_function,
        "results_tag": results_tag,
        "objective_evaluations": eval_count,
        "n_mc_eval": n_mc_eval,
        "n_eval_mc": eval_count * n_mc_eval,
        "final_posterior_samples": final_posterior_samples,
        "posterior_beta_var_tol": posterior_beta_var_tol,
        "posterior_beta_var_streak": posterior_beta_var_streak,
        "posterior_beta_var_tol_iter": posterior_beta_var_tol_iter,
    },
)

print(f"Results saved to: {out_path}")
