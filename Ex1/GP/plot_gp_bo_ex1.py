#!/usr/bin/env python3
"""
Plot GP-BO results for Ex1 (Linear Static Problem).

Creates:
1. Normalized objective function plot (f/f*) with local regression comparison
2. MSE vs beta scatter plot colored by iteration
3. Convergence plot (best-so-far MSE)

Usage:
    python plot_gp_bo_ex1.py
    python plot_gp_bo_ex1.py --results Results/rbf/thompson/bo_ex1.csv
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from statsmodels.nonparametric.smoothers_lowess import lowess

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting.") from exc


def main():
    parser = argparse.ArgumentParser(description="Plot GP-BO results (Ex1).")
    parser.add_argument(
        "--results",
        type=str,
        default="Results/rbf/thompson/bo_ex1.csv",
        help="Path to bo_ex1.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for plots (default: same as results file)",
    )
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    results_path = (workdir / args.results).resolve()

    if args.out_dir is None:
        out_dir = results_path.parent
    else:
        out_dir = (workdir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD PRE-SIMULATED DATA (for local regression comparison)
    # =========================================================================
    model_dir = workdir.parent / "Model"
    dist_data = sio.loadmat(model_dir / "dist.mat")
    dist_matrix = dist_data["dist"]  # Shape: (1000, 143)

    Nrep, num_Beta = dist_matrix.shape

    # Ex1 parameters
    k = 8  # ROM dimension (minimum beta value)
    doExp = 3.456255510150231e-04  # Reference distance

    beta_min = k
    beta_max = k + num_Beta - 1
    beta_range = np.arange(beta_min, beta_max + 1)
    log_beta_range = np.log(beta_range)

    # MSE per beta: mean of (dosrom - doExp)^2 across all reps
    mse_all_data = np.mean((dist_matrix - doExp)**2, axis=0)

    # Local regression for MSE vs log(beta)
    frac_loess_mse = 0.2
    mse_local = lowess(mse_all_data, log_beta_range,
                       frac=frac_loess_mse, return_sorted=False)

    # Find optimal beta from local regression
    mse_local_min = np.min(mse_local)
    beta_opt_local_idx = np.argmin(mse_local)
    beta_opt_local = beta_range[beta_opt_local_idx]

    # Find beta range within tolerance (10%)
    tol_f = 0.1
    beta_opt_range_mask = mse_local / mse_local_min < (1 + tol_f)
    beta_opt_range = beta_range[beta_opt_range_mask]
    beta_opt_range_min = beta_opt_range[0] if len(
        beta_opt_range) > 0 else beta_opt_local
    beta_opt_range_max = beta_opt_range[-1] if len(
        beta_opt_range) > 0 else beta_opt_local

    # Scale factor for normalization
    mse_scale = mse_local_min

    # =========================================================================
    # LOAD BO RESULTS
    # =========================================================================
    data = np.genfromtxt(results_path, delimiter=",", names=True)
    iters = data["iter"].astype(int)
    betas = data["beta"].astype(int)
    mse = data["mse"]

    best_idx = np.argmin(mse)
    best_beta = betas[best_idx]
    best_mse = mse[best_idx]

    print(f"Loaded {len(mse)} samples from {results_path}")
    print(f"Best beta (GP): {best_beta}, MSE: {best_mse:.6e}")
    print(f"Best beta (local): {beta_opt_local}, MSE: {mse_local_min:.6e}")

    # =========================================================================
    # Plot 1: Normalized Objective Function (f/f*)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot optimal region (shaded)
    ax.axvspan(beta_opt_range_min, beta_opt_range_max, alpha=0.1, color='gray',
               label=f'Optimal region (±{tol_f*100:.0f}%)')

    # Plot local regression (normalized)
    ax.plot(beta_range, mse_local / mse_scale, 'r-', linewidth=2,
            label='Local regression')

    # Plot observations (normalized)
    ax.scatter(betas, mse / mse_scale, c='k', s=30, zorder=5,
               alpha=0.5, label=f'BO samples (N={len(betas)})')

    # Mark best observed point
    ax.scatter(best_beta, best_mse / mse_scale,
               c='blue', s=150, marker='x', zorder=6, linewidths=2,
               label=f'β* GP: {best_beta}')

    # Mark local regression optimum
    ax.scatter(beta_opt_local, mse_local_min / mse_scale,
               c='red', s=150, marker='x', zorder=6, linewidths=2,
               label=f'β* local: {beta_opt_local}')

    # Vertical line at local optimum
    ax.axvline(x=beta_opt_local, color='r',
               linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('β', fontsize=14)
    ax.set_ylabel('f / f*', fontsize=14)
    ax.set_title('Normalized Objective Function (Ex1)', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(beta_min, beta_max)
    ax.set_ylim(0, 8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "gp_bo_normalized.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # =========================================================================
    # Plot 2: MSE vs beta (scatter colored by iteration)
    # =========================================================================
    plt.figure(figsize=(8, 5))
    plt.scatter(betas, mse, s=25, c=iters, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Iteration")
    plt.scatter([best_beta], [best_mse], color="red", s=60, marker="x",
                label=f"Best: β={best_beta}")
    plt.xlabel("β", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title("GP-BO samples: MSE vs β (Ex1)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "gp_bo_mse_vs_beta.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # =========================================================================
    # Plot 3: Best-so-far MSE over evaluations
    # =========================================================================
    best_so_far = np.minimum.accumulate(mse)

    plt.figure(figsize=(8, 5))
    plt.plot(best_so_far, color="#1f77b4", linewidth=2)
    plt.axhline(y=mse_local_min, color='r', linestyle='--', linewidth=1,
                label=f'Local opt: {mse_local_min:.2e}')
    plt.xlabel("Evaluation", fontsize=12)
    plt.ylabel("Best-so-far MSE", fontsize=12)
    plt.title("GP-BO Convergence (Ex1)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "gp_bo_convergence.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # =========================================================================
    # Plot 4: Normalized convergence (f/f*)
    # =========================================================================
    best_so_far_norm = best_so_far / mse_scale

    plt.figure(figsize=(8, 5))
    plt.plot(best_so_far_norm, color="#1f77b4", linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--',
                linewidth=1, label='f* (local opt)')
    plt.xlabel("Evaluation", fontsize=12)
    plt.ylabel("Best-so-far f / f*", fontsize=12)
    plt.title("GP-BO Normalized Convergence (Ex1)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "gp_bo_convergence_norm.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()
