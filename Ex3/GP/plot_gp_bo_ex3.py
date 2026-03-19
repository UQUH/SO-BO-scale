#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting.") from exc


def main():
    parser = argparse.ArgumentParser(description="Plot GP-BO results (Ex3).")
    parser.add_argument(
        "--results",
        type=str,
        default="Results/bo_ex3.csv",
        help="Path to bo_ex3.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Results",
        help="Directory for plots",
    )
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    results_path = (workdir / args.results).resolve()
    out_dir = (workdir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.genfromtxt(results_path, delimiter=",", names=True)
    iters = data["iter"].astype(int)
    betas = data["beta"].astype(int)
    mse = data["mse"]

    best_idx = np.argmin(mse)
    best_beta = betas[best_idx]
    best_mse = mse[best_idx]

    # Plot 1: MSE vs beta (scatter)
    plt.figure(figsize=(8, 5))
    plt.scatter(betas, mse, s=25, c=iters, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Iteration")
    plt.scatter([best_beta], [best_mse], color="red", s=60, marker="x")
    plt.xlabel("beta")
    plt.ylabel("MSE")
    plt.title("GP-BO samples: MSE vs beta")
    plt.tight_layout()
    out_path = out_dir / "gp_bo_mse_vs_beta.png"
    plt.savefig(out_path, dpi=150)

    # Plot 2: Best-so-far MSE over evaluations
    order = np.argsort(np.arange(len(mse)))
    mse_seq = mse[order]
    best_so_far = np.minimum.accumulate(mse_seq)
    plt.figure(figsize=(8, 5))
    plt.plot(best_so_far, color="#1f77b4", linewidth=2)
    plt.xlabel("Evaluation")
    plt.ylabel("Best-so-far MSE")
    plt.title("GP-BO convergence")
    plt.tight_layout()
    out_path = out_dir / "gp_bo_convergence.png"
    plt.savefig(out_path, dpi=150)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
