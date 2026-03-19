#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import minimize_scalar

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from robustness_utils import (
    cumulative_best,
    deterministic_result_path,
    ensure_dir,
    load_truth,
    score_beta,
)


def run_1d_reference(example: str) -> None:
    truth = load_truth(example)
    beta_grid = truth["beta_grid"]
    true_mse = truth["true_mse"]

    beta_evals: list[float] = []

    def objective(beta: float) -> float:
        beta_evals.append(float(beta))
        return float(score_beta(beta, beta_grid, true_mse))

    t_start = time.perf_counter()
    result = minimize_scalar(
        objective,
        bounds=(float(beta_grid.min()), float(beta_grid.max())),
        method="bounded",
        options={"xatol": 0.1},
    )
    wall_time_sec = time.perf_counter() - t_start

    evaluated_beta_seq = np.asarray(beta_evals, dtype=float)
    eval_true_mse_seq = score_beta(evaluated_beta_seq, beta_grid, true_mse)
    best_true_mse_so_far_seq = cumulative_best(eval_true_mse_seq)
    eval_count_seq = np.arange(1, evaluated_beta_seq.size + 1, dtype=float)

    final_recommended_beta = float(np.round(result.x))
    final_recommended_true_mse = float(score_beta(final_recommended_beta, beta_grid, true_mse))
    best_idx = int(np.argmin(eval_true_mse_seq))
    final_best_evaluated_beta = float(evaluated_beta_seq[best_idx])
    final_best_true_mse = float(eval_true_mse_seq[best_idx])

    out_path = deterministic_result_path(example, "1D")
    ensure_dir(out_path.parent)
    sio.savemat(
        out_path,
        {
            "seed": np.nan,
            "method": "1D",
            "example": example,
            "evaluated_beta_seq": evaluated_beta_seq,
            "eval_true_mse_seq": eval_true_mse_seq,
            "best_true_mse_so_far_seq": best_true_mse_so_far_seq,
            "eval_count_seq": eval_count_seq,
            "final_recommended_beta": final_recommended_beta,
            "final_recommended_true_mse": final_recommended_true_mse,
            "final_best_evaluated_beta": final_best_evaluated_beta,
            "final_best_true_mse": final_best_true_mse,
            "stop_eval_count": int(evaluated_beta_seq.size),
            "total_mc_evaluations": int(evaluated_beta_seq.size),
            "n_mc_per_objective": 1,
            "is_deterministic": True,
            "wall_time_sec": wall_time_sec,
        },
    )
    print(f"Saved 1D robustness result: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic 1D robustness reference for one example.")
    parser.add_argument("--example", required=True, choices=["Ex1", "Ex2", "Ex3", "Ex3_sigma03"])
    args = parser.parse_args()
    run_1d_reference(args.example)


if __name__ == "__main__":
    main()
